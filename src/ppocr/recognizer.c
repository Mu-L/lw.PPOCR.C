#include "lw_infer.h"
#include "abi_compat_internal.h"

/* Public REC handle: preprocessing -> graph execution -> UTF-8 CTC decoding. */

#include "error_internal.h"
#include "executor_internal.h"
#include "model_internal.h"
#include "profile_internal.h"
#include "rec_internal.h"
#include "session_internal.h"
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#include "x64_rec_backend_internal.h"
#include "x64_rec_fast_internal.h"
#endif

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#include <stdio.h>
#endif
#if defined(_MSC_VER)
#include <malloc.h>
#endif
#include <string.h>

#define LW_REC_DEFAULT_TARGET_WIDTH 960u
#define LW_REC_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)
#define LW_REC_RESIDENT_WIDTH_COUNT 5u
#define LW_MEDIUM_REC_LWM_CHECKSUM UINT64_C(0x5c1ad5136616c165)
static const uint32_t lw_rec_adaptive_widths[LW_REC_RESIDENT_WIDTH_COUNT] = {
    192u, 320u, 480u, 640u, 960u
};

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
typedef struct lw_x64_rec_backend_slot {
    uint32_t target_width;
    lw_x64_rec_program* program;
    lw_x64_rec_instance* instance;
} lw_x64_rec_backend_slot;
typedef struct lw_medium_fast_shared {
    uint32_t reference_count;
    lw_x64_rec_fast_plan* packed_template;
} lw_medium_fast_shared;
#endif

typedef struct lw_rec_resident_slot {
    lw_session* session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan* fast_plan;
    uint8_t fast_plan_disabled;
#endif
    float* input;
    float* probabilities;
    uint32_t* best_indices;
    float* best_probabilities;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    uint32_t target_width;
    uint32_t time_steps;
} lw_rec_resident_slot;

struct lw_recognizer {
    lw_model* model;
    lw_session* session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan* fast_plan;
    lw_medium_fast_shared* fast_shared;
    uint8_t fast_plan_disabled;
    uint8_t fast_backend_disabled;
#endif
    lw_rec_dictionary* dictionary;
    float* input;
    float* probabilities;
    uint32_t* best_indices;
    float* best_probabilities;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    lw_session* cached_session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan* cached_fast_plan;
    uint8_t cached_fast_plan_disabled;
#endif
    float* cached_input;
    float* cached_probabilities;
    uint32_t* cached_best_indices;
    float* cached_best_probabilities;
    uint64_t cached_input_element_count;
    uint64_t cached_probability_element_count;
    uint32_t cached_target_width;
    uint32_t cached_time_steps;
    uint64_t max_image_pixels;
    lw_session_options session_options;
    uint32_t current_target_width;
    uint32_t current_time_steps;
    uint32_t adaptive_width_enabled;
    uint8_t resident_widths_enabled;
    lw_rec_resident_slot resident_slots[LW_REC_RESIDENT_WIDTH_COUNT];
    /* Optional intra-op pool shared by every width slot/session of this
     * recognizer.  A recognizer serves one line at a time (line workers hold
     * clones), so the pool is never run concurrently. */
    lw_thread_pool* intra_pool;
    uint32_t intra_workers;
    lw_recognizer_info info;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_backend_slot x64_slots[LW_REC_RESIDENT_WIDTH_COUNT];
    /* One shared workspace borrowed by every x64 slot instance: slots of a
     * recognizer never run concurrently (one line at a time; other threads
     * use clones with their own block), so the arena/scratch/CTC buffers are
     * sized to the largest compiled width instead of summed over all five. */
    uint8_t* x64_shared_arena;
    uint8_t* x64_shared_scratch;
    float* x64_shared_ctc_scores;
    uint32_t* x64_shared_best_indices;
    float* x64_shared_best_probabilities;
#endif
};

/* Keep allocation checks width-neutral so bounded uint32_t dimensions do not
 * trigger GCC's -Wtype-limits on 64-bit builds. */
static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
static void medium_fast_shared_release(lw_medium_fast_shared* shared) {
    if (shared == NULL) return;
    if (--shared->reference_count == 0u) {
        lw_x64_rec_fast_plan_free(shared->packed_template);
        free(shared);
    }
}
#endif

static void release_cached_session(lw_recognizer* recognizer) {
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan_free(recognizer->cached_fast_plan);
    recognizer->cached_fast_plan = NULL;
    recognizer->cached_fast_plan_disabled = 0u;
#endif
    free(recognizer->cached_best_probabilities);
    free(recognizer->cached_best_indices);
    free(recognizer->cached_probabilities);
    free(recognizer->cached_input);
    lw_session_free(recognizer->cached_session);
    recognizer->cached_session = NULL;
    recognizer->cached_input = NULL;
    recognizer->cached_probabilities = NULL;
    recognizer->cached_best_indices = NULL;
    recognizer->cached_best_probabilities = NULL;
    recognizer->cached_input_element_count = 0u;
    recognizer->cached_probability_element_count = 0u;
    recognizer->cached_target_width = 0u;
    recognizer->cached_time_steps = 0u;
}

static void release_resident_slot(lw_rec_resident_slot* slot) {
    if (slot == NULL) {
        return;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan_free(slot->fast_plan);
#endif
    free(slot->best_probabilities);
    free(slot->best_indices);
    free(slot->probabilities);
    free(slot->input);
    lw_session_free(slot->session);
    memset(slot, 0, sizeof(*slot));
}

static void swap_active_with_resident_slot(lw_recognizer* recognizer,
                                            lw_rec_resident_slot* slot) {
    lw_session* session = recognizer->session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan* fast_plan = recognizer->fast_plan;
    uint8_t fast_plan_disabled = recognizer->fast_plan_disabled;
#endif
    float* input = recognizer->input;
    float* probabilities = recognizer->probabilities;
    uint32_t* best_indices = recognizer->best_indices;
    float* best_probabilities = recognizer->best_probabilities;
    uint64_t input_element_count = recognizer->input_element_count;
    uint64_t probability_element_count = recognizer->probability_element_count;
    uint32_t target_width = recognizer->current_target_width;
    uint32_t time_steps = recognizer->current_time_steps;

    recognizer->session = slot->session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    recognizer->fast_plan = slot->fast_plan;
    recognizer->fast_plan_disabled = slot->fast_plan_disabled;
#endif
    recognizer->input = slot->input;
    recognizer->probabilities = slot->probabilities;
    recognizer->best_indices = slot->best_indices;
    recognizer->best_probabilities = slot->best_probabilities;
    recognizer->input_element_count = slot->input_element_count;
    recognizer->probability_element_count = slot->probability_element_count;
    recognizer->current_target_width = slot->target_width;
    recognizer->current_time_steps = slot->time_steps;

    slot->session = session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    slot->fast_plan = fast_plan;
    slot->fast_plan_disabled = fast_plan_disabled;
#endif
    slot->input = input;
    slot->probabilities = probabilities;
    slot->best_indices = best_indices;
    slot->best_probabilities = best_probabilities;
    slot->input_element_count = input_element_count;
    slot->probability_element_count = probability_element_count;
    slot->target_width = target_width;
    slot->time_steps = time_steps;
}
static void activate_cached_session(lw_recognizer* recognizer) {
    lw_session* session = recognizer->session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan* fast_plan = recognizer->fast_plan;
    uint8_t fast_plan_disabled = recognizer->fast_plan_disabled;
#endif
    float* input = recognizer->input;
    float* probabilities = recognizer->probabilities;
    uint32_t* best_indices = recognizer->best_indices;
    float* best_probabilities = recognizer->best_probabilities;
    uint64_t input_element_count = recognizer->input_element_count;
    uint64_t probability_element_count = recognizer->probability_element_count;
    uint32_t target_width = recognizer->current_target_width;
    uint32_t time_steps = recognizer->current_time_steps;

    recognizer->session = recognizer->cached_session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    recognizer->fast_plan = recognizer->cached_fast_plan;
    recognizer->fast_plan_disabled = recognizer->cached_fast_plan_disabled;
#endif
    recognizer->input = recognizer->cached_input;
    recognizer->probabilities = recognizer->cached_probabilities;
    recognizer->best_indices = recognizer->cached_best_indices;
    recognizer->best_probabilities = recognizer->cached_best_probabilities;
    recognizer->input_element_count = recognizer->cached_input_element_count;
    recognizer->probability_element_count = recognizer->cached_probability_element_count;
    recognizer->current_target_width = recognizer->cached_target_width;
    recognizer->current_time_steps = recognizer->cached_time_steps;

    recognizer->cached_session = session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    recognizer->cached_fast_plan = fast_plan;
    recognizer->cached_fast_plan_disabled = fast_plan_disabled;
#endif
    recognizer->cached_input = input;
    recognizer->cached_probabilities = probabilities;
    recognizer->cached_best_indices = best_indices;
    recognizer->cached_best_probabilities = best_probabilities;
    recognizer->cached_input_element_count = input_element_count;
    recognizer->cached_probability_element_count = probability_element_count;
    recognizer->cached_target_width = target_width;
    recognizer->cached_time_steps = time_steps;
}

static void clear_result(lw_recognition_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_recognizer_options_init(lw_recognizer_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL) {
        return;
    }
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->target_width = LW_REC_DEFAULT_TARGET_WIDTH;
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_REC_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_recognizer_info_init(lw_recognizer_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_recognition_result_init(lw_recognition_result* result) {
    if (result == NULL) {
        return;
    }
    clear_result(result);
}

static lw_status validate_options(const lw_recognizer_options* options, uint32_t* target_width,
                                  uint64_t* max_image_pixels, lw_model_options* model_options,
                                  lw_session_options* session_options, lw_error* error) {
    lw_recognizer_options defaults;
    lw_recognizer_options values;
    lw_recognizer_options_init(&defaults);
    values = defaults;
    *target_width = defaults.target_width;
    *max_image_pixels = defaults.max_image_pixels;
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (options == NULL) {
        return LW_STATUS_OK;
    }
    if (!lw_abi_copy_input_prefix(&values, sizeof(values), options)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid recognizer options structure");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    values.struct_size = (uint32_t)sizeof(values);
    if (values.reserved0 != 0u || values.reserved1 != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid recognizer options structure");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (values.target_width != 0u) {
        *target_width = values.target_width;
    }
    if (values.max_model_file_size != 0u) {
        model_options->max_file_size = values.max_model_file_size;
    }
    if (values.max_workspace_size != 0u) {
        session_options->max_workspace_size = values.max_workspace_size;
    }
    if (values.max_tensor_size != 0u) {
        session_options->max_tensor_size = values.max_tensor_size;
    }
    if (values.max_image_pixels != 0u) {
        *max_image_pixels = values.max_image_pixels;
    }
    if (*target_width > INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "recognizer target width exceeds the tensor ABI");
        return LW_STATUS_INVALID_SHAPE;
    }
    return LW_STATUS_OK;
}

static lw_status configure_session(lw_recognizer* recognizer, uint32_t target_width,
                                   lw_session_info* configured_info, lw_error* error) {
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session_info session_info;
    lw_session* session = NULL;
    float* input = NULL;
    float* probabilities = NULL;
    uint32_t* best_indices = NULL;
    float* best_probabilities = NULL;
    uint64_t input_element_count;
    uint64_t probability_element_count;
    uint32_t time_steps;
    uint32_t class_count;
    int use_ctc_greedy;
    lw_status status;

    if (recognizer->resident_widths_enabled != 0u) {
        uint32_t slot_index;
        if (recognizer->current_target_width == target_width) {
            if (configured_info != NULL) {
                lw_session_info_init(configured_info);
                status = lw_session_get_info(recognizer->session, configured_info);
                if (status != LW_STATUS_OK) {
                    lw_set_error(error, status, "unable to read resident recognizer session information");
                    return status;
                }
            }
            return LW_STATUS_OK;
        }
        for (slot_index = 0u; slot_index < LW_REC_RESIDENT_WIDTH_COUNT; ++slot_index) {
            if (recognizer->resident_slots[slot_index].target_width == target_width) {
                swap_active_with_resident_slot(recognizer, &recognizer->resident_slots[slot_index]);
                if (configured_info != NULL) {
                    lw_session_info_init(configured_info);
                    status = lw_session_get_info(recognizer->session, configured_info);
                    if (status != LW_STATUS_OK) {
                        lw_set_error(error, status, "unable to read resident recognizer session information");
                        return status;
                    }
                }
                return LW_STATUS_OK;
            }
        }
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "requested REC width is not resident");
        return LW_STATUS_INVALID_ARGUMENT;
    }

    if (recognizer->cached_session != NULL && recognizer->cached_target_width == target_width) {
        activate_cached_session(recognizer);
        if (configured_info != NULL) {
            lw_session_info_init(configured_info);
            status = lw_session_get_info(recognizer->session, configured_info);
            if (status != LW_STATUS_OK) {
                activate_cached_session(recognizer);
                lw_set_error(error, status, "unable to read recognizer session information");
                return status;
            }
        }
        return LW_STATUS_OK;
    }

    /* Keep at most two concrete widths. Discarding an inactive cache entry
     * before construction bounds peak memory to the old active and candidate
     * sessions; the active session remains valid if construction fails. */
    release_cached_session(recognizer);

    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_REC_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)target_width;
    status = lw_session_create_with_prepared_source(
        recognizer->model, &input_desc, 1u, &recognizer->session_options, 0u,
        recognizer->session, &session, error);
    if (status != LW_STATUS_OK) {
        return status;
    }

    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 3u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] <= 0 ||
        output_desc.dimensions[2] <= 0 ||
        (uint32_t)output_desc.dimensions[2] !=
            lw_rec_dictionary_class_count(recognizer->dictionary)) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "recognizer model output and dictionary are incompatible");
        return LW_STATUS_INVALID_SHAPE;
    }

    input_element_count = (uint64_t)3u * LW_REC_INPUT_HEIGHT * target_width;
    probability_element_count =
        (uint64_t)(uint32_t)output_desc.dimensions[1] * (uint32_t)output_desc.dimensions[2];
    time_steps = (uint32_t)output_desc.dimensions[1];
    class_count = (uint32_t)output_desc.dimensions[2];
    /* A terminal, last-axis Softmax can be consumed as greedy CTC classes.
     * Keep the full probability buffer only for models that need the generic
     * graph-output contract; official REC models use two values per step. */
    use_ctc_greedy = lw_session_supports_ctc_greedy_f32(session, time_steps, class_count);
#if defined(LW_EXPERIMENTAL_CTC_TILED)
    if (use_ctc_greedy) {
        /* Recreate the session with the internal physical CTC plan selected at
         * construction time.  The public session ABI remains generic; only
         * the recognizer's private greedy path may elide terminal Softmax. */
        lw_session_free(session);
        session = NULL;
        status = lw_session_create_with_prepared_source(
            recognizer->model, &input_desc, 1u, &recognizer->session_options,
            LW_SESSION_PLAN_CTC_GREEDY, recognizer->session, &session, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
    }
#endif
    if (!allocation_fits(input_element_count, sizeof(*input)) ||
        !allocation_fits(probability_element_count, sizeof(*probabilities)) ||
        !allocation_fits(time_steps, sizeof(*best_indices)) ||
        !allocation_fits(time_steps, sizeof(*best_probabilities))) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer buffer size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    input = (float*)malloc((size_t)input_element_count * sizeof(*input));
    if (use_ctc_greedy) {
        best_indices = (uint32_t*)malloc((size_t)time_steps * sizeof(*best_indices));
        best_probabilities = (float*)malloc((size_t)time_steps * sizeof(*best_probabilities));
    } else {
        probabilities = (float*)malloc((size_t)probability_element_count * sizeof(*probabilities));
    }
    if (input == NULL || (use_ctc_greedy && (best_indices == NULL || best_probabilities == NULL)) ||
        (!use_ctc_greedy && probabilities == NULL)) {
        free(best_probabilities);
        free(best_indices);
        free(probabilities);
        free(input);
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer buffers");
        return LW_STATUS_OUT_OF_MEMORY;
    }

    lw_session_info_init(&session_info);
    status = lw_session_get_info(session, &session_info);
    if (status != LW_STATUS_OK) {
        free(best_probabilities);
        free(best_indices);
        free(probabilities);
        free(input);
        lw_session_free(session);
        lw_set_error(error, status, "unable to read recognizer session information");
        return status;
    }

    /* Publish the new shape only after every allocation and validation has
     * succeeded. A failed adaptive switch therefore leaves the old session
     * usable and avoids a partially configured recognizer. */
    recognizer->cached_session = recognizer->session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    recognizer->cached_fast_plan = recognizer->fast_plan;
    recognizer->cached_fast_plan_disabled = recognizer->fast_plan_disabled;
    recognizer->fast_plan = NULL;
    recognizer->fast_plan_disabled = 0u;
#endif
    recognizer->cached_input = recognizer->input;
    recognizer->cached_probabilities = recognizer->probabilities;
    recognizer->cached_best_indices = recognizer->best_indices;
    recognizer->cached_best_probabilities = recognizer->best_probabilities;
    recognizer->cached_input_element_count = recognizer->input_element_count;
    recognizer->cached_probability_element_count = recognizer->probability_element_count;
    recognizer->cached_target_width = recognizer->current_target_width;
    recognizer->cached_time_steps = recognizer->current_time_steps;
    recognizer->session = session;
    lw_session_set_external_thread_pool(session, recognizer->intra_pool,
                                        recognizer->intra_workers);
    recognizer->input = input;
    recognizer->probabilities = probabilities;
    recognizer->best_indices = best_indices;
    recognizer->best_probabilities = best_probabilities;
    recognizer->input_element_count = input_element_count;
    recognizer->probability_element_count = probability_element_count;
    recognizer->current_target_width = target_width;
    recognizer->current_time_steps = time_steps;
    if (configured_info != NULL) {
        *configured_info = session_info;
    }
    return LW_STATUS_OK;
}

static uint32_t adaptive_target_width(uint32_t source_width, uint32_t source_height,
                                      uint32_t maximum_width) {
    uint64_t scaled_width;
    uint32_t index;
    if (source_width == 0u || source_height == 0u || maximum_width <= 320u) {
        return maximum_width;
    }
    scaled_width =
        ((uint64_t)LW_REC_INPUT_HEIGHT * source_width + source_height - 1u) / source_height;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        if (lw_rec_adaptive_widths[index] >= maximum_width) {
            break;
        }
        if (scaled_width <= lw_rec_adaptive_widths[index]) {
            return lw_rec_adaptive_widths[index];
        }
    }
    return maximum_width;
}

lw_status lw_recognizer_enable_resident_widths(lw_recognizer* recognizer, lw_error* error) {
    uint32_t index;
    if (recognizer == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "recognizer is required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (recognizer->resident_widths_enabled != 0u) {
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
    if (recognizer->info.target_width != 960u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "resident REC widths require a 960-pixel maximum width");
        return LW_STATUS_UNSUPPORTED;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    /* Fully compiled adaptive widths do not need five canonical sessions. */
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        if (recognizer->x64_slots[index].instance == NULL) break;
    }
    if (index == LW_REC_RESIDENT_WIDTH_COUNT) {
        lw_set_error(error, LW_STATUS_OK, "");
        return LW_STATUS_OK;
    }
#endif
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_recognizer temporary = *recognizer;
        lw_session_info session_info;
        lw_status status;
        lw_rec_resident_slot* slot = &recognizer->resident_slots[index];
        if (lw_rec_adaptive_widths[index] == recognizer->current_target_width) {
            continue;
        }
        memset(slot, 0, sizeof(*slot));
        temporary.session = NULL;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        temporary.fast_plan = NULL;
        temporary.cached_fast_plan = NULL;
        temporary.fast_plan_disabled = 0u;
        temporary.cached_fast_plan_disabled = 0u;
#endif
        temporary.input = NULL;
        temporary.probabilities = NULL;
        temporary.best_indices = NULL;
        temporary.best_probabilities = NULL;
        temporary.input_element_count = 0u;
        temporary.probability_element_count = 0u;
        temporary.cached_session = NULL;
        temporary.cached_input = NULL;
        temporary.cached_probabilities = NULL;
        temporary.cached_best_indices = NULL;
        temporary.cached_best_probabilities = NULL;
        temporary.cached_input_element_count = 0u;
        temporary.cached_probability_element_count = 0u;
        temporary.cached_target_width = 0u;
        temporary.cached_time_steps = 0u;
        temporary.current_target_width = 0u;
        temporary.current_time_steps = 0u;
        temporary.resident_widths_enabled = 0u;
        /* Keep the active session as a prepared-source reference while the
         * temporary recognizer constructs this resident width.  The active
         * session remains owned by the real recognizer; the temporary copy
         * only publishes the newly created slot below. */
        temporary.session = recognizer->session;
        temporary.current_target_width = recognizer->current_target_width;
        lw_session_info_init(&session_info);
        status = configure_session(&temporary, lw_rec_adaptive_widths[index], &session_info, error);
        if (status != LW_STATUS_OK) {
            uint32_t cleanup_index;
            for (cleanup_index = 0u; cleanup_index < LW_REC_RESIDENT_WIDTH_COUNT; ++cleanup_index) {
                release_resident_slot(&recognizer->resident_slots[cleanup_index]);
            }
            return status;
        }
        slot->session = temporary.session;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        slot->fast_plan = temporary.fast_plan;
        slot->fast_plan_disabled = temporary.fast_plan_disabled;
#endif
        slot->input = temporary.input;
        slot->probabilities = temporary.probabilities;
        slot->best_indices = temporary.best_indices;
        slot->best_probabilities = temporary.best_probabilities;
        slot->input_element_count = temporary.input_element_count;
        slot->probability_element_count = temporary.probability_element_count;
        slot->target_width = temporary.current_target_width;
        slot->time_steps = temporary.current_time_steps;
    }
    recognizer->resident_widths_enabled = 1u;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
lw_status lw_recognizer_enable_adaptive_width(lw_recognizer* recognizer, uint32_t enabled,
                                              lw_error* error) {
    if (recognizer == NULL || enabled > 1u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "recognizer and a boolean adaptive-width flag are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    recognizer->adaptive_width_enabled = enabled;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

uint32_t lw_recognizer_target_width_for_image(const lw_recognizer* recognizer,
                                              uint32_t source_width, uint32_t source_height) {
    if (recognizer == NULL) {
        return 0u;
    }
    if (recognizer->adaptive_width_enabled == 0u) {
        return recognizer->info.target_width;
    }
    return adaptive_target_width(source_width, source_height, recognizer->info.target_width);
}

uint32_t lw_recognizer_current_target_width(const lw_recognizer* recognizer) {
    return recognizer == NULL ? 0u : recognizer->current_target_width;
}

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
static void recognizer_try_backends(lw_recognizer* recognizer);
static void recognizer_release_backends(lw_recognizer* recognizer);
#if !defined(LW_EXPERIMENTAL_CTC_TILED)
static int recognizer_has_full_compiled_coverage(const lw_recognizer* recognizer) {
    uint32_t index;
    int target_is_standard = 0;
    if (recognizer == NULL) return 0;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        const uint32_t width = lw_rec_adaptive_widths[index];
        const lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        if (width == recognizer->info.target_width) target_is_standard = 1;
        if (width <= recognizer->info.target_width &&
            (slot->target_width != width || slot->program == NULL || slot->instance == NULL))
            return 0;
    }
    return target_is_standard;
}
#endif

static void recognizer_report_compiled_memory(const lw_recognizer* recognizer) {
    const char* enabled = getenv("LW_REC_MEMORY_PROFILE");
    uint32_t index;
    uint64_t arena_bytes = 0u;
    uint64_t scratch_bytes = 0u;
    uint64_t ctc_rows = 0u;
    if (enabled == NULL || enabled[0] != '1' || enabled[1] != '\0') return;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        const lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        if (slot->program == NULL) continue;
        if (slot->program->arena_bytes > arena_bytes) arena_bytes = slot->program->arena_bytes;
        if (slot->program->scratch_bytes > scratch_bytes) scratch_bytes = slot->program->scratch_bytes;
        if (slot->program->ctc.rows > ctc_rows) ctc_rows = slot->program->ctc.rows;
        fprintf(stderr, "REC_MEMORY width=%u owned=%llu borrowed=%llu borrowed_count=%u\n",
                slot->target_width,
                (unsigned long long)slot->program->owned_constant_bytes,
                (unsigned long long)slot->program->borrowed_constant_bytes,
                slot->program->borrowed_constant_count);
    }
    fprintf(stderr, "REC_MEMORY shared_arena=%llu shared_scratch=%llu ctc_workspace=%llu medium_fast_shared=%u\n",
            (unsigned long long)(recognizer->x64_shared_arena == NULL ? 0u : arena_bytes),
            (unsigned long long)(recognizer->x64_shared_scratch == NULL ? 0u : scratch_bytes),
            (unsigned long long)(recognizer->x64_shared_ctc_scores == NULL ? 0u :
                ctc_rows * (sizeof(float) * 2u + sizeof(uint32_t))),
            recognizer->fast_shared == NULL ? 0u : 1u);
}
#endif

lw_status lw_recognizer_create(const char* model_path_utf8, const char* dictionary_path_utf8,
                               const lw_recognizer_options* options, lw_recognizer** out_recognizer,
                               lw_error* error) {
    lw_model_options model_options;
    lw_session_options session_options;
    lw_session_info session_info;
    lw_recognizer* recognizer = NULL;
    uint32_t target_width;
    uint64_t max_image_pixels;
    uint32_t max_label_bytes;
    lw_status status;
    if (out_recognizer != NULL) {
        *out_recognizer = NULL;
    }
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' || dictionary_path_utf8 == NULL ||
        dictionary_path_utf8[0] == '\0' || out_recognizer == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path, dictionary path, and output recognizer are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &target_width, &max_image_pixels, &model_options,
                              &session_options, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    recognizer = (lw_recognizer*)calloc(1u, sizeof(*recognizer));
    if (recognizer == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    recognizer->max_image_pixels = max_image_pixels;
    status = lw_model_load(model_path_utf8, &model_options, &recognizer->model, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    status = lw_rec_dictionary_load(dictionary_path_utf8, &recognizer->dictionary, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    recognizer->session_options = session_options;
    status = configure_session(recognizer, target_width, &session_info, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    max_label_bytes = lw_rec_dictionary_max_label_byte_count(recognizer->dictionary);
    if (max_label_bytes == 0u ||
        (uint64_t)recognizer->current_time_steps > (UINT64_MAX - 1u) / max_label_bytes) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "recognizer maximum text capacity overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    lw_recognizer_info_init(&recognizer->info);
    recognizer->info.target_width = target_width;
    recognizer->info.input_height = LW_REC_INPUT_HEIGHT;
    recognizer->info.time_steps = recognizer->current_time_steps;
    recognizer->info.class_count = lw_rec_dictionary_class_count(recognizer->dictionary);
    recognizer->info.max_text_capacity =
        (uint64_t)recognizer->info.time_steps * max_label_bytes + 1u;
    recognizer->info.workspace_size = session_info.workspace_size;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    recognizer_try_backends(recognizer);
#if !defined(LW_EXPERIMENTAL_CTC_TILED)
    if (!recognizer_has_full_compiled_coverage(recognizer) &&
        recognizer->model->info.content_checksum == LW_MEDIUM_REC_LWM_CHECKSUM) {
        lw_cpu_capabilities cpu = lw_get_cpu_capabilities();
        if (lw_simd_level_is_avx2(cpu.simd) && cpu.has_avx2_fma) {
            lw_medium_fast_shared* shared = (lw_medium_fast_shared*)calloc(1u, sizeof(*shared));
            lw_error fast_error;
            if (shared != NULL) {
                lw_error_init(&fast_error);
                if (lw_x64_rec_fast_plan_create(recognizer->session,
                                                &shared->packed_template,
                                                &fast_error) == LW_STATUS_OK) {
                    /* The template only owns immutable packed coefficients and
                     * folded bias; actual executions use private session plans. */
                    free(shared->packed_template->nhwc_workspace);
                    shared->packed_template->nhwc_workspace = NULL;
                    shared->packed_template->nhwc_workspace_bytes = 0u;
                    free(shared->packed_template->scratch);
                    shared->packed_template->scratch = NULL;
                    shared->packed_template->scratch_bytes = 0u;
                    shared->reference_count = 1u;
                    recognizer->fast_shared = shared;
                } else {
                    free(shared);
                }
            }
        }
    }
#endif
    recognizer_report_compiled_memory(recognizer);
#endif
    *out_recognizer = recognizer;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_recognizer_free(recognizer);
    return status;
}

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#if defined(_MSC_VER)
static void* rec_ws_aligned_alloc(size_t size) { return _aligned_malloc(size, 64u); }
static void rec_ws_aligned_free(void* p) { _aligned_free(p); }
#else
static void* rec_ws_aligned_alloc(size_t size) {
    void* p = NULL;
    return posix_memalign(&p, 64u, size) == 0 ? p : NULL;
}
static void rec_ws_aligned_free(void* p) { free(p); }
#endif

static void recognizer_release_shared_workspace(lw_recognizer* recognizer) {
    rec_ws_aligned_free(recognizer->x64_shared_ctc_scores);
    rec_ws_aligned_free(recognizer->x64_shared_best_indices);
    rec_ws_aligned_free(recognizer->x64_shared_best_probabilities);
    rec_ws_aligned_free(recognizer->x64_shared_arena);
    rec_ws_aligned_free(recognizer->x64_shared_scratch);
    recognizer->x64_shared_ctc_scores = NULL;
    recognizer->x64_shared_best_indices = NULL;
    recognizer->x64_shared_best_probabilities = NULL;
    recognizer->x64_shared_arena = NULL;
    recognizer->x64_shared_scratch = NULL;
}

/* Create one instance per compiled slot, borrowing the shared workspace when
 * it allocated cleanly; otherwise fall back to per-slot owned workspaces. */
static void recognizer_create_slot_instances(lw_recognizer* recognizer) {
    uint64_t arena_bytes = 0u;
    uint64_t scratch_bytes = 0u;
    uint64_t ctc_rows = 0u;
    uint32_t index;
    int ctc_needed = 0;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        const lw_x64_rec_program* program = recognizer->x64_slots[index].program;
        if (program == NULL) continue;
        if (program->arena_bytes > arena_bytes) arena_bytes = program->arena_bytes;
        if (program->scratch_bytes > scratch_bytes) scratch_bytes = program->scratch_bytes;
        if (program->ctc.enabled != 0u) {
            ctc_needed = 1;
            if (program->ctc.rows > ctc_rows) ctc_rows = program->ctc.rows;
        }
    }
    if (arena_bytes == 0u || arena_bytes > SIZE_MAX || scratch_bytes > SIZE_MAX ||
        ctc_rows > SIZE_MAX / sizeof(float)) {
        return;
    }
    recognizer->x64_shared_arena = (uint8_t*)rec_ws_aligned_alloc((size_t)arena_bytes);
    if (scratch_bytes > 0u) {
        recognizer->x64_shared_scratch = (uint8_t*)rec_ws_aligned_alloc((size_t)scratch_bytes);
    }
    if (ctc_needed != 0) {
        recognizer->x64_shared_ctc_scores = (float*)rec_ws_aligned_alloc((size_t)(ctc_rows * sizeof(float)));
        recognizer->x64_shared_best_indices = (uint32_t*)rec_ws_aligned_alloc((size_t)(ctc_rows * sizeof(uint32_t)));
        recognizer->x64_shared_best_probabilities = (float*)rec_ws_aligned_alloc((size_t)(ctc_rows * sizeof(float)));
    }
    if (recognizer->x64_shared_arena == NULL ||
        (scratch_bytes > 0u && recognizer->x64_shared_scratch == NULL) ||
        (ctc_needed != 0 && (recognizer->x64_shared_ctc_scores == NULL ||
                             recognizer->x64_shared_best_indices == NULL ||
                             recognizer->x64_shared_best_probabilities == NULL))) {
        /* Shared allocation failed: per-slot owned workspaces below. */
        recognizer_release_shared_workspace(recognizer);
    }
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        lw_error backend_error;
        lw_status status;
        if (slot->program == NULL) continue;
        lw_error_init(&backend_error);
        if (recognizer->x64_shared_arena != NULL) {
            status = lw_x64_rec_instance_create_borrowed(slot->program,
                recognizer->x64_shared_arena, arena_bytes,
                recognizer->x64_shared_scratch, scratch_bytes,
                recognizer->x64_shared_ctc_scores, recognizer->x64_shared_best_indices,
                recognizer->x64_shared_best_probabilities, ctc_rows,
                &slot->instance, &backend_error);
        } else {
            status = lw_x64_rec_instance_create(slot->program, &slot->instance,
                                                &backend_error);
        }
        if (status != LW_STATUS_OK) {
            lw_x64_rec_program_free(slot->program);
            slot->program = NULL;
            slot->instance = NULL;
        }
        if (slot->instance != NULL) {
            lw_x64_rec_instance_set_thread_pool(slot->instance, recognizer->intra_pool,
                                                recognizer->intra_workers);
        }
    }
}

static lw_x64_rec_backend_slot* recognizer_backend_slot(lw_recognizer* recognizer,
                                                        uint32_t width) {
    uint32_t index;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        if (slot->target_width == width && slot->instance != NULL) return slot;
    }
    return NULL;
}

/* The local rotated A/B benchmark selects NHWC for all five adaptive widths.
 * A failed compile or instance allocation leaves the canonical path available. */
static void recognizer_try_backends(lw_recognizer* recognizer) {
#if !defined(__EMSCRIPTEN__)
    lw_cpu_capabilities cpu = lw_get_cpu_capabilities();
#endif
    int32_t root_index;
    lw_x64_rec_program* root_program = NULL;
    uint32_t index;
#if !defined(__EMSCRIPTEN__)
    if (!lw_simd_level_is_avx2(cpu.simd) || !cpu.has_avx2_fma) return;
#endif
    root_index = (int32_t)LW_REC_RESIDENT_WIDTH_COUNT - 1;
    while (root_index >= 0 &&
           lw_rec_adaptive_widths[root_index] > recognizer->info.target_width)
        --root_index;
    if (root_index < 0) return;
    {
        lw_x64_rec_backend_slot* root = &recognizer->x64_slots[root_index];
        lw_error backend_error;
        root->target_width = lw_rec_adaptive_widths[root_index];
        lw_error_init(&backend_error);
        if (lw_x64_rec_backend_compile(recognizer->model, root->target_width,
                                       &root->program, &backend_error) == LW_X64_REC_COMPILE_OK &&
            root->program != NULL && root->program->ctc.enabled != 0u &&
            root->program->time_steps != 0u &&
            root->program->class_count == recognizer->info.class_count) {
            root_program = root->program;
        } else {
            lw_x64_rec_program_free(root->program);
            root->program = NULL;
        }
    }
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        lw_error backend_error;
        uint32_t width = lw_rec_adaptive_widths[index];
        if (width > recognizer->info.target_width || (int32_t)index == root_index) continue;
        slot->target_width = width;
        lw_error_init(&backend_error);
        if (root_program == NULL ||
            lw_x64_rec_backend_compile_shared(recognizer->model, width, root_program,
                                              &slot->program, &backend_error) != LW_X64_REC_COMPILE_OK) {
            lw_x64_rec_program_free(slot->program);
            slot->program = NULL;
            lw_error_init(&backend_error);
            (void)lw_x64_rec_backend_compile(recognizer->model, width,
                                             &slot->program, &backend_error);
        }
        if (slot->program == NULL || slot->program->ctc.enabled == 0u ||
            slot->program->time_steps == 0u ||
            slot->program->class_count != recognizer->info.class_count) {
            lw_x64_rec_program_free(slot->program);
            slot->program = NULL;
            slot->instance = NULL;
        }
    }
    recognizer_create_slot_instances(recognizer);
#if defined(__EMSCRIPTEN__)
    {
        uint32_t compiled = 0u;
        for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
            if (recognizer->x64_slots[index].instance != NULL) ++compiled;
        }
        fprintf(stderr, "LW_WASM_COMPILED_REC widths=%u/%u\n",
                compiled, LW_REC_RESIDENT_WIDTH_COUNT);
    }
#endif
}

static void recognizer_release_backends(lw_recognizer* recognizer) {
    uint32_t index;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_x64_rec_backend_slot* slot = &recognizer->x64_slots[index];
        lw_x64_rec_instance_free(slot->instance);
        lw_x64_rec_program_free(slot->program);
        memset(slot, 0, sizeof(*slot));
    }
    recognizer_release_shared_workspace(recognizer);
}

/* The hybrid plan supports this exact Medium LWM graph across the five
 * adaptive widths. Small has known argmax mismatches and must stay on the
 * canonical executor until its numerical contract is fixed. */
static int recognizer_try_medium_fast_plan(lw_recognizer* recognizer) {
#if defined(LW_EXPERIMENTAL_CTC_TILED)
    (void)recognizer;
    return 0;
#else
    lw_cpu_capabilities cpu;
    lw_error backend_error;
    uint32_t width_index;
    int owns_probabilities = 0;
    if (recognizer->model->info.content_checksum != LW_MEDIUM_REC_LWM_CHECKSUM ||
        recognizer->fast_shared == NULL ||
        recognizer->fast_backend_disabled != 0u ||
        recognizer->fast_plan_disabled != 0u) {
        return 0;
    }
    cpu = lw_get_cpu_capabilities();
    if (!lw_simd_level_is_avx2(cpu.simd) || !cpu.has_avx2_fma) return 0;
    for (width_index = 0u; width_index < LW_REC_RESIDENT_WIDTH_COUNT; ++width_index) {
        if (recognizer->current_target_width == lw_rec_adaptive_widths[width_index]) break;
    }
    if (width_index == LW_REC_RESIDENT_WIDTH_COUNT) return 0;
    if (recognizer->probabilities == NULL) {
        if (!allocation_fits(recognizer->probability_element_count, sizeof(float))) {
            recognizer->fast_plan_disabled = 1u;
            return 0;
        }
        recognizer->probabilities = (float*)malloc(
            (size_t)recognizer->probability_element_count * sizeof(float));
        if (recognizer->probabilities == NULL) {
            recognizer->fast_plan_disabled = 1u;
            return 0;
        }
        owns_probabilities = 1;
    }
    if (recognizer->fast_plan == NULL) {
        lw_error_init(&backend_error);
        if (lw_x64_rec_fast_plan_create_shared(recognizer->session,
                                               recognizer->fast_shared->packed_template,
                                               &recognizer->fast_plan, &backend_error) != LW_STATUS_OK) {
            if (owns_probabilities != 0) {
                free(recognizer->probabilities);
                recognizer->probabilities = NULL;
            }
            recognizer->fast_plan_disabled = 1u;
            return 0;
        }
    }
    return 1;
#endif
}
#endif

lw_status lw_recognizer_clone(const lw_recognizer* source, lw_recognizer** out_recognizer,
                              lw_error* error) {
    lw_recognizer* clone;
    lw_session_info session_info;
    lw_status status;
    if (out_recognizer != NULL) {
        *out_recognizer = NULL;
    }
    if (source == NULL || out_recognizer == NULL || source->model == NULL ||
        source->dictionary == NULL || source->current_target_width == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "source recognizer and output handle are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    clone = (lw_recognizer*)calloc(1u, sizeof(*clone));
    if (clone == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate recognizer clone");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    clone->model = source->model;
    clone->dictionary = source->dictionary;
    clone->max_image_pixels = source->max_image_pixels;
    clone->session_options = source->session_options;
    clone->adaptive_width_enabled = source->adaptive_width_enabled;
    clone->info = source->info;
    lw_model_retain(clone->model);
    lw_rec_dictionary_retain(clone->dictionary);
    lw_session_info_init(&session_info);
    status = configure_session(clone, source->current_target_width, &session_info, error);
    if (status != LW_STATUS_OK) {
        lw_recognizer_free(clone);
        return status;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    clone->fast_backend_disabled = source->fast_backend_disabled;
    clone->fast_plan_disabled = source->fast_plan_disabled;
    clone->fast_shared = source->fast_shared;
    if (clone->fast_shared != NULL) ++clone->fast_shared->reference_count;
#endif
    status = lw_session_share_prepared_constants(clone->session, source->session, error);
    if (status != LW_STATUS_OK) {
        lw_recognizer_free(clone);
        return status;
    }
    clone->info.time_steps = clone->current_time_steps;
    clone->info.workspace_size = session_info.workspace_size;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    {
        uint32_t index;
        for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
            const lw_x64_rec_backend_slot* source_slot = &source->x64_slots[index];
            lw_x64_rec_backend_slot* clone_slot = &clone->x64_slots[index];
            if (source_slot->instance == NULL) continue;
            clone_slot->target_width = source_slot->target_width;
            clone_slot->program = source_slot->program;
            lw_x64_rec_program_retain(clone_slot->program);
        }
        recognizer_create_slot_instances(clone);
    }
#endif
    lw_recognizer_set_intra_op_thread_count(clone, source->intra_workers);
    *out_recognizer = clone;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
/* Test-only internal hook: disable the compiled x64 REC backend so the same
 * binary can measure canonical recognizer timing. */
void lw_recognizer_test_disable_x64_backend(lw_recognizer* recognizer) {
    uint32_t index;
    if (recognizer == NULL) return;
    recognizer_release_backends(recognizer);
    recognizer->fast_backend_disabled = 1u;
    lw_x64_rec_fast_plan_free(recognizer->fast_plan);
    recognizer->fast_plan = NULL;
    recognizer->fast_plan_disabled = 1u;
    lw_x64_rec_fast_plan_free(recognizer->cached_fast_plan);
    recognizer->cached_fast_plan = NULL;
    recognizer->cached_fast_plan_disabled = 1u;
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        lw_rec_resident_slot* slot = &recognizer->resident_slots[index];
        lw_x64_rec_fast_plan_free(slot->fast_plan);
        slot->fast_plan = NULL;
        slot->fast_plan_disabled = 1u;
    }
    medium_fast_shared_release(recognizer->fast_shared);
    recognizer->fast_shared = NULL;
}
#endif

void lw_recognizer_free(lw_recognizer* recognizer) {
    if (recognizer == NULL) {
        return;
    }
    free(recognizer->best_probabilities);
    free(recognizer->best_indices);
    free(recognizer->probabilities);
    free(recognizer->input);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_fast_plan_free(recognizer->fast_plan);
    recognizer_release_backends(recognizer);
#endif
    lw_session_free(recognizer->session);
    {
        uint32_t slot_index;
        for (slot_index = 0u; slot_index < LW_REC_RESIDENT_WIDTH_COUNT; ++slot_index) {
            release_resident_slot(&recognizer->resident_slots[slot_index]);
        }
    }
    release_cached_session(recognizer);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    medium_fast_shared_release(recognizer->fast_shared);
#endif
    lw_thread_pool_free(recognizer->intra_pool);
    recognizer->intra_pool = NULL;
    lw_rec_dictionary_free(recognizer->dictionary);
    lw_model_free(recognizer->model);
    free(recognizer);
}

static void recognizer_attach_intra_pool(lw_recognizer* recognizer) {
    uint32_t index;
    if (recognizer->session != NULL) {
        lw_session_set_external_thread_pool(recognizer->session, recognizer->intra_pool,
                                            recognizer->intra_workers);
    }
    if (recognizer->cached_session != NULL) {
        lw_session_set_external_thread_pool(recognizer->cached_session, recognizer->intra_pool,
                                            recognizer->intra_workers);
    }
    for (index = 0u; index < LW_REC_RESIDENT_WIDTH_COUNT; ++index) {
        if (recognizer->resident_slots[index].session != NULL) {
            lw_session_set_external_thread_pool(recognizer->resident_slots[index].session,
                                                recognizer->intra_pool, recognizer->intra_workers);
        }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        if (recognizer->x64_slots[index].instance != NULL) {
            lw_x64_rec_instance_set_thread_pool(recognizer->x64_slots[index].instance,
                                                recognizer->intra_pool, recognizer->intra_workers);
        }
#endif
    }
}

void lw_recognizer_set_intra_op_thread_count(lw_recognizer* recognizer, uint32_t thread_count) {
    if (recognizer == NULL) return;
    if (thread_count == 0u) thread_count = 1u;
    if (thread_count > LW_PARALLEL_MAX_WORKERS) thread_count = LW_PARALLEL_MAX_WORKERS;
    lw_thread_pool_free(recognizer->intra_pool);
    recognizer->intra_pool =
        thread_count > 1u ? lw_thread_pool_create(thread_count) : NULL;
    recognizer->intra_workers =
        thread_count > 1u ? lw_thread_pool_worker_count(recognizer->intra_pool) : 1u;
    recognizer_attach_intra_pool(recognizer);
}

lw_status lw_recognizer_get_info(const lw_recognizer* recognizer, lw_recognizer_info* info) {
    if (recognizer == NULL || info == NULL ||
        !lw_abi_copy_output_prefix(info, info->struct_size, &recognizer->info,
                                   sizeof(recognizer->info))) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return LW_STATUS_OK;
}

static lw_status recognizer_recognize_bgr_u8_impl(lw_recognizer* recognizer, const uint8_t* source,
                                                  uint64_t source_byte_count, uint32_t source_width,
                                                  uint32_t source_height, uint32_t source_stride,
                                                  char* text_utf8, uint64_t text_capacity,
                                                  lw_recognition_result* result,
                                                  lw_pipeline_component_profile* profile,
                                                  lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width = 0u;
    uint64_t required_capacity = 0u;
    float score = 0.0f;
    uint32_t emitted_count = 0u;
    uint32_t target_width;
    uint32_t execution_time_steps;
    uint32_t width_bucket = 0u;
    uint64_t node_nanoseconds_before[LW_EXECUTION_PROFILE_NODE_CAPACITY];
    uint64_t node_invocations_before[LW_EXECUTION_PROFILE_NODE_CAPACITY];
    lw_recognition_result output;
    uint32_t result_size;
    uint64_t started;
    lw_status status;
    uint32_t* execution_best_indices = NULL;
    float* execution_best_probabilities = NULL;
    int backend_active = 0;
    int hybrid_active = 0;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_backend_slot* backend_slot = NULL;
    float* backend_input = NULL;
    uint64_t backend_input_count = 0u;
#endif
    if (recognizer == NULL || source == NULL || result == NULL ||
        result->struct_size < sizeof(uint32_t) || (text_utf8 == NULL && text_capacity != 0u)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "recognizer, BGR source, and initialized result are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    result_size = result->struct_size;
    clear_result(&output);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > recognizer->max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    target_width = lw_recognizer_target_width_for_image(recognizer, source_width, source_height);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    backend_slot = recognizer_backend_slot(recognizer, target_width);
    backend_active = backend_slot != NULL;
#endif
    if (profile != NULL) {
        if (backend_active || recognizer->resident_widths_enabled != 0u ||
            target_width == recognizer->current_target_width ||
            target_width == recognizer->cached_target_width) {
            if (profile->session_cache_hits != UINT64_MAX) {
                ++profile->session_cache_hits;
            }
        } else {
            if (profile->session_cache_misses != UINT64_MAX) {
                ++profile->session_cache_misses;
            }
            if (profile->session_reconfigurations != UINT64_MAX) {
                ++profile->session_reconfigurations;
            }
        }
    }
    if (!backend_active && target_width != recognizer->current_target_width) {
        status = configure_session(recognizer, target_width, NULL, error);
        if (status != LW_STATUS_OK) {
            return status;
        }
    }
    execution_time_steps = recognizer->current_time_steps;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (backend_active) execution_time_steps = backend_slot->program->time_steps;
#endif
    started = lw_pipeline_profile_now(profile);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (backend_active) {
        backend_input = lw_x64_rec_instance_input(backend_slot->instance, &backend_input_count);
        if (backend_slot->program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
            status = lw_rec_preprocess_bgr_u8(source, source_byte_count, source_width,
                                              source_height, source_stride, target_width, backend_input,
                                              backend_input_count, &resized_width);
        } else {
            status = lw_rec_preprocess_bgr_u8_nhwc(source, source_byte_count, source_width,
                                                   source_height, source_stride, target_width,
                                                   backend_input, backend_input_count,
                                                   &resized_width);
        }
    } else
#endif
    {
        status = lw_rec_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                          source_stride, recognizer->current_target_width, recognizer->input,
                                          recognizer->input_element_count, &resized_width);
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (status != LW_STATUS_OK && backend_active) {
        backend_active = 0;
        if (target_width != recognizer->current_target_width) {
            status = configure_session(recognizer, target_width, NULL, error);
            if (status != LW_STATUS_OK) return status;
        }
        execution_time_steps = recognizer->current_time_steps;
        status = lw_rec_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                          source_stride, recognizer->current_target_width, recognizer->input,
                                          recognizer->input_element_count, &resized_width);
    }
#endif
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (!backend_active) {
        hybrid_active = recognizer_try_medium_fast_plan(recognizer);
    }
#endif
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->preprocess_nanoseconds,
                                    started, profile);
    if (profile != NULL) {
        width_bucket = lw_rec_width_histogram_bucket(resized_width);
        memcpy(node_nanoseconds_before, profile->execution.node_nanoseconds,
               sizeof(node_nanoseconds_before));
        memcpy(node_invocations_before, profile->execution.node_invocations,
               sizeof(node_invocations_before));
    }    started = lw_pipeline_profile_now(profile);
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (backend_active && status == LW_STATUS_OK) {
        status = lw_x64_rec_instance_run(backend_slot->instance, error);
        if (status == LW_STATUS_OK) {
            execution_best_indices = backend_slot->instance->best_indices;
            execution_best_probabilities = backend_slot->instance->best_probabilities;
        } else {
            /* The recognizer input contains the backend layout after a successful
             * backend preprocess. Do not silently feed that NHWC buffer to the
             * canonical NCHW executor after a runtime backend failure. */
            return status;
        }
    }
#endif
    if (!backend_active) {
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
        if (hybrid_active) {
            status = lw_x64_rec_fast_run(
                recognizer->fast_plan, recognizer->input, recognizer->input_element_count,
                recognizer->probabilities, recognizer->probability_element_count, error);
            if (status != LW_STATUS_OK) {
                lw_x64_rec_fast_plan_free(recognizer->fast_plan);
                recognizer->fast_plan = NULL;
                recognizer->fast_plan_disabled = 1u;
                hybrid_active = 0;
                lw_error_init(error);
            }
        }
#endif
        if (!hybrid_active && recognizer->best_indices != NULL) {
            status = lw_execute_session_f32_ctc_greedy(
                recognizer->session, recognizer->input, recognizer->input_element_count,
                recognizer->best_indices, recognizer->best_probabilities,
                execution_time_steps, recognizer->info.class_count,
                profile == NULL ? NULL : &profile->execution, error);
            if (status == LW_STATUS_OK) {
                execution_best_indices = recognizer->best_indices;
                execution_best_probabilities = recognizer->best_probabilities;
            }
        } else if (!hybrid_active) {
            status = profile == NULL
                         ? lw_execute_session_f32(
                               recognizer->session, recognizer->input,
                               recognizer->input_element_count, recognizer->probabilities,
                               recognizer->probability_element_count, error)
                         : lw_execute_session_f32_profiled(
                               recognizer->session, recognizer->input,
                               recognizer->input_element_count, recognizer->probabilities,
                               recognizer->probability_element_count, &profile->execution, error);
        }
    }
    if (status == LW_STATUS_OK && profile != NULL) {
        lw_pipeline_profile_capture_node_width_delta(
            profile, width_bucket, node_nanoseconds_before, node_invocations_before);
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->graph_nanoseconds, started,
                                    profile);
    if (status != LW_STATUS_OK) {
        return status;
    }
    if (profile != NULL) {
        uint64_t* count = (backend_active || hybrid_active)
                              ? &profile->compiled_backend_lines
                              : &profile->canonical_fallback_lines;
        if (*count != UINT64_MAX) ++*count;
    }
    started = lw_pipeline_profile_now(profile);
    if (execution_best_indices != NULL) {
        if (text_utf8 != NULL && text_capacity >= recognizer->info.max_text_capacity) {
            status = lw_rec_ctc_decode_greedy_known_capacity_f32(
                recognizer->dictionary, execution_best_indices,
                execution_best_probabilities, execution_time_steps,
                recognizer->info.class_count, text_utf8, text_capacity, &required_capacity,
                &score, &emitted_count, error);
        } else {
            status = lw_rec_ctc_decode_greedy_f32(
                recognizer->dictionary, execution_best_indices,
                execution_best_probabilities, execution_time_steps,
                recognizer->info.class_count, text_utf8, text_capacity, &required_capacity,
                &score, &emitted_count, error);
        }
    } else if (text_utf8 != NULL && text_capacity >= recognizer->info.max_text_capacity) {
        status = lw_rec_ctc_decode_known_capacity_f32(
            recognizer->dictionary, recognizer->probabilities,
            recognizer->probability_element_count, execution_time_steps,
            recognizer->info.class_count, text_utf8, text_capacity, &required_capacity, &score,
            &emitted_count, error);
    } else {
        status = lw_rec_ctc_decode_f32(recognizer->dictionary, recognizer->probabilities,
                                       recognizer->probability_element_count,
                                       execution_time_steps, recognizer->info.class_count,
                                       text_utf8, text_capacity, &required_capacity, &score,
                                       &emitted_count, error);
    }
    output.emitted_count = emitted_count;
    output.score = score;
    output.resized_width = resized_width;
    output.time_steps = execution_time_steps;
    output.required_text_capacity = required_capacity;
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->postprocess_nanoseconds,
                                    started, profile);
    (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
    return status;
}

lw_status lw_recognizer_recognize_bgr_u8(lw_recognizer* recognizer, const uint8_t* source,
                                         uint64_t source_byte_count, uint32_t source_width,
                                         uint32_t source_height, uint32_t source_stride,
                                         char* text_utf8, uint64_t text_capacity,
                                         lw_recognition_result* result, lw_error* error) {
    return recognizer_recognize_bgr_u8_impl(recognizer, source, source_byte_count, source_width,
                                            source_height, source_stride, text_utf8, text_capacity,
                                            result, NULL, error);
}

lw_status lw_recognizer_recognize_bgr_u8_profiled(lw_recognizer* recognizer, const uint8_t* source,
                                                  uint64_t source_byte_count, uint32_t source_width,
                                                  uint32_t source_height, uint32_t source_stride,
                                                  char* text_utf8, uint64_t text_capacity,
                                                  lw_recognition_result* result,
                                                  lw_pipeline_component_profile* profile,
                                                  lw_error* error) {
    if (profile == NULL || profile->execution.struct_size != sizeof(profile->execution) ||
        profile->execution.clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized recognizer profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return recognizer_recognize_bgr_u8_impl(recognizer, source, source_byte_count, source_width,
                                            source_height, source_stride, text_utf8, text_capacity,
                                            result, profile, error);
}
