#include "lw_infer.h"
#include "abi_compat_internal.h"

/* Public CLS handle for detecting whether a text crop needs 180-degree rotation. */

#include "cls_internal.h"
#include "error_internal.h"
#include "executor_internal.h"
#include "profile_internal.h"

#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
#include "x64_rec_backend_internal.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_CLS_DEFAULT_MAX_IMAGE_PIXELS UINT64_C(40000000)

struct lw_classifier {
    lw_model* model;
    lw_session* session;
    float* input;
    float probabilities[LW_CLS_CLASS_COUNT];
    uint64_t input_element_count;
    uint64_t max_image_pixels;
    lw_classifier_info info;
    lw_cls_preprocess_workspace preprocess_workspace;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    /* Compiled x64 backbone (reuses the REC lowering with the CLS input
     * shape and no CTC tail). NULL when unsupported; the canonical session
     * above remains the fallback. */
    lw_x64_rec_program* x64_program;
    lw_x64_rec_instance* x64_instance;
#endif
};

static void clear_result(lw_classification_result* result) {
    memset(result, 0, sizeof(*result));
    result->struct_size = (uint32_t)sizeof(*result);
}

void lw_classifier_options_init(lw_classifier_options* options) {
    lw_model_options model_options;
    lw_session_options session_options;
    if (options == NULL) {
        return;
    }
    lw_model_options_init(&model_options);
    lw_session_options_init(&session_options);
    memset(options, 0, sizeof(*options));
    options->struct_size = (uint32_t)sizeof(*options);
    options->max_model_file_size = model_options.max_file_size;
    options->max_workspace_size = session_options.max_workspace_size;
    options->max_tensor_size = session_options.max_tensor_size;
    options->max_image_pixels = LW_CLS_DEFAULT_MAX_IMAGE_PIXELS;
}

void lw_classifier_info_init(lw_classifier_info* info) {
    if (info == NULL) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
}

void lw_classification_result_init(lw_classification_result* result) {
    if (result == NULL) {
        return;
    }
    clear_result(result);
}

static lw_status validate_options(const lw_classifier_options* options, uint64_t* max_image_pixels,
                                  lw_model_options* model_options,
                                  lw_session_options* session_options, lw_error* error) {
    lw_classifier_options defaults;
    lw_classifier_options values;
    lw_classifier_options_init(&defaults);
    values = defaults;
    *max_image_pixels = defaults.max_image_pixels;
    lw_model_options_init(model_options);
    lw_session_options_init(session_options);
    if (options == NULL) {
        return LW_STATUS_OK;
    }
    if (!lw_abi_copy_input_prefix(&values, sizeof(values), options)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid classifier options structure");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    values.struct_size = (uint32_t)sizeof(values);
    if (values.reserved != 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid classifier options structure");
        return LW_STATUS_INVALID_ARGUMENT;
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
    return LW_STATUS_OK;
}

lw_status lw_classifier_create(const char* model_path_utf8, const lw_classifier_options* options,
                               lw_classifier** out_classifier, lw_error* error) {
    lw_model_options model_options;
    lw_session_options session_options;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_session_info session_info;
    lw_classifier* classifier = NULL;
    uint64_t max_image_pixels;
    lw_status status;
    if (out_classifier != NULL) {
        *out_classifier = NULL;
    }
    if (model_path_utf8 == NULL || model_path_utf8[0] == '\0' || out_classifier == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "model path and output classifier are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    status = validate_options(options, &max_image_pixels, &model_options, &session_options, error);
    if (status != LW_STATUS_OK) {
        return status;
    }
    classifier = (lw_classifier*)calloc(1u, sizeof(*classifier));
    if (classifier == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate classifier handle");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    classifier->max_image_pixels = max_image_pixels;
    status = lw_model_load(model_path_utf8, &model_options, &classifier->model, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)LW_CLS_INPUT_HEIGHT;
    input_desc.dimensions[3] = (int32_t)LW_CLS_INPUT_WIDTH;
    status = lw_session_create(classifier->model, &input_desc, 1u, &session_options,
                               &classifier->session, error);
    if (status != LW_STATUS_OK) {
        goto fail;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(classifier->session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 2u ||
        output_desc.dimensions[0] != 1 ||
        output_desc.dimensions[1] != (int32_t)LW_CLS_CLASS_COUNT) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE,
                     "classifier model output must have shape [1,2]");
        status = LW_STATUS_INVALID_SHAPE;
        goto fail;
    }
    classifier->input_element_count = (uint64_t)3u * LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    if (classifier->input_element_count > SIZE_MAX / sizeof(*classifier->input)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "classifier input buffer size overflows");
        status = LW_STATUS_OUT_OF_BOUNDS;
        goto fail;
    }
    classifier->input =
        (float*)malloc((size_t)classifier->input_element_count * sizeof(*classifier->input));
    if (classifier->input == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate classifier input buffer");
        status = LW_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    lw_session_info_init(&session_info);
    status = lw_session_get_info(classifier->session, &session_info);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "unable to read classifier session information");
        goto fail;
    }
    lw_classifier_info_init(&classifier->info);
    classifier->info.input_width = LW_CLS_INPUT_WIDTH;
    classifier->info.input_height = LW_CLS_INPUT_HEIGHT;
    classifier->info.class_count = LW_CLS_CLASS_COUNT;
    classifier->info.workspace_size = session_info.workspace_size;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    {
        lw_error backend_error;
        lw_error_init(&backend_error);
        if (lw_x64_rec_backend_compile_input(classifier->model, LW_CLS_INPUT_HEIGHT,
                                             LW_CLS_INPUT_WIDTH, LW_X64_REC_COMPILE_NHWC, 0u,
                                             &classifier->x64_program,
                                             &backend_error) != LW_X64_REC_COMPILE_OK ||
            classifier->x64_program == NULL ||
            classifier->x64_program->unsupported_nodes != 0u ||
            lw_x64_rec_instance_create(classifier->x64_program, &classifier->x64_instance,
                                       &backend_error) != LW_STATUS_OK) {
            lw_x64_rec_instance_free(classifier->x64_instance);
            lw_x64_rec_program_free(classifier->x64_program);
            classifier->x64_instance = NULL;
            classifier->x64_program = NULL;
        }
#if defined(__EMSCRIPTEN__)
        if (classifier->x64_program != NULL && classifier->x64_instance != NULL) {
            uint64_t packed_bytes = 0u;
            uint32_t constant_index;
            for (constant_index = 0u;
                 constant_index < classifier->x64_program->packed_constant_count;
                 ++constant_index) {
                packed_bytes += classifier->x64_program->constants[constant_index].bytes;
            }
            (void)fprintf(stderr,
                "LW_WASM_COMPILED_CLS layout=nhwc ops=%u unsupported=%u "
                "arena_bytes=%llu packed_bytes=%llu\n",
                classifier->x64_program->op_count,
                classifier->x64_program->unsupported_nodes,
                (unsigned long long)classifier->x64_program->arena_bytes,
                (unsigned long long)packed_bytes);
        }
#endif
    }
#endif
    *out_classifier = classifier;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;

fail:
    lw_classifier_free(classifier);
    return status;
}

void lw_classifier_free(lw_classifier* classifier) {
    if (classifier == NULL) {
        return;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_x64_rec_instance_free(classifier->x64_instance);
    lw_x64_rec_program_free(classifier->x64_program);
#endif
    free(classifier->input);
    lw_session_free(classifier->session);
    lw_model_free(classifier->model);
    lw_cls_preprocess_workspace_free(&classifier->preprocess_workspace);
    free(classifier);
}

lw_status lw_classifier_get_info(const lw_classifier* classifier, lw_classifier_info* info) {
    if (classifier == NULL || info == NULL ||
        !lw_abi_copy_output_prefix(info, info->struct_size, &classifier->info,
                                   sizeof(classifier->info))) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return LW_STATUS_OK;
}

static lw_status classifier_classify_bgr_u8_impl(lw_classifier* classifier, const uint8_t* source,
                                                 uint64_t source_byte_count, uint32_t source_width,
                                                 uint32_t source_height, uint32_t source_stride,
                                                 lw_classification_result* result,
                                                 lw_pipeline_component_profile* profile,
                                                 lw_error* error) {
    uint64_t source_pixels;
    uint32_t resized_width = 0u;
    uint32_t label;
    lw_classification_result output;
    uint32_t result_size;
    uint64_t started;
    lw_status status;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    uint64_t backend_input_count = 0u;
    float* backend_input = classifier == NULL || classifier->x64_instance == NULL
        ? NULL : lw_x64_rec_instance_input(classifier->x64_instance, &backend_input_count);
#if defined(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
    int direct_nhwc_input = 0;
#endif
#endif
    if (classifier == NULL || source == NULL || result == NULL ||
        result->struct_size < sizeof(uint32_t)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "classifier, BGR source, and initialized result are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    result_size = result->struct_size;
    clear_result(&output);
    if (source_width == 0u || source_height == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "source image dimensions must be positive");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    source_pixels = (uint64_t)source_width * source_height;
    if (source_pixels > classifier->max_image_pixels) {
        lw_set_error(error, LW_STATUS_MEMORY_LIMIT, "source image exceeds max_image_pixels");
        return LW_STATUS_MEMORY_LIMIT;
    }
    started = lw_pipeline_profile_now(profile);
#if defined(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (backend_input != NULL && backend_input_count == classifier->input_element_count) {
        status = lw_cls_preprocess_bgr_u8_fixed_nhwc(
            source, source_byte_count, source_width, source_height, source_stride,
            backend_input, backend_input_count, &resized_width,
            &classifier->preprocess_workspace);
        direct_nhwc_input = 1;
    } else
#endif
    status = lw_cls_preprocess_bgr_u8_fixed(source, source_byte_count, source_width,
                                            source_height, source_stride, classifier->input,
                                            classifier->input_element_count, &resized_width,
                                            &classifier->preprocess_workspace);
#else
    status = lw_cls_preprocess_bgr_u8(source, source_byte_count, source_width, source_height,
                                      source_stride, classifier->input,
                                      classifier->input_element_count, &resized_width);
#endif
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "BGR source layout is invalid");
        return status;
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->preprocess_nanoseconds,
                                    started, profile);
    started = lw_pipeline_profile_now(profile);
    status = LW_STATUS_UNSUPPORTED;
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    if (classifier->x64_instance != NULL && backend_input != NULL &&
        backend_input_count == classifier->input_element_count) {
#if defined(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
        if (!direct_nhwc_input) {
#endif
            /* NCHW planes -> interleaved NHWC; pure data movement, bit-exact. */
            const uint32_t height = LW_CLS_INPUT_HEIGHT;
            const uint32_t width = LW_CLS_INPUT_WIDTH;
            uint32_t y;
            uint32_t x;
            for (y = 0u; y < height; ++y) {
                for (x = 0u; x < width; ++x) {
                    uint32_t channel;
                    for (channel = 0u; channel < 3u; ++channel) {
                        backend_input[((size_t)y * width + x) * 3u + channel] =
                            classifier->input[((size_t)channel * height + y) * width + x];
                    }
                }
            }
#if defined(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
        }
#endif
        status = lw_x64_rec_instance_run(classifier->x64_instance, error);
        if (status == LW_STATUS_OK) {
            if (profile != NULL)
                lw_profile_add_value(&profile->compiled_backend_runs, 1u);
            const lw_x64_rec_program* program = classifier->x64_program;
            const float* probabilities =
                (const float*)(const void*)(classifier->x64_instance->arena +
                                            (size_t)program->values[program->output_value]
                                                .offset);
            classifier->probabilities[0] = probabilities[0];
            classifier->probabilities[1] = probabilities[1];
        }
    }
#endif
    if (status == LW_STATUS_UNSUPPORTED) {
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH) && defined(LW_EXPERIMENTAL_CLS_FIXED_POINT_RESIZE)
        if (direct_nhwc_input) {
            /* Recreate canonical NCHW input before the session fallback. */
            status = lw_cls_preprocess_bgr_u8_fixed(
                source, source_byte_count, source_width, source_height, source_stride,
                classifier->input, classifier->input_element_count, &resized_width,
                &classifier->preprocess_workspace);
            if (status != LW_STATUS_OK) {
                lw_set_error(error, status, "CLS fallback preprocessing failed");
                return status;
            }
        }
#endif
        status = profile == NULL
                     ? lw_execute_session_f32(classifier->session, classifier->input,
                                              classifier->input_element_count,
                                              classifier->probabilities, LW_CLS_CLASS_COUNT, error)
                     : lw_execute_session_f32_profiled(
                           classifier->session, classifier->input, classifier->input_element_count,
                           classifier->probabilities, LW_CLS_CLASS_COUNT, &profile->execution,
                           error);
        if (status == LW_STATUS_OK && profile != NULL)
            lw_profile_add_value(&profile->canonical_fallback_runs, 1u);
    }
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->graph_nanoseconds, started,
                                    profile);
    if (status != LW_STATUS_OK) {
        return status;
    }
    started = lw_pipeline_profile_now(profile);
    if (!isfinite(classifier->probabilities[0]) || !isfinite(classifier->probabilities[1])) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "classifier output contains a non-finite value");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    label = classifier->probabilities[1] > classifier->probabilities[0] ? 1u : 0u;
    output.label = label;
    output.score = classifier->probabilities[label];
    output.resized_width = resized_width;
    output.orientation_degrees = label == 0u ? 0u : 180u;
    lw_pipeline_profile_add_elapsed(profile == NULL ? NULL : &profile->postprocess_nanoseconds,
                                    started, profile);
    (void)lw_abi_copy_output_prefix(result, result_size, &output, sizeof(output));
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_classifier_classify_bgr_u8(lw_classifier* classifier, const uint8_t* source,
                                        uint64_t source_byte_count, uint32_t source_width,
                                        uint32_t source_height, uint32_t source_stride,
                                        lw_classification_result* result, lw_error* error) {
    return classifier_classify_bgr_u8_impl(classifier, source, source_byte_count, source_width,
                                           source_height, source_stride, result, NULL, error);
}

lw_status lw_classifier_classify_bgr_u8_profiled(lw_classifier* classifier, const uint8_t* source,
                                                 uint64_t source_byte_count, uint32_t source_width,
                                                 uint32_t source_height, uint32_t source_stride,
                                                 lw_classification_result* result,
                                                 lw_pipeline_component_profile* profile,
                                                 lw_error* error) {
    if (profile == NULL || profile->execution.struct_size != sizeof(profile->execution) ||
        profile->execution.clock == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT,
                     "an initialized classifier profile and clock are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return classifier_classify_bgr_u8_impl(classifier, source, source_byte_count, source_width,
                                           source_height, source_stride, result, profile, error);
}
