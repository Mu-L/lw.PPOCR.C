#include "lw_infer.h"
#include "lwm_read.h"
#include "session_internal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static void make_rec_input(lw_tensor_desc* input, int32_t batch, int32_t width) {
    lw_tensor_desc_init(input);
    input->dtype = LW_DTYPE_F32;
    input->rank = 4u;
    input->dimensions[0] = batch;
    input->dimensions[1] = 3;
    input->dimensions[2] = 48;
    input->dimensions[3] = width;
}

static uint64_t aligned_tensor_bytes(uint64_t bytes) {
    if (bytes > UINT64_MAX - (LW_WORKSPACE_ALIGNMENT - 1u)) {
        return UINT64_MAX;
    }
    return (bytes + (LW_WORKSPACE_ALIGNMENT - 1u)) &
           ~(uint64_t)(LW_WORKSPACE_ALIGNMENT - 1u);
}

static uint64_t semantic_workspace_lower_bound(const lw_model* model,
                                               const lw_session* session) {
    uint64_t maximum = 0u;
    uint32_t node;
    for (node = 0u; node <= model->info.node_count; ++node) {
        uint64_t live = 0u;
        uint32_t tensor_index;
        for (tensor_index = 0u; tensor_index < model->info.tensor_count; ++tensor_index) {
            const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
            uint64_t bytes;
            if (tensor->birth_node < 0 || tensor->birth_node > (int32_t)node ||
                tensor->last_use_node < (int32_t)node) {
                continue;
            }
            if (lw_execution_plan_is_skipped_tensor(session, tensor_index)) {
                continue;
            }
            bytes = aligned_tensor_bytes(tensor->byte_size);
            if (bytes == UINT64_MAX || live > UINT64_MAX - bytes) {
                return UINT64_MAX;
            }
            live += bytes;
        }
        if (live > maximum) {
            maximum = live;
        }
    }
    return maximum;
}

static int check_execution_table(const lw_model* model, const lw_session* session) {
    uint32_t node_index;
#if defined(LW_EXPERIMENTAL_PREPARED_EXECUTION)
    if (session->execution_nodes == NULL ||
        session->execution_node_count != model->info.node_count) {
        return 0;
    }
    for (node_index = 0u; node_index < session->execution_node_count; ++node_index) {
        const uint8_t* disk = model->bytes + (size_t)model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        const lw_bound_node* bound = &session->execution_nodes[node_index];
        if (bound->node_index != node_index ||
            bound->operator_type != lwm_read_u16(disk) ||
            bound->input_count != lwm_read_u16(disk + 2u) ||
            bound->output_index != lwm_read_u32(disk + 40u)) {
            return 0;
        }
        for (uint32_t input_index = 0u; input_index < bound->input_count; ++input_index) {
            if (bound->input_indices[input_index] !=
                lwm_read_u32(disk + 8u + (size_t)input_index * sizeof(uint32_t))) {
                return 0;
            }
        }
        if (bound->prepared_constant != NULL &&
            bound->prepared_constant != &session->prepared_constants[node_index]) {
            return 0;
        }
    }
#else
    (void)model;
    (void)node_index;
    if (session->execution_nodes != NULL || session->execution_node_count != 0u) {
        return 0;
    }
#endif
    return 1;
}
static int create_and_check(const lw_model* model, int32_t batch, int32_t width,
                            int32_t expected_steps, uint64_t* workspace_size) {
    lw_tensor_desc input;
    lw_tensor_desc output;
    lw_session_info info;
    lw_session* session = NULL;
    lw_error error;
    lw_status status;
    uint64_t lower_bound;
    make_rec_input(&input, batch, width);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed for width %" PRId32 ": %s: %s\n", width,
                lw_status_string(status), error.message);
        return 0;
    }
    if (!check_execution_table(model, session)) {
        fprintf(stderr, "unexpected prepared execution table for width %" PRId32 "\n", width);
        lw_session_free(session);
        return 0;
    }
    lw_session_info_init(&info);
    lw_tensor_desc_init(&output);
    lower_bound = semantic_workspace_lower_bound(model, session);
    if (lw_session_get_info(session, &info) != LW_STATUS_OK ||
        lw_session_get_output_desc(session, 0u, &output) != LW_STATUS_OK ||
        info.tensor_count != 274u || info.input_count != 1u || info.output_count != 1u ||
        info.workspace_size == 0u || (info.workspace_size & 63u) != 0u ||
        output.dtype != LW_DTYPE_F32 || output.rank != 3u || output.dimensions[0] != batch ||
        output.dimensions[1] != expected_steps || output.dimensions[2] != 6906 ||
        lower_bound == UINT64_MAX || info.workspace_size < lower_bound) {
        fprintf(stderr, "unexpected session plan or output shape for width %" PRId32 "\n", width);
        lw_session_free(session);
        return 0;
    }
    *workspace_size = info.workspace_size;
    lw_session_free(session);
    return 1;
}

static int check_plan_determinism(const lw_model* model) {
    lw_tensor_desc input;
    lw_session* first = NULL;
    lw_session* second = NULL;
    lw_error error;
    lw_status status;
    uint32_t i;
    make_rec_input(&input, 1, 320);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &first, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "first deterministic session create failed: %s\n", error.message);
        return 0;
    }
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &second, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "second deterministic session create failed: %s\n", error.message);
        lw_session_free(first);
        return 0;
    }
    if (first->workspace_bytes != second->workspace_bytes) {
        fprintf(stderr, "workspace planner is not deterministic\n");
        lw_session_free(second);
        lw_session_free(first);
        return 0;
    }
    for (i = 0u; i < model->info.tensor_count; ++i) {
        const lw_runtime_tensor* left = &first->tensors[i];
        const lw_runtime_tensor* right = &second->tensors[i];
        if (left->workspace_offset != right->workspace_offset ||
            left->birth_node != right->birth_node ||
            left->last_use_node != right->last_use_node ||
            left->workspace_live != right->workspace_live) {
            fprintf(stderr, "workspace planner changed tensor %" PRIu32 " between creates\n", i);
            lw_session_free(second);
            lw_session_free(first);
            return 0;
        }
    }
    lw_session_free(second);
    lw_session_free(first);
    return 1;
}

static int check_prepared_constant_sharing(const lw_model* model) {
    lw_tensor_desc first_input;
    lw_tensor_desc second_input;
    lw_session* first = NULL;
    lw_session* second = NULL;
    lw_error error;
    lw_status status;
    if (model == NULL) {
        return 0;
    }
    make_rec_input(&first_input, 1, 320);
    make_rec_input(&second_input, 1, 960);
    lw_error_init(&error);
    status = lw_session_create(model, &first_input, 1u, NULL, &first, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "prepared sharing source create failed: %s\n", error.message);
        return 0;
    }
    lw_error_init(&error);
    status = lw_session_create_with_prepared_source(
        model, &second_input, 1u, NULL, 0u, first, &second, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "prepared sharing candidate create failed: %s\n", error.message);
        lw_session_free(first);
        return 0;
    }
    if (first->shared_prepared_constants == NULL || second->shared_prepared_constants == NULL) {
        /* Scalar/unsupported-ISA builds legitimately have no packed arena. */
        lw_session_free(second);
        lw_session_free(first);
        return 1;
    }
    if (!lw_session_prepared_constants_compatible(second, first)) {
        fprintf(stderr, "prepared constants unexpectedly differ across REC widths\n");
        lw_session_free(second);
        lw_session_free(first);
        return 0;
    }
    if (second->packed_weights != first->packed_weights ||
        lw_atomic_u32_load_acquire(&first->shared_prepared_constants->ref_count) != 2u) {
        fprintf(stderr, "prepared constants sharing during create failed\n");
        lw_session_free(second);
        lw_session_free(first);
        return 0;
    }
    lw_session_free(second);
    if (lw_atomic_u32_load_acquire(&first->shared_prepared_constants->ref_count) != 1u) {
        fprintf(stderr, "prepared constants reference count did not release\n");
        lw_session_free(first);
        return 0;
    }
    lw_session_free(first);
    return 1;
}

typedef struct prepared_share_stress_context {
    const lw_model* model;
    const lw_session* source;
    volatile uint32_t failures;
} prepared_share_stress_context;

static void prepared_share_stress_callback(void* opaque, uint32_t worker_index,
                                           uint32_t worker_count) {
    prepared_share_stress_context* context = (prepared_share_stress_context*)opaque;
    const int32_t widths[] = {640, 960};
    uint32_t iteration;
    (void)worker_index;
    (void)worker_count;
    for (iteration = 0u; iteration < 16u; ++iteration) {
        uint32_t width_index;
        for (width_index = 0u; width_index < 2u; ++width_index) {
            lw_tensor_desc input;
            lw_session* session = NULL;
            lw_error error;
            lw_status status;
            make_rec_input(&input, 1, widths[width_index]);
            lw_error_init(&error);
            status = lw_session_create_with_prepared_source(
                context->model, &input, 1u, NULL, 0u, context->source, &session, &error);
            if (status != LW_STATUS_OK || session == NULL) {
                uint32_t expected = context->failures;
                while (!lw_atomic_u32_compare_exchange_acq_rel(
                    &context->failures, &expected, expected + 1u)) {
                }
                continue;
            }
            if (context->source->shared_prepared_constants != NULL &&
                session->shared_prepared_constants != context->source->shared_prepared_constants) {
                uint32_t expected = context->failures;
                while (!lw_atomic_u32_compare_exchange_acq_rel(
                    &context->failures, &expected, expected + 1u)) {
                }
                lw_session_free(session);
                continue;
            }
            lw_session_free(session);
        }
    }
}

static int check_prepared_constant_sharing_stress(const lw_model* model) {
    lw_tensor_desc input;
    lw_session* source = NULL;
    lw_thread_pool* pool = NULL;
    prepared_share_stress_context context;
    lw_error error;
    lw_status status;

    make_rec_input(&input, 1, 320);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &source, &error);
    if (status != LW_STATUS_OK || source == NULL) {
        fprintf(stderr, "prepared sharing stress source create failed: %s\n", error.message);
        return 0;
    }
    lw_error_init(&error);
    status = lw_session_share_prepared_constants(source, source, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "prepared self-share failed: %s\n", error.message);
        lw_session_free(source);
        return 0;
    }
    pool = lw_thread_pool_create(4u);
    if (pool == NULL) {
        lw_session_free(source);
        return 1;
    }
    context.model = model;
    context.source = source;
    context.failures = 0u;
    lw_thread_pool_run(pool, 4u, prepared_share_stress_callback, &context);
    lw_thread_pool_free(pool);
    if (context.failures != 0u ||
        (source->shared_prepared_constants != NULL &&
         lw_atomic_u32_load_acquire(&source->shared_prepared_constants->ref_count) != 1u)) {
        fprintf(stderr, "prepared sharing stress failed: %" PRIu32 "\n", context.failures);
        lw_session_free(source);
        return 0;
    }
    lw_session_free(source);
    return 1;
}
static int check_prepared_source_release_order(const lw_model* model) {
    lw_tensor_desc source_input;
    lw_tensor_desc child_input;
    lw_session* source = NULL;
    lw_session* child = NULL;
    lw_shared_prepared_constants* shared = NULL;
    lw_error error;
    lw_status status;

    make_rec_input(&source_input, 1, 320);
    make_rec_input(&child_input, 1, 960);
    lw_error_init(&error);
    status = lw_session_create(model, &source_input, 1u, NULL, &source, &error);
    if (status != LW_STATUS_OK || source == NULL) {
        fprintf(stderr, "source-release test: source create failed: %s\n", error.message);
        return 0;
    }
    if (source->shared_prepared_constants == NULL) {
        lw_session_free(source);
        return 1;
    }
    lw_error_init(&error);
    status = lw_session_create_with_prepared_source(
        model, &child_input, 1u, NULL, 0u, source, &child, &error);
    if (status != LW_STATUS_OK || child == NULL) {
        fprintf(stderr, "source-release test: child create failed: %s\n", error.message);
        lw_session_free(child);
        lw_session_free(source);
        return 0;
    }
    shared = child->shared_prepared_constants;
    if (shared == NULL || shared != source->shared_prepared_constants ||
        lw_atomic_u32_load_acquire(&shared->ref_count) != 2u) {
        fprintf(stderr, "source-release test: arena/refcount mismatch\n");
        lw_session_free(child);
        lw_session_free(source);
        return 0;
    }
    lw_session_free(source);
    source = NULL;
    if (lw_atomic_u32_load_acquire(&shared->ref_count) != 1u ||
        child->prepared_constants != shared->constants ||
        child->packed_weights != shared->packed_weights) {
        fprintf(stderr, "source-release test: child bindings became invalid\n");
        lw_session_free(child);
        return 0;
    }
    lw_session_free(child);
    return 1;
}
int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_error error;
    lw_status status;
    uint64_t workspace_320 = 0u;
    uint64_t workspace_odd = 0u;
    uint64_t workspace_minimum = 0u;
    uint64_t workspace_640 = 0u;
    lw_tensor_desc input;
    lw_session_options options;
    lw_session* session = NULL;
    uint32_t i;
    if (argc != 2) {
        fprintf(stderr, "expected path to rec.lwm\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    if (!check_plan_determinism(model) || !check_prepared_constant_sharing(model) ||
        !check_prepared_constant_sharing_stress(model) ||
        !check_prepared_source_release_order(model) ||
        !create_and_check(model, 1, 7, 1, &workspace_minimum) ||
        !create_and_check(model, 1, 320, 40, &workspace_320) ||
        !create_and_check(model, 1, 321, 40, &workspace_odd) ||
        !create_and_check(model, 2, 640, 80, &workspace_640) || workspace_odd < workspace_320 ||
        workspace_640 <= workspace_320) {
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 320);
    lw_session_options_init(&options);
    options.max_workspace_size = workspace_320 - 1u;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, &options, &session, &error);
    if (status != LW_STATUS_MEMORY_LIMIT || session != NULL) {
        fprintf(stderr, "workspace memory limit was not enforced\n");
        lw_model_free(model);
        return 1;
    }

    lw_session_options_init(&options);
    options.max_tensor_size = 1024u;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, &options, &session, &error);
    if (status != LW_STATUS_MEMORY_LIMIT || session != NULL) {
        fprintf(stderr, "single tensor memory limit was not enforced\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 320);
    input.dimensions[2] = 47;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_INVALID_SHAPE || session != NULL) {
        fprintf(stderr, "invalid REC input height was not rejected\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 1);
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_INVALID_SHAPE || session != NULL) {
        fprintf(stderr, "too-small REC input width was not rejected\n");
        lw_model_free(model);
        return 1;
    }

    make_rec_input(&input, 1, 64);
    for (i = 0u; i < 100u; ++i) {
        lw_error_init(&error);
        status = lw_session_create(model, &input, 1u, NULL, &session, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "repeated session create failed at iteration %" PRIu32 ": %s\n", i,
                    error.message);
            lw_model_free(model);
            return 1;
        }
        lw_session_free(session);
        session = NULL;
    }
    printf("workspace_320=%" PRIu64 " workspace_batch2_640=%" PRIu64 "\n", workspace_320,
           workspace_640);
    lw_model_free(model);
    return 0;
}
