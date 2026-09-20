#include "lw_infer.h"
#include "lwm_read.h"
#include "session_internal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_i32(const char* text, int32_t* value) {
    char* end = NULL;
    long parsed;
    if (text == NULL || *text == '\0') {
        return 0;
    }
    parsed = strtol(text, &end, 10);
    if (*end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return 0;
    }
    *value = (int32_t)parsed;
    return 1;
}

static uint64_t align_bytes(uint64_t value) {
    if (value > UINT64_MAX - (LW_WORKSPACE_ALIGNMENT - 1u)) {
        return UINT64_MAX;
    }
    return (value + (LW_WORKSPACE_ALIGNMENT - 1u)) &
           ~(uint64_t)(LW_WORKSPACE_ALIGNMENT - 1u);
}

static uint64_t semantic_live_lower_bound(const lw_session* session, uint32_t* peak_node) {
    uint64_t maximum = 0u;
    uint32_t node;
    for (node = 0u; node <= session->model->info.node_count; ++node) {
        uint64_t live = 0u;
        uint32_t tensor_index;
        for (tensor_index = 0u; tensor_index < session->model->info.tensor_count;
             ++tensor_index) {
            const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
            uint64_t size;
            if (tensor->birth_node < 0 || tensor->birth_node > (int32_t)node ||
                tensor->last_use_node < (int32_t)node) {
                continue;
            }
            if (lw_execution_plan_is_skipped_tensor(session, tensor_index)) {
                continue;
            }
            size = align_bytes(tensor->byte_size);
            if (size == UINT64_MAX || live > UINT64_MAX - size) {
                return UINT64_MAX;
            }
            live += size;
        }
        if (live > maximum) {
            maximum = live;
            if (peak_node != NULL) {
                *peak_node = node;
            }
        }
    }
    return maximum;
}

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input;
    lw_error error;
    lw_status status;
    int32_t width = 320;
    int32_t batch = 1;
    uint64_t lower_bound;
    uint32_t peak_node = 0u;
    uint32_t prepared_constant_nodes = 0u;
    uint32_t node_index;
    uint32_t prepared_constants_ref_count = 0u;
    int ctc_greedy = 0;
    int ctc_skip = 0;

    if (argc >= 5 && strcmp(argv[4], "--ctc-greedy") == 0) {
        ctc_greedy = 1;
    }
    if (argc < 2 || argc > 5 || (argc >= 3 && !parse_i32(argv[2], &width)) ||
        (argc >= 4 && !parse_i32(argv[3], &batch)) || (argc == 5 && !ctc_greedy)) {
        fprintf(stderr,
                "usage: workspace-report-driver <model.lwm> [width] [batch] [--ctc-greedy]\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = batch;
    input.dimensions[1] = 3;
    input.dimensions[2] = 48;
    input.dimensions[3] = width;
    lw_error_init(&error);
    status = ctc_greedy ? lw_session_create_ctc_greedy(model, &input, 1u, NULL, &session, &error)
                        : lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        lw_model_free(model);
        return 1;
    }
    ctc_skip = lw_execution_plan_is_ctc_greedy_skip(session,
                                                     lwm_read_u32(model->bytes +
                                                                  (size_t)model->output_offset));
#if defined(LW_EXPERIMENTAL_CTC_TILED)
    if (ctc_greedy && !ctc_skip) {
        fprintf(stderr, "CTC greedy plan did not elide the terminal output tensor\n");
        lw_session_free(session);
        lw_model_free(model);
        return 1;
    }
#endif
    lower_bound = semantic_live_lower_bound(session, &peak_node);
    if (session->prepared_constants != NULL) {
        for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
            if (session->prepared_constants[node_index].kind != LW_PREPARED_CONSTANT_NONE) {
                ++prepared_constant_nodes;
            }
        }
    }
    if (session->shared_prepared_constants != NULL) {
        prepared_constants_ref_count = session->shared_prepared_constants->ref_count;
    }
    printf("{\"schema_version\":1,\"model_file_bytes\":%" PRIu64
           ",\"tensor_count\":%" PRIu32 ",\"node_count\":%" PRIu32
           ",\"batch\":%" PRId32 ",\"width\":%" PRId32
           ",\"workspace_bytes\":%zu,\"semantic_live_lower_bound\":%" PRIu64
           ",\"semantic_peak_node\":%" PRIu32 ",\"ctc_greedy\":%s"
           ",\"ctc_skip_tensor\":%s"
           ",\"packed_weight_bytes\":%zu,\"prepared_constant_nodes\":%" PRIu32
           ",\"prepared_constants_ref_count\":%" PRIu32
           ",\"model_checksum\":\"0x%016" PRIx64 "\"}\n",
           model->info.file_size, model->info.tensor_count, model->info.node_count, batch, width,
           session->workspace_bytes, lower_bound, peak_node, ctc_greedy ? "true" : "false",
           ctc_skip ? "true" : "false", session->packed_weight_bytes, prepared_constant_nodes,
           prepared_constants_ref_count, model->info.content_checksum);
    lw_session_free(session);
    lw_model_free(model);
    return 0;
}
