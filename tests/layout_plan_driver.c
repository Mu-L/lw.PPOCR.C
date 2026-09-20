#include "layout_planner_internal.h"
#include "model_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"

#include "lw_infer.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_width(const char* text, int32_t* width) {
    char* end = NULL;
    long value;
    if (text == NULL || width == NULL) {
        return 0;
    }
    value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1 || value > INT32_MAX) {
        return 0;
    }
    *width = (int32_t)value;
    return 1;
}

static const char* operation_name(uint16_t operation) {
    switch (operation) {
    case LW_OP_CONV: return "Conv";
    case LW_OP_ADD: return "Add";
    case LW_OP_MUL: return "Mul";
    case LW_OP_DIV: return "Div";
    case LW_OP_ERF: return "Erf";
    case LW_OP_HARD_SIGMOID: return "HardSigmoid";
    case LW_OP_BATCH_NORMALIZATION: return "BatchNormalization";
    case LW_OP_REDUCE_MEAN: return "ReduceMean";
    case LW_OP_RELU: return "Relu";
    case LW_OP_AVERAGE_POOL: return "AveragePool";
    case LW_OP_SQUEEZE: return "Squeeze";
    case LW_OP_TRANSPOSE: return "Transpose";
    case LW_OP_UNSQUEEZE: return "Unsqueeze";
    case LW_OP_MATMUL: return "MatMul";
    case LW_OP_SOFTMAX: return "Softmax";
    case LW_OP_RESHAPE: return "Reshape";
    case LW_OP_CONCAT: return "Concat";
    case LW_OP_CONV_TRANSPOSE: return "ConvTranspose";
    case LW_OP_MAX_POOL: return "MaxPool";
    case LW_OP_RESIZE: return "Resize";
    case LW_OP_SIGMOID: return "Sigmoid";
    case LW_OP_SUB: return "Sub";
    case LW_OP_SQRT: return "Sqrt";
    case LW_OP_POW: return "Pow";
    case LW_OP_SLICE: return "Slice";
    default: return "Unknown";
    }
}

static void dump_nodes(const lw_model* model, const lw_layout_plan* plan) {
    uint32_t node_index;
    printf("nodes:\n");
    for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
        const uint8_t* node = model->bytes + (size_t)model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        uint16_t input_count = lwm_read_u16(node + 2u);
        uint16_t input_slot;
        printf("%03" PRIu32 " %-18s %s inputs=", node_index,
               operation_name(lwm_read_u16(node)),
               plan->node_layout[node_index] == LW_FAST_LAYOUT_NHWC ? "NHWC" : "NCHW");
        for (input_slot = 0u; input_slot < input_count && input_slot < LWM_V0_MAX_NODE_INPUTS;
             ++input_slot) {
            if (input_slot != 0u) {
                putchar(',');
            }
            printf("%" PRIu32, lwm_read_u32(node + 8u + (size_t)input_slot * 4u));
        }
        printf(" output=%" PRIu32 "\n", lwm_read_u32(node + 40u));
    }
}
int main(int argc, char** argv) {
    const char* model_path;
    int32_t width = 960;
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input;
    lw_layout_planner_options options;
    lw_layout_plan plan;
    lw_error error;
    lw_status status;
    uint32_t node;
    uint32_t tensor;
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s model.lwm [width] [direct-nhwc]\n", argv[0]);
        return 2;
    }
    model_path = argv[1];
    if (argc >= 3 && !parse_width(argv[2], &width)) {
        fprintf(stderr, "invalid width: %s\n", argv[2]);
        return 2;
    }
    memset(&options, 0, sizeof(options));
    if (argc == 4 && strcmp(argv[3], "direct-nhwc") == 0) {
        options.allow_direct_nhwc_graph_input = 1u;
    } else if (argc == 4) {
        fprintf(stderr, "unknown option: %s\n", argv[3]);
        return 2;
    }
    memset(&plan, 0, sizeof(plan));
    lw_error_init(&error);
    status = lw_model_load(model_path, NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = 1;
    input.dimensions[1] = 3;
    input.dimensions[2] = 48;
    input.dimensions[3] = width;
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        lw_model_free(model);
        return 1;
    }
    lw_error_init(&error);
    status = lw_layout_plan_build(session, &options, &plan, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "layout plan failed: %s: %s\n", lw_status_string(status), error.message);
        lw_session_free(session);
        lw_model_free(model);
        return 1;
    }
    if (plan.tensor_count != model->info.tensor_count || plan.node_count != model->info.node_count ||
        plan.tensor_layout == NULL || plan.tensor_available_layouts == NULL ||
        plan.node_layout == NULL) {
        fprintf(stderr, "layout plan has invalid dimensions\n");
        lw_layout_plan_free(&plan);
        lw_session_free(session);
        lw_model_free(model);
        return 1;
    }
    for (node = 0u; node < plan.node_count; ++node) {
        if (plan.node_layout[node] != LW_FAST_LAYOUT_NCHW &&
            plan.node_layout[node] != LW_FAST_LAYOUT_NHWC) {
            fprintf(stderr, "node %" PRIu32 " has invalid layout\n", node);
            lw_layout_plan_free(&plan);
            lw_session_free(session);
            lw_model_free(model);
            return 1;
        }
    }
    for (tensor = 0u; tensor < plan.tensor_count; ++tensor) {
        if (plan.tensor_layout[tensor] != LW_FAST_LAYOUT_NCHW &&
            plan.tensor_layout[tensor] != LW_FAST_LAYOUT_NHWC) {
            fprintf(stderr, "tensor %" PRIu32 " has invalid layout\n", tensor);
            lw_layout_plan_free(&plan);
            lw_session_free(session);
            lw_model_free(model);
            return 1;
        }
    }
    printf("model=%s input=[1,3,48,%" PRId32 "]\n", model_path, width);
    printf("layout_plan tensor_count=%" PRIu32 " node_count=%" PRIu32
           " nhwc_nodes=%" PRIu32 " nchw_nodes=%" PRIu32
           " conversions=%" PRIu32 " islands=%" PRIu32
           " direct_input=%u width=%" PRId32 "\n",
           plan.tensor_count, plan.node_count, plan.nhwc_node_count, plan.nchw_node_count,
           plan.layout_conversion_count, plan.nhwc_island_count,
           (unsigned)plan.graph_input_direct_nhwc, width);
    dump_nodes(model, &plan);
    lw_layout_plan_free(&plan);
    lw_session_free(session);
    lw_model_free(model);
    return 0;
}