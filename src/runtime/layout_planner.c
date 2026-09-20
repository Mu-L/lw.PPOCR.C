#include "layout_planner_internal.h"

#include "lwm_read.h"
#include "model_internal.h"
#include "operator_internal.h"
#include "session_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t* layout_node_bytes(const lw_session* session, uint32_t node_index) {
    if (session == NULL || session->model == NULL || node_index >= session->model->info.node_count) {
        return NULL;
    }
    return session->model->bytes + (size_t)session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
}

static const uint8_t* layout_node_params(const lw_session* session, const uint8_t* node) {
    uint64_t offset;
    if (session == NULL || session->model == NULL || node == NULL) {
        return NULL;
    }
    offset = lwm_read_u64(node + 56u);
    if (offset == 0u || offset > session->model->byte_count ||
        session->model->byte_count - (size_t)offset < 4u) {
        return NULL;
    }
    return session->model->bytes + (size_t)offset;
}

static int layout_tensor_valid(const lw_session* session, uint32_t tensor_index) {
    return session != NULL && session->model != NULL && session->tensors != NULL &&
           tensor_index < session->model->info.tensor_count;
}

static int layout_tensor_is_constant(const lw_session* session, uint32_t tensor_index) {
    return layout_tensor_valid(session, tensor_index) &&
           (session->tensors[tensor_index].flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u;
}

static uint64_t layout_tensor_elements(const lw_runtime_tensor* tensor) {
    if (tensor == NULL || tensor->dtype != LW_DTYPE_F32 || tensor->byte_size % sizeof(float) != 0u) {
        return 0u;
    }
    return tensor->byte_size / sizeof(float);
}

static int layout_same_shape(const lw_runtime_tensor* left, const lw_runtime_tensor* right) {
    uint32_t dimension;
    if (left == NULL || right == NULL || left->rank != right->rank) {
        return 0;
    }
    for (dimension = 0u; dimension < left->rank && dimension < LW_MAX_DIMS; ++dimension) {
        if (left->dimensions[dimension] != right->dimensions[dimension]) {
            return 0;
        }
    }
    return 1;
}

static int layout_rank4_f32(const lw_runtime_tensor* tensor) {
    return tensor != NULL && tensor->dtype == LW_DTYPE_F32 && tensor->rank == 4u &&
           tensor->dimensions[0] > 0 && tensor->dimensions[1] > 0 &&
           tensor->dimensions[2] > 0 && tensor->dimensions[3] > 0;
}

int lw_layout_tensor_is_neutral(const lw_session* session, uint32_t tensor_index) {
    const lw_runtime_tensor* tensor;
    if (!layout_tensor_valid(session, tensor_index)) {
        return 0;
    }
    tensor = &session->tensors[tensor_index];
    return layout_rank4_f32(tensor) &&
           (tensor->dimensions[1] == 1 ||
            (tensor->dimensions[2] == 1 && tensor->dimensions[3] == 1));
}

static int layout_channel_vector(const lw_runtime_tensor* tensor,
                                 const lw_runtime_tensor* full) {
    if (!layout_rank4_f32(tensor) || !layout_rank4_f32(full)) {
        return 0;
    }
    return tensor->dimensions[1] == full->dimensions[1] && tensor->dimensions[2] == 1 &&
           tensor->dimensions[3] == 1 &&
           (tensor->dimensions[0] == 1 || tensor->dimensions[0] == full->dimensions[0]);
}

static int layout_channel_broadcast(const lw_session* session, uint32_t tensor_index,
                                    const lw_runtime_tensor* full) {
    const lw_runtime_tensor* tensor;
    if (!layout_tensor_valid(session, tensor_index) || full == NULL) {
        return 0;
    }
    tensor = &session->tensors[tensor_index];
    if (layout_tensor_elements(tensor) == 1u) {
        return 1;
    }
    if (layout_tensor_is_constant(session, tensor_index) &&
        layout_tensor_elements(tensor) == (uint64_t)(uint32_t)full->dimensions[1]) {
        return 1;
    }
    return layout_channel_vector(tensor, full);
}
static int layout_binary_compatible(const lw_session* session, uint32_t left_index,
                                    uint32_t right_index, uint32_t output_index) {
    const lw_runtime_tensor* left;
    const lw_runtime_tensor* right;
    const lw_runtime_tensor* output;
    if (!layout_tensor_valid(session, left_index) || !layout_tensor_valid(session, right_index) ||
        !layout_tensor_valid(session, output_index)) {
        return 0;
    }
    left = &session->tensors[left_index];
    right = &session->tensors[right_index];
    output = &session->tensors[output_index];
    if (!layout_rank4_f32(output)) {
        return 0;
    }
    if (layout_channel_broadcast(session, left_index, output) ||
        layout_channel_broadcast(session, right_index, output)) {
        return 1;
    }
    return layout_same_shape(left, right) && layout_same_shape(left, output);
}

static int layout_conv_capable(const lw_session* session, const uint8_t* node,
                               const uint8_t* params) {
    uint32_t input_count;
    uint32_t input_index;
    uint32_t weight_index;
    uint32_t group;
    int32_t kernel_height;
    int32_t kernel_width;
    int32_t stride_height;
    int32_t stride_width;
    int32_t dilation_height;
    int32_t dilation_width;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* weight;
    const lw_runtime_tensor* output;
    if (node == NULL || params == NULL || lwm_read_u16(node + 4u) != 1u) {
        return 0;
    }
    input_count = lwm_read_u16(node + 2u);
    if (input_count < 2u || input_count > 3u) {
        return 0;
    }
    input_index = lwm_read_u32(node + 8u);
    weight_index = lwm_read_u32(node + 12u);
    if (!layout_tensor_valid(session, input_index) || !layout_tensor_valid(session, weight_index) ||
        !layout_tensor_valid(session, lwm_read_u32(node + 40u)) ||
        !layout_tensor_is_constant(session, weight_index)) {
        return 0;
    }
    input = &session->tensors[input_index];
    weight = &session->tensors[weight_index];
    output = &session->tensors[lwm_read_u32(node + 40u)];
    if (!layout_rank4_f32(input) || !layout_rank4_f32(output) || weight->rank != 4u ||
        weight->dtype != LW_DTYPE_F32) {
        return 0;
    }
    group = lwm_read_u32(params + 4u);
    kernel_height = lwm_read_i32(params + 8u);
    kernel_width = lwm_read_i32(params + 12u);
    stride_height = lwm_read_i32(params + 16u);
    stride_width = lwm_read_i32(params + 20u);
    dilation_height = lwm_read_i32(params + 24u);
    dilation_width = lwm_read_i32(params + 28u);
    if (group == 0u || kernel_height <= 0 || kernel_width <= 0 || stride_height <= 0 ||
        stride_width <= 0 || dilation_height != 1 || dilation_width != 1 ||
        weight->dimensions[2] != kernel_height || weight->dimensions[3] != kernel_width) {
        return 0;
    }
    if (group == 1u) {
        return weight->dimensions[0] >= 16 && (weight->dimensions[0] & 15) == 0;
    }
    return group == (uint32_t)input->dimensions[1] && weight->dimensions[1] == 1 &&
           (group & 7u) == 0u;
}

static int layout_conv_transpose_capable(const lw_session* session, const uint8_t* node,
                                         const uint8_t* params) {
    uint32_t input_index;
    uint32_t weight_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* weight;
    const lw_runtime_tensor* output;
    if (node == NULL || params == NULL || lwm_read_u16(node + 2u) < 2u ||
        lwm_read_u16(node + 4u) != 1u) {
        return 0;
    }
    input_index = lwm_read_u32(node + 8u);
    weight_index = lwm_read_u32(node + 12u);
    if (!layout_tensor_valid(session, input_index) || !layout_tensor_valid(session, weight_index) ||
        !layout_tensor_valid(session, lwm_read_u32(node + 40u)) ||
        !layout_tensor_is_constant(session, weight_index)) {
        return 0;
    }
    input = &session->tensors[input_index];
    weight = &session->tensors[weight_index];
    output = &session->tensors[lwm_read_u32(node + 40u)];
    return layout_rank4_f32(input) && layout_rank4_f32(output) && weight->rank == 4u &&
           weight->dtype == LW_DTYPE_F32 && lwm_read_u32(params + 4u) == 1u &&
           lwm_read_i32(params + 8u) == 2 && lwm_read_i32(params + 12u) == 2 &&
           lwm_read_i32(params + 16u) == 2 && lwm_read_i32(params + 20u) == 2 &&
           lwm_read_i32(params + 24u) == 1 && lwm_read_i32(params + 28u) == 1 &&
           lwm_read_i32(params + 32u) == 0 && lwm_read_i32(params + 36u) == 0 &&
           lwm_read_i32(params + 40u) == 0 && lwm_read_i32(params + 44u) == 0;
}

static int layout_elementwise_capable(const lw_session* session, const uint8_t* node,
                                      uint16_t operation) {
    uint32_t input_count = lwm_read_u16(node + 2u);
    uint32_t output_index = lwm_read_u32(node + 40u);
    uint32_t input_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    if (input_count == 0u || lwm_read_u16(node + 4u) != 1u ||
        !layout_tensor_valid(session, output_index)) {
        return 0;
    }
    input_index = lwm_read_u32(node + 8u);
    if (!layout_tensor_valid(session, input_index)) {
        return 0;
    }
    input = &session->tensors[input_index];
    output = &session->tensors[output_index];
    if (!layout_rank4_f32(input) || !layout_rank4_f32(output) ||
        !layout_same_shape(input, output)) {
        return 0;
    }
    if (operation == LW_OP_ADD || operation == LW_OP_MUL || operation == LW_OP_DIV ||
        operation == LW_OP_SUB || operation == LW_OP_POW) {
        return input_count == 2u &&
               layout_binary_compatible(session, input_index, lwm_read_u32(node + 12u),
                                        output_index);
    }
    return operation == LW_OP_RELU || operation == LW_OP_SIGMOID || operation == LW_OP_SQRT ||
           operation == LW_OP_ERF || operation == LW_OP_HARD_SIGMOID;
}

int lw_layout_node_nhwc_capable(const lw_session* session, uint32_t node_index) {
    const uint8_t* node = layout_node_bytes(session, node_index);
    const uint8_t* params = layout_node_params(session, node);
    uint16_t operation;
    uint32_t input_index;
    uint32_t output_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    if (node == NULL) {
        return 0;
    }
    operation = lwm_read_u16(node);
    input_index = lwm_read_u32(node + 8u);
    output_index = lwm_read_u32(node + 40u);
    switch (operation) {
    case LW_OP_CONV:
        return layout_conv_capable(session, node, params);
    case LW_OP_CONV_TRANSPOSE:
        return layout_conv_transpose_capable(session, node, params);
    case LW_OP_ADD:
    case LW_OP_MUL:
    case LW_OP_DIV:
    case LW_OP_SUB:
    case LW_OP_POW:
    case LW_OP_RELU:
    case LW_OP_SIGMOID:
    case LW_OP_SQRT:
    case LW_OP_ERF:
    case LW_OP_HARD_SIGMOID:
        return layout_elementwise_capable(session, node, operation);
    case LW_OP_BATCH_NORMALIZATION:
        if (lwm_read_u16(node + 2u) < 5u || lwm_read_u16(node + 4u) != 1u ||
            !layout_tensor_valid(session, input_index) || !layout_tensor_valid(session, output_index)) {
            return 0;
        }
        input = &session->tensors[input_index];
        output = &session->tensors[output_index];
        return layout_rank4_f32(input) && layout_rank4_f32(output) &&
               layout_same_shape(input, output);
    case LW_OP_AVERAGE_POOL:
    case LW_OP_MAX_POOL:
        if (params == NULL || lwm_read_u16(node + 2u) != 1u || lwm_read_u16(node + 4u) != 1u ||
            !layout_tensor_valid(session, input_index) || !layout_tensor_valid(session, output_index)) {
            return 0;
        }
        input = &session->tensors[input_index];
        output = &session->tensors[output_index];
        return layout_rank4_f32(input) && layout_rank4_f32(output) &&
               lwm_read_i32(params + 8u) > 0 && lwm_read_i32(params + 12u) > 0 &&
               lwm_read_i32(params + 16u) > 0 && lwm_read_i32(params + 20u) > 0;
    case LW_OP_RESIZE:
        if (params == NULL || lwm_read_u16(node + 2u) != 1u || lwm_read_u16(node + 4u) != 1u ||
            !layout_tensor_valid(session, input_index) || !layout_tensor_valid(session, output_index) ||
            lwm_read_u16(params + 2u) != 4u) {
            return 0;
        }
        input = &session->tensors[input_index];
        output = &session->tensors[output_index];
        return layout_rank4_f32(input) && layout_rank4_f32(output) &&
               lwm_read_i32(params + 4u) == 1 && lwm_read_i32(params + 8u) == 1;
    default:
        return 0;
    }
}

static int layout_dense_conv_node(const lw_session* session, const uint8_t* node) {
    const uint8_t* params = layout_node_params(session, node);
    return node != NULL && lwm_read_u16(node) == LW_OP_CONV && params != NULL &&
           lwm_read_u32(params + 4u) == 1u;
}

static int layout_node_activation_input(const lw_session* session, const uint8_t* node,
                                        uint32_t input_slot, uint32_t* tensor_index) {
    uint32_t input_count;
    uint32_t candidate;
    if (node == NULL || tensor_index == NULL) {
        return 0;
    }
    input_count = lwm_read_u16(node + 2u);
    if (input_slot >= input_count || input_slot >= LWM_V0_MAX_NODE_INPUTS) {
        return 0;
    }
    candidate = lwm_read_u32(node + 8u + (size_t)input_slot * sizeof(uint32_t));
    if (!layout_tensor_valid(session, candidate) || layout_tensor_is_constant(session, candidate)) {
        return 0;
    }
    *tensor_index = candidate;
    return 1;
}

static void layout_plan_mark_conversion(lw_layout_plan* plan, uint8_t* availability) {
    if (availability != NULL) {
        *availability = (uint8_t)(LW_LAYOUT_AVAILABLE_NCHW | LW_LAYOUT_AVAILABLE_NHWC);
    }
    if (plan->layout_conversion_count != UINT32_MAX) {
        ++plan->layout_conversion_count;
    }
}

void lw_layout_plan_free(lw_layout_plan* plan) {
    if (plan == NULL) {
        return;
    }
    free(plan->tensor_layout);
    free(plan->tensor_available_layouts);
    free(plan->node_layout);
    memset(plan, 0, sizeof(*plan));
}

lw_status lw_layout_plan_build(const lw_session* session,
                               const lw_layout_planner_options* options,
                               lw_layout_plan* plan, lw_error* error) {
    lw_layout_planner_options defaults;
    uint32_t tensor_index;
    uint32_t node_index;
    int in_nhwc_island = 0;
    if (session == NULL || session->model == NULL || session->tensors == NULL || plan == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "layout planner requires a session and plan");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    memset(&defaults, 0, sizeof(defaults));
    if (options == NULL) {
        options = &defaults;
    }
    lw_layout_plan_free(plan);
    plan->tensor_count = session->model->info.tensor_count;
    plan->node_count = session->model->info.node_count;
    if (plan->tensor_count != 0u) {
        plan->tensor_layout = (uint8_t*)calloc(plan->tensor_count, sizeof(uint8_t));
        plan->tensor_available_layouts =
            (uint8_t*)calloc(plan->tensor_count, sizeof(uint8_t));
    }
    if (plan->node_count != 0u) {
        plan->node_layout = (uint8_t*)calloc(plan->node_count, sizeof(uint8_t));
    }
    if ((plan->tensor_count != 0u &&
         (plan->tensor_layout == NULL || plan->tensor_available_layouts == NULL)) ||
        (plan->node_count != 0u && plan->node_layout == NULL)) {
        lw_layout_plan_free(plan);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate layout plan");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (tensor_index = 0u; tensor_index < plan->tensor_count; ++tensor_index) {
        plan->tensor_layout[tensor_index] = LW_FAST_LAYOUT_NCHW;
        plan->tensor_available_layouts[tensor_index] = LW_LAYOUT_AVAILABLE_NCHW;
        if ((session->tensors[tensor_index].flags & LWM_V0_TENSOR_FLAG_INPUT) != 0u &&
            options->allow_direct_nhwc_graph_input != 0u &&
            layout_rank4_f32(&session->tensors[tensor_index])) {
            plan->tensor_layout[tensor_index] = LW_FAST_LAYOUT_NHWC;
            plan->tensor_available_layouts[tensor_index] =
                (uint8_t)(LW_LAYOUT_AVAILABLE_NCHW | LW_LAYOUT_AVAILABLE_NHWC);
            plan->graph_input_direct_nhwc = 1u;
        }
    }
    for (node_index = 0u; node_index < plan->node_count; ++node_index) {
        const uint8_t* node = layout_node_bytes(session, node_index);
        const uint16_t operation = node == NULL ? 0u : lwm_read_u16(node);
        const uint32_t output_index = node == NULL ? UINT32_MAX : lwm_read_u32(node + 40u);
        const int capable = lw_layout_node_nhwc_capable(session, node_index);
        int has_nhwc_input = 0;
        int has_nchw_input = 0;
        uint32_t input_slot;
        int choose_nhwc = 0;
        for (input_slot = 0u; node != NULL && input_slot < lwm_read_u16(node + 2u); ++input_slot) {
            uint32_t input_index;
            if (!layout_node_activation_input(session, node, input_slot, &input_index)) {
                continue;
            }
            if (plan->tensor_layout[input_index] == LW_FAST_LAYOUT_NHWC) {
                has_nhwc_input = 1;
            } else {
                has_nchw_input = 1;
            }
        }
        if (capable && (has_nhwc_input || layout_dense_conv_node(session, node) ||
                        (operation == LW_OP_CONV_TRANSPOSE && has_nhwc_input))) {
            choose_nhwc = 1;
        }
        if (choose_nhwc) {
            plan->node_layout[node_index] = LW_FAST_LAYOUT_NHWC;
            if (plan->nhwc_node_count != UINT32_MAX) {
                ++plan->nhwc_node_count;
            }
            if (!in_nhwc_island && plan->nhwc_island_count != UINT32_MAX) {
                ++plan->nhwc_island_count;
            }
            in_nhwc_island = 1;
            for (input_slot = 0u; input_slot < lwm_read_u16(node + 2u); ++input_slot) {
                uint32_t input_index;
                if (layout_node_activation_input(session, node, input_slot, &input_index) &&
                    plan->tensor_layout[input_index] != LW_FAST_LAYOUT_NHWC) {
                    layout_plan_mark_conversion(plan,
                                                &plan->tensor_available_layouts[input_index]);
                }
            }
            if (output_index < plan->tensor_count) {
                plan->tensor_layout[output_index] = LW_FAST_LAYOUT_NHWC;
                plan->tensor_available_layouts[output_index] = LW_LAYOUT_AVAILABLE_NHWC;
            }
        } else {
            plan->node_layout[node_index] = LW_FAST_LAYOUT_NCHW;
            if (plan->nchw_node_count != UINT32_MAX) {
                ++plan->nchw_node_count;
            }
            in_nhwc_island = 0;
            for (input_slot = 0u; node != NULL && input_slot < lwm_read_u16(node + 2u); ++input_slot) {
                uint32_t input_index;
                if (layout_node_activation_input(session, node, input_slot, &input_index) &&
                    plan->tensor_layout[input_index] == LW_FAST_LAYOUT_NHWC) {
                    layout_plan_mark_conversion(plan,
                                                &plan->tensor_available_layouts[input_index]);
                }
            }
            if (output_index < plan->tensor_count) {
                plan->tensor_layout[output_index] = LW_FAST_LAYOUT_NCHW;
                plan->tensor_available_layouts[output_index] = LW_LAYOUT_AVAILABLE_NCHW;
            }
        }
        (void)has_nchw_input;
    }
    for (tensor_index = 0u; tensor_index < plan->tensor_count; ++tensor_index) {
        if ((session->tensors[tensor_index].flags & LWM_V0_TENSOR_FLAG_OUTPUT) != 0u &&
            plan->tensor_layout[tensor_index] == LW_FAST_LAYOUT_NHWC) {
            layout_plan_mark_conversion(plan,
                                        &plan->tensor_available_layouts[tensor_index]);
            plan->tensor_layout[tensor_index] = LW_FAST_LAYOUT_NCHW;
        }
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}