#include "x64_rec_fast_internal.h"

#include "executor_internal.h"
#include "lwm_read.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fast_element_count(const lw_runtime_tensor* tensor) {
    return tensor == NULL || tensor->dtype != LW_DTYPE_F32 ? 0u : tensor->byte_size / sizeof(float);
}

static float* fast_nhwc_pointer(const lw_x64_rec_fast_plan* plan, uint32_t tensor_index) {
    const lw_x64_fast_tensor_state* state;
    if (plan == NULL || tensor_index >= plan->tensor_count) {
        return NULL;
    }
    state = &plan->tensors[tensor_index];
    if (state->neutral) {
        return (float*)lw_executor_tensor_output_data(plan->session, tensor_index);
    }
    if (state->nhwc_offset == LW_X64_FAST_OFFSET_NONE || plan->nhwc_workspace == NULL) {
        return NULL;
    }
    return (float*)(void*)(plan->nhwc_workspace + (size_t)state->nhwc_offset);
}

static int fast_layout_tracked_tensor(const lw_runtime_tensor* tensor) {
    return tensor != NULL && tensor->dtype == LW_DTYPE_F32 && tensor->rank == 4u &&
           tensor->dimensions[0] > 0 && tensor->dimensions[1] > 0 &&
           tensor->dimensions[2] > 0 && tensor->dimensions[3] > 0;
}

static void fast_publish_nchw(lw_x64_rec_fast_plan* plan, uint32_t tensor_index) {
    lw_x64_fast_tensor_state* state;
    if (plan == NULL || tensor_index >= plan->tensor_count) return;
    state = &plan->tensors[tensor_index];
    state->available_layouts = state->neutral
        ? (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC)
        : LW_X64_FAST_HAVE_NCHW;
}

static void fast_publish_nhwc(lw_x64_rec_fast_plan* plan, uint32_t tensor_index) {
    lw_x64_fast_tensor_state* state;
    if (plan == NULL || tensor_index >= plan->tensor_count) return;
    state = &plan->tensors[tensor_index];
    state->available_layouts = state->neutral
        ? (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC)
        : LW_X64_FAST_HAVE_NHWC;
}

static void fast_reset_layouts(lw_x64_rec_fast_plan* plan) {
    uint32_t index;
    if (plan == NULL) return;
    plan->conversion_count = 0u;
    plan->conversion_bytes = 0u;
    for (index = 0u; index < plan->tensor_count; ++index) {
        plan->tensors[index].available_layouts = 0u;
    }
    fast_publish_nchw(plan, plan->graph_input_index);
}

static void fast_add_conversion(lw_x64_rec_fast_plan* plan, uint64_t bytes) {
    if (plan->conversion_count != UINT64_MAX) ++plan->conversion_count;
    if (plan->conversion_bytes > UINT64_MAX - bytes) plan->conversion_bytes = UINT64_MAX;
    else plan->conversion_bytes += bytes;
}

static lw_status fast_ensure_nhwc(lw_x64_rec_fast_plan* plan, uint32_t tensor_index,
                                  uint32_t graph_input_index, const float* graph_input,
                                  lw_error* error) {
    const lw_runtime_tensor* tensor;
    const float* source;
    float* destination;
    if (plan == NULL || tensor_index >= plan->tensor_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid fast tensor index");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if ((plan->tensors[tensor_index].available_layouts & LW_X64_FAST_HAVE_NHWC) != 0u) {
        return LW_STATUS_OK;
    }
    tensor = &plan->session->tensors[tensor_index];
    if (tensor->rank != 4u || tensor->dtype != LW_DTYPE_F32 ||
        (plan->tensors[tensor_index].available_layouts & LW_X64_FAST_HAVE_NCHW) == 0u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "tensor cannot be converted to NHWC");
        return LW_STATUS_UNSUPPORTED;
    }
    source = lw_executor_tensor_input_data(plan->session, tensor_index, graph_input_index,
                                            graph_input);
    destination = fast_nhwc_pointer(plan, tensor_index);
    if (source == NULL || destination == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "NCHW to NHWC conversion failed");
        return LW_STATUS_UNSUPPORTED;
    }
    lw_x64_fast_nchw_to_nhwc(source, destination, (uint32_t)tensor->dimensions[0],
                             (uint32_t)tensor->dimensions[1], (uint32_t)tensor->dimensions[2],
                             (uint32_t)tensor->dimensions[3]);
    plan->tensors[tensor_index].available_layouts = (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC);
    fast_add_conversion(plan, tensor->byte_size);
    return LW_STATUS_OK;
}

static lw_status fast_ensure_nchw(lw_x64_rec_fast_plan* plan, uint32_t tensor_index,
                                  lw_error* error) {
    const lw_runtime_tensor* tensor;
    const float* source;
    float* destination;
    if (plan == NULL || tensor_index >= plan->tensor_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid fast tensor index");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if ((plan->tensors[tensor_index].available_layouts & LW_X64_FAST_HAVE_NCHW) != 0u) {
        return LW_STATUS_OK;
    }
    tensor = &plan->session->tensors[tensor_index];
    if (tensor->rank != 4u || tensor->dtype != LW_DTYPE_F32) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "tensor cannot be converted to NCHW");
        return LW_STATUS_UNSUPPORTED;
    }
    source = fast_nhwc_pointer(plan, tensor_index);
    destination = lw_executor_tensor_output_data(plan->session, tensor_index);
    if (source == NULL || destination == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "NHWC to NCHW conversion failed");
        return LW_STATUS_UNSUPPORTED;
    }
    lw_x64_fast_nhwc_to_nchw(source, destination, (uint32_t)tensor->dimensions[0],
                             (uint32_t)tensor->dimensions[1], (uint32_t)tensor->dimensions[2],
                             (uint32_t)tensor->dimensions[3]);
    plan->tensors[tensor_index].available_layouts = (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC);
    fast_add_conversion(plan, tensor->byte_size);
    return LW_STATUS_OK;
}

static lw_status fast_execute_generic_node(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                           uint32_t graph_input_index, const float* graph_input,
                                           uint32_t* consumed_nodes, lw_error* error) {
    const uint8_t* node;
    uint16_t input_count;
    uint32_t input_slot;
    uint32_t output_index;
    uint32_t consumed = 1u;
    lw_status status;
    if (plan == NULL || node_index >= plan->node_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid fast node index");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
    input_count = lwm_read_u16(node + 2u);
    if (input_count > LWM_V0_MAX_NODE_INPUTS) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "node input count exceeds limit");
        return LW_STATUS_INVALID_SHAPE;
    }
    for (input_slot = 0u; input_slot < input_count; ++input_slot) {
        uint32_t input_index = lwm_read_u32(node + 8u + (size_t)input_slot * sizeof(uint32_t));
        if (input_index >= plan->tensor_count) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "node input index out of bounds");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        if ((plan->session->tensors[input_index].flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u &&
            fast_layout_tracked_tensor(&plan->session->tensors[input_index])) {
            status = fast_ensure_nchw(plan, input_index, error);
            if (status != LW_STATUS_OK) return status;
        }
    }
    status = lw_executor_execute_best_node_f32(plan->session, node_index, graph_input_index,
                                                graph_input, NULL, &consumed);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "generic fast executor node failed");
        return status;
    }
    if (consumed == 0u || consumed > plan->node_count - node_index) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "legacy node consumption is invalid");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    output_index = lwm_read_u32(plan->session->model->bytes +
                                (size_t)plan->session->model->node_offset +
                                (size_t)(node_index + consumed - 1u) * LWM_V0_NODE_SIZE + 40u);
    if (output_index >= plan->tensor_count) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "node output index out of bounds");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    if (fast_layout_tracked_tensor(&plan->session->tensors[output_index])) {
        fast_publish_nchw(plan, output_index);
    }
    if (consumed_nodes != NULL) *consumed_nodes = consumed;
    return LW_STATUS_OK;
}
static lw_status fast_execute_pointwise(lw_x64_rec_fast_plan* plan,
                                        const lw_x64_fast_node* node,
                                        uint32_t graph_input_index,
                                        const float* graph_input,
                                        lw_error* error) {
    const lw_x64_fast_conv* conv = &node->data.conv;
    float* input;
    float* output;
    lw_nhwc_epilogue epilogue;
    lw_status status;
    status = fast_ensure_nhwc(plan, conv->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, conv->input_index);
    output = fast_nhwc_pointer(plan, conv->output_index);
    if (input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "pointwise fast buffers are unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    memset(&epilogue, 0, sizeof(epilogue));
    epilogue.bias = conv->bias;
    lw_avx2_fma_nhwc_pointwise_f32(input, conv->packed_weights, &epilogue, output,
                                   conv->input_height * conv->input_width,
                                   conv->input_channels, conv->output_channels);
    fast_publish_nhwc(plan, conv->output_index);
    return LW_STATUS_OK;
}

static lw_status fast_execute_dense(lw_x64_rec_fast_plan* plan,
                                    const lw_x64_fast_node* node,
                                    uint32_t graph_input_index,
                                    const float* graph_input,
                                    lw_error* error) {
    const lw_x64_fast_conv* conv = &node->data.conv;
    lw_nhwc_dense_desc desc;
    lw_nhwc_epilogue epilogue;
    float* input;
    float* output;
    lw_status status;
    status = fast_ensure_nhwc(plan, conv->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, conv->input_index);
    output = fast_nhwc_pointer(plan, conv->output_index);
    if (input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "dense fast buffers are unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    memset(&desc, 0, sizeof(desc));
    desc.batch = 1u;
    desc.input_channels = conv->input_channels;
    desc.input_height = conv->input_height;
    desc.input_width = conv->input_width;
    desc.output_channels = conv->output_channels;
    desc.output_height = conv->output_height;
    desc.output_width = conv->output_width;
    desc.kernel_h = conv->kernel_h;
    desc.kernel_w = conv->kernel_w;
    desc.stride_h = conv->stride_h;
    desc.stride_w = conv->stride_w;
    desc.pad_top = conv->pad_top;
    desc.pad_left = conv->pad_left;
    desc.dense_kc = conv->dense_kc;
    memset(&epilogue, 0, sizeof(epilogue));
    epilogue.bias = conv->bias;
    status = lw_avx2_fma_nhwc_dense_f32(input, conv->packed_weights, &epilogue, output,
                                        &desc, plan->scratch, plan->scratch_bytes);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "dense fast kernel failed");
        return status;
    }
    fast_publish_nhwc(plan, conv->output_index);
    return LW_STATUS_OK;
}

static lw_status fast_execute_node(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                   uint32_t graph_input_index, const float* graph_input,
                                   uint32_t* consumed_nodes, lw_error* error) {
    const lw_x64_fast_node* node;
    lw_status status;
    if (plan == NULL || node_index >= plan->node_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid fast node index");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    node = &plan->nodes[node_index];
    if (node->kind == LW_X64_FAST_NODE_POINTWISE) {
        status = fast_execute_pointwise(plan, node, graph_input_index, graph_input, error);
    } else if (node->kind == LW_X64_FAST_NODE_DENSE) {
        status = fast_execute_dense(plan, node, graph_input_index, graph_input, error);
    } else {
        return fast_execute_generic_node(plan, node_index, graph_input_index, graph_input,
                                         consumed_nodes, error);
    }
    if (status == LW_STATUS_OK && consumed_nodes != NULL) *consumed_nodes = 1u;
    return status;
}
lw_status lw_x64_rec_fast_run(lw_x64_rec_fast_plan* plan, const float* input,
                              uint64_t input_element_count, float* output,
                              uint64_t output_element_count, lw_error* error) {
    const lw_runtime_tensor* input_tensor;
    const lw_runtime_tensor* output_tensor;
    const float* graph_output;
    uint32_t node_index;
    lw_status status;
    if (plan == NULL || plan->session == NULL || plan->session->model == NULL || input == NULL ||
        output == NULL || plan->graph_input_index >= plan->tensor_count ||
        plan->graph_output_index >= plan->tensor_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid fast execution arguments");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    input_tensor = &plan->session->tensors[plan->graph_input_index];
    output_tensor = &plan->session->tensors[plan->graph_output_index];
    if (input_tensor->dtype != LW_DTYPE_F32 || output_tensor->dtype != LW_DTYPE_F32 ||
        input_element_count != fast_element_count(input_tensor) ||
        output_element_count != fast_element_count(output_tensor)) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "fast execution tensor shape mismatch");
        return LW_STATUS_INVALID_SHAPE;
    }
    fast_reset_layouts(plan);
    node_index = 0u;
    while (node_index < plan->node_count) {
        uint32_t consumed = 1u;
        status = fast_execute_node(plan, node_index, plan->graph_input_index, input,
                                   &consumed, error);
        if (status != LW_STATUS_OK) return status;
        node_index += consumed;
    }
    if (fast_layout_tracked_tensor(output_tensor)) {
        status = fast_ensure_nchw(plan, plan->graph_output_index, error);
        if (status != LW_STATUS_OK) return status;
    }
    graph_output = lw_executor_tensor_input_data(plan->session, plan->graph_output_index,
                                                 plan->graph_input_index, input);
    if (graph_output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "fast executor graph output is unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    memcpy(output, graph_output, (size_t)output_tensor->byte_size);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
