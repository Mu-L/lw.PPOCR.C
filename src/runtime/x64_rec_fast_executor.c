#include "x64_rec_fast_internal.h"

#include "executor_internal.h"
#include "operator_internal.h"
#include "lwm_read.h"
#include "../simd/simd_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const float* fast_elementwise_constant_f32(const lw_session* session, uint32_t tensor_index) {
    const lw_runtime_tensor* tensor;
    const uint8_t* disk;
    uint64_t data_offset;
    if (session == NULL || session->model == NULL || tensor_index >= session->model->info.tensor_count) return NULL;
    tensor = &session->tensors[tensor_index];
    if (tensor->dtype != LW_DTYPE_F32 || (tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u) return NULL;
    disk = session->model->bytes + (size_t)session->model->tensor_offset +
           (size_t)tensor_index * LWM_V0_TENSOR_SIZE;
    data_offset = lwm_read_u64(disk + 48u);
    if (data_offset > session->model->byte_count || tensor->byte_size > session->model->byte_count - data_offset) return NULL;
    return (const float*)(const void*)(session->model->bytes + (size_t)data_offset);
}
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

static void fast_begin_run(lw_x64_rec_fast_plan* plan,
                           lw_x64_fast_input_layout input_layout) {
    uint32_t index;
    if (plan == NULL) return;
    plan->conversion_count = 0u;
    plan->conversion_bytes = 0u;
    plan->depthwise_x2_invocations = 0u;
    plan->depthwise_x1_invocations = 0u;
    for (index = 0u; index < plan->tensor_count; ++index) {
        plan->tensors[index].available_layouts = 0u;
    }
    if (input_layout == LW_X64_FAST_INPUT_NHWC) {
        fast_publish_nhwc(plan, plan->graph_input_index);
    } else {
        fast_publish_nchw(plan, plan->graph_input_index);
    }
}

float* lw_x64_rec_fast_graph_input_nhwc(lw_x64_rec_fast_plan* plan,
                                        uint64_t* element_count) {
    const lw_runtime_tensor* tensor;
    if (element_count != NULL) *element_count = 0u;
    if (plan == NULL || plan->graph_input_index >= plan->tensor_count ||
        plan->layout.graph_input_direct_nhwc == 0u) return NULL;
    tensor = &plan->session->tensors[plan->graph_input_index];
    if (tensor->dtype != LW_DTYPE_F32 || tensor->rank != 4u) return NULL;
    if (element_count != NULL) *element_count = tensor->byte_size / sizeof(float);
    return fast_nhwc_pointer(plan, plan->graph_input_index);
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
                                        const lw_x64_physical_op* node,
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
    epilogue.activation = conv->activation;
    if (conv->residual_index != UINT32_MAX) {
        status = fast_ensure_nhwc(plan, conv->residual_index, graph_input_index, graph_input, error);
        if (status != LW_STATUS_OK) return status;
        epilogue.residual = fast_nhwc_pointer(plan, conv->residual_index);
        if (epilogue.residual == NULL) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED, "pointwise residual buffer is unavailable");
            return LW_STATUS_UNSUPPORTED;
        }
    }
    lw_avx2_fma_nhwc_pointwise_f32(input, conv->packed_weights, &epilogue, output,
                                   conv->input_height * conv->input_width,
                                   conv->input_channels, conv->output_channels);
    fast_publish_nhwc(plan, conv->output_index);
    return LW_STATUS_OK;
}

static lw_status fast_execute_dense(lw_x64_rec_fast_plan* plan,
                                    const lw_x64_physical_op* node,
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
    desc.pad_bottom = conv->pad_bottom;
    desc.pad_right = conv->pad_right;
    desc.dense_kc = conv->dense_kc;
    memset(&epilogue, 0, sizeof(epilogue));
    epilogue.bias = conv->bias;
    epilogue.activation = conv->activation;
    if (conv->residual_index != UINT32_MAX) {
        status = fast_ensure_nhwc(plan, conv->residual_index, graph_input_index, graph_input, error);
        if (status != LW_STATUS_OK) return status;
        epilogue.residual = fast_nhwc_pointer(plan, conv->residual_index);
        if (epilogue.residual == NULL) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED, "dense residual buffer is unavailable");
            return LW_STATUS_UNSUPPORTED;
        }
    }
    status = lw_avx2_fma_nhwc_dense_prepared_f32(input, conv->packed_weights,
        conv->dense_input_offsets_k, conv->dense_patch_offsets_k, &epilogue, output,
                                        &desc, plan->scratch, plan->scratch_bytes);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "dense fast kernel failed");
        return status;
    }
    fast_publish_nhwc(plan, conv->output_index);
    return LW_STATUS_OK;
}

static lw_status fast_elementwise_scalar_value(lw_x64_rec_fast_plan* plan, uint32_t tensor_index,
                                               uint32_t graph_input_index, const float* graph_input,
                                               float* value, lw_error* error) {
    const float* data;
    lw_status status;
    if (value == NULL || tensor_index >= plan->tensor_count) return LW_STATUS_INVALID_ARGUMENT;
    data = fast_elementwise_constant_f32(plan->session, tensor_index);
    if (data == NULL) {
        status = fast_ensure_nhwc(plan, tensor_index, graph_input_index, graph_input, error);
        if (status != LW_STATUS_OK) return status;
        data = fast_nhwc_pointer(plan, tensor_index);
    }
    if (data == NULL) return LW_STATUS_UNSUPPORTED;
    *value = data[0];
    return LW_STATUS_OK;
}

static lw_status fast_execute_elementwise(lw_x64_rec_fast_plan* plan,
                                          const lw_x64_physical_op* node,
                                          uint32_t graph_input_index,
                                          const float* graph_input,
                                          lw_error* error) {
    const lw_x64_fast_elementwise* op = &node->data.elementwise;
    const lw_runtime_tensor* output_tensor = &plan->session->tensors[op->output_index];
    float* input = NULL;
    float* output;
    uint64_t elements = output_tensor->byte_size / sizeof(float);
    lw_status status;
    if (node->kind == LW_X64_FAST_NODE_CONTIGUOUS_UNARY ||
        op->broadcast_kind == LW_X64_FAST_BROADCAST_NONE ||
        op->broadcast_kind == LW_X64_FAST_BROADCAST_RIGHT_SCALAR ||
        op->broadcast_kind == LW_X64_FAST_BROADCAST_RIGHT_CHANNEL) {
        status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
        if (status != LW_STATUS_OK) return status;
        input = fast_nhwc_pointer(plan, op->input_index);
    }
    output = fast_nhwc_pointer(plan, op->output_index);
    if ((input == NULL && node->kind == LW_X64_FAST_NODE_CONTIGUOUS_UNARY) || output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "elementwise fast buffers are unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    if (node->kind == LW_X64_FAST_NODE_CONTIGUOUS_UNARY) {
        switch (op->operation) {
        case LW_OP_RELU: lw_avx2_relu_contiguous_f32(input, output, elements); break;
        case LW_OP_ERF: lw_avx2_erf_f32(input, output, elements); break;
        case LW_OP_SQRT: status = lw_scalar_sqrt_f32(input, output, elements); if (status != LW_STATUS_OK) return status; break;
        case LW_OP_SIGMOID: status = lw_scalar_sigmoid_f32(input, output, elements); if (status != LW_STATUS_OK) return status; break;
        case LW_OP_HARD_SIGMOID: lw_avx2_hard_sigmoid_contiguous_f32(input, output, elements, op->alpha, op->beta); break;
        default: return LW_STATUS_UNSUPPORTED;
        }
    } else {
        const float* right = NULL;
        const float* full = input;
        const float* channel = NULL;
        lw_scalar_binary_op operation = op->operation == LW_OP_ADD ? LW_SCALAR_BINARY_ADD :
            (op->operation == LW_OP_MUL ? LW_SCALAR_BINARY_MUL :
             (op->operation == LW_OP_DIV ? LW_SCALAR_BINARY_DIV :
              (op->operation == LW_OP_SUB ? LW_SCALAR_BINARY_SUB : LW_SCALAR_BINARY_POW)));
        if (op->broadcast_kind == LW_X64_FAST_BROADCAST_NONE) {
            status = fast_ensure_nhwc(plan, op->rhs_index, graph_input_index, graph_input, error);
            if (status != LW_STATUS_OK) return status;
            right = fast_nhwc_pointer(plan, op->rhs_index);
            if (input == NULL || right == NULL) return LW_STATUS_UNSUPPORTED;
            lw_avx2_binary_contiguous_f32(operation, input, right, output, elements);
        } else if (op->broadcast_kind == LW_X64_FAST_BROADCAST_RIGHT_SCALAR) {
            float scalar;
            status = fast_elementwise_scalar_value(plan, op->rhs_index, graph_input_index, graph_input, &scalar, error);
            if (status != LW_STATUS_OK || input == NULL) return status != LW_STATUS_OK ? status : LW_STATUS_UNSUPPORTED;
            lw_avx2_binary_right_scalar_f32(operation, input, scalar, output, elements);
        } else if (op->broadcast_kind == LW_X64_FAST_BROADCAST_LEFT_SCALAR) {
            float scalar;
            uint64_t index;
            status = fast_elementwise_scalar_value(plan, op->input_index, graph_input_index, graph_input, &scalar, error);
            if (status != LW_STATUS_OK) return status;
            status = fast_ensure_nhwc(plan, op->rhs_index, graph_input_index, graph_input, error);
            if (status != LW_STATUS_OK) return status;
            right = fast_nhwc_pointer(plan, op->rhs_index);
            if (right == NULL) return LW_STATUS_UNSUPPORTED;
            for (index = 0u; index < elements; ++index) {
                if (operation == LW_SCALAR_BINARY_ADD) output[index] = scalar + right[index];
                else if (operation == LW_SCALAR_BINARY_MUL) output[index] = scalar * right[index];
                else if (operation == LW_SCALAR_BINARY_SUB) output[index] = scalar - right[index];
                else if (operation == LW_SCALAR_BINARY_DIV) output[index] = scalar / right[index];
                else output[index] = powf(scalar, right[index]);
            }
        } else if (op->broadcast_kind == LW_X64_FAST_BROADCAST_RIGHT_CHANNEL) {
            channel = fast_elementwise_constant_f32(plan->session, op->rhs_index);
            if (channel == NULL) {
                status = fast_ensure_nhwc(plan, op->rhs_index, graph_input_index, graph_input, error);
                if (status != LW_STATUS_OK) return status;
                channel = fast_nhwc_pointer(plan, op->rhs_index);
            }
            if (full == NULL || channel == NULL) return LW_STATUS_UNSUPPORTED;
            lw_avx2_binary_channel_f32(operation, full, channel, output,
                                        elements / op->channels, op->channels, 0);
        } else {
            channel = fast_elementwise_constant_f32(plan->session, op->input_index);
            if (channel == NULL) {
                status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
                if (status != LW_STATUS_OK) return status;
                channel = fast_nhwc_pointer(plan, op->input_index);
            }
            status = fast_ensure_nhwc(plan, op->rhs_index, graph_input_index, graph_input, error);
            if (status != LW_STATUS_OK) return status;
            full = fast_nhwc_pointer(plan, op->rhs_index);
            if (full == NULL || channel == NULL) return LW_STATUS_UNSUPPORTED;
            lw_avx2_binary_channel_f32(operation, full, channel, output,
                                        elements / op->channels, op->channels, 1);
        }
    }
    fast_publish_nhwc(plan, op->output_index);
    return LW_STATUS_OK;
}
static lw_status fast_execute_reduce_mean(lw_x64_rec_fast_plan* plan,
                                          const lw_x64_physical_op* node,
                                          uint32_t graph_input_index,
                                          const float* graph_input,
                                          lw_error* error) {
    const lw_x64_fast_spatial* op = &node->data.spatial;
    const lw_runtime_tensor* input_tensor = &plan->session->tensors[op->input_index];
    const lw_runtime_tensor* output_tensor = &plan->session->tensors[op->output_index];
    float* input;
    float* output;
    lw_status status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, op->input_index);
    output = fast_nhwc_pointer(plan, op->output_index);
    if (input == NULL || output == NULL) return LW_STATUS_UNSUPPORTED;
    lw_avx2_nhwc_reduce_mean_hw_f32(input, output,
                                    (uint32_t)input_tensor->dimensions[0],
                                    (uint32_t)input_tensor->dimensions[2],
                                    (uint32_t)input_tensor->dimensions[3],
                                    (uint32_t)input_tensor->dimensions[1]);
    fast_publish_nhwc(plan, op->output_index);
    (void)output_tensor;
    return LW_STATUS_OK;
}

static lw_status fast_execute_pool(lw_x64_rec_fast_plan* plan,
                                   const lw_x64_physical_op* node,
                                   uint32_t graph_input_index,
                                   const float* graph_input,
                                   lw_error* error) {
    const lw_x64_fast_spatial* op = &node->data.spatial;
    const lw_runtime_tensor* input_tensor = &plan->session->tensors[op->input_index];
    const lw_runtime_tensor* output_tensor = &plan->session->tensors[op->output_index];
    float* input;
    float* output;
    lw_status status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, op->input_index);
    output = fast_nhwc_pointer(plan, op->output_index);
    if (input == NULL || output == NULL) return LW_STATUS_UNSUPPORTED;
    lw_avx2_nhwc_pool_f32(input, output,
                          (uint32_t)input_tensor->dimensions[0],
                          (uint32_t)input_tensor->dimensions[2],
                          (uint32_t)input_tensor->dimensions[3],
                          (uint32_t)output_tensor->dimensions[2],
                          (uint32_t)output_tensor->dimensions[3],
                          (uint32_t)input_tensor->dimensions[1],
                          op->kernel_h, op->kernel_w, op->stride_h, op->stride_w,
                          op->pad_top, op->pad_left,
                          op->count_include_pad, op->is_max);
    fast_publish_nhwc(plan, op->output_index);
    return LW_STATUS_OK;
}

static lw_status fast_execute_batch_norm(lw_x64_rec_fast_plan* plan,
                                         const lw_x64_physical_op* node,
                                         uint32_t graph_input_index,
                                         const float* graph_input,
                                         lw_error* error) {
    const lw_x64_fast_spatial* op = &node->data.spatial;
    const lw_runtime_tensor* input_tensor = &plan->session->tensors[op->input_index];
    float* input;
    float* output;
    lw_status status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, op->input_index);
    output = fast_nhwc_pointer(plan, op->output_index);
    if (input == NULL || output == NULL) return LW_STATUS_UNSUPPORTED;
    if (op->affine_mul == NULL || op->affine_add == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "batch normalization affine is unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    lw_avx2_nhwc_affine_f32(input, op->affine_mul, op->affine_add, output,
                            (uint32_t)((uint64_t)input_tensor->dimensions[0] *
                                       input_tensor->dimensions[2] * input_tensor->dimensions[3]),
                            (uint32_t)input_tensor->dimensions[1]);
    fast_publish_nhwc(plan, op->output_index);
    return LW_STATUS_OK;
}
static lw_status fast_execute_concat(lw_x64_rec_fast_plan* plan,
                                     const lw_x64_physical_op* node,
                                     uint32_t graph_input_index,
                                     const float* graph_input,
                                     lw_error* error) {
    const lw_x64_fast_spatial* op = &node->data.spatial;
    const lw_runtime_tensor* output_tensor = &plan->session->tensors[op->output_index];
    float* output = fast_nhwc_pointer(plan, op->output_index);
    uint64_t pixels = (uint64_t)output_tensor->dimensions[0] * output_tensor->dimensions[2] * output_tensor->dimensions[3];
    uint32_t slot;
    uint32_t output_channels = 0u;
    if (output == NULL) return LW_STATUS_UNSUPPORTED;
    for (slot = 0u; slot < op->input_count; ++slot) {
        uint32_t input_index = op->input_indices[slot];
        const lw_runtime_tensor* input_tensor = &plan->session->tensors[input_index];
        float* input;
        uint64_t pixel;
        lw_status status = fast_ensure_nhwc(plan, input_index, graph_input_index, graph_input, error);
        if (status != LW_STATUS_OK) return status;
        input = fast_nhwc_pointer(plan, input_index);
        if (input == NULL) return LW_STATUS_UNSUPPORTED;
        for (pixel = 0u; pixel < pixels; ++pixel) {
            memcpy(output + pixel * output_tensor->dimensions[1] + output_channels,
                   input + pixel * input_tensor->dimensions[1],
                   (size_t)op->input_channels[slot] * sizeof(float));
        }
        output_channels += op->input_channels[slot];
    }
    if (output_channels != (uint32_t)output_tensor->dimensions[1]) return LW_STATUS_INVALID_SHAPE;
    fast_publish_nhwc(plan, op->output_index);
    return LW_STATUS_OK;
}
static lw_status fast_execute_depthwise(lw_x64_rec_fast_plan* plan,
                                        const lw_x64_physical_op* node,
                                        uint32_t graph_input_index,
                                        const float* graph_input,
                                        lw_nhwc_depthwise_stats* depthwise_stats,
                                        lw_error* error) {
    const lw_x64_fast_conv* conv = &node->data.conv;
    lw_nhwc_depthwise_desc desc;
    float* input;
    float* output;
    lw_status status;
    status = fast_ensure_nhwc(plan, conv->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, conv->input_index);
    output = fast_nhwc_pointer(plan, conv->output_index);
    if (input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "depthwise fast buffers are unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    memset(&desc, 0, sizeof(desc));
    desc.batch = 1u;
    desc.channels = conv->input_channels;
    desc.input_height = conv->input_height;
    desc.input_width = conv->input_width;
    desc.output_height = conv->output_height;
    desc.output_width = conv->output_width;
    desc.kernel_h = conv->kernel_h;
    desc.kernel_w = conv->kernel_w;
    desc.stride_h = conv->stride_h;
    desc.stride_w = conv->stride_w;
    desc.pad_top = conv->pad_top;
    desc.pad_left = conv->pad_left;
    desc.pad_bottom = conv->pad_bottom;
    desc.pad_right = conv->pad_right;
    status = depthwise_stats == NULL
        ? lw_avx2_fma_nhwc_depthwise_f32(input, conv->packed_weights, conv->bias,
                                         output, &desc)
        : lw_avx2_fma_nhwc_depthwise_profiled_f32(input, conv->packed_weights, conv->bias,
                                                  output, &desc, depthwise_stats);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "depthwise fast kernel failed");
        return status;
    }
    fast_publish_nhwc(plan, conv->output_index);
    return LW_STATUS_OK;
}
static lw_status fast_execute_gelu(lw_x64_rec_fast_plan* plan,
                                   const lw_x64_physical_op* node,
                                   uint32_t graph_input_index,
                                   const float* graph_input,
                                   lw_error* error) {
    const lw_x64_fast_elementwise* op = &node->data.elementwise;
    const lw_runtime_tensor* tensor = &plan->session->tensors[op->output_index];
    float* input;
    float* output;
    lw_status status;
    if (plan->layout.node_layout[node->semantic_begin] == LW_FAST_LAYOUT_NCHW) {
        input = (float*)lw_executor_tensor_input_data(plan->session, op->input_index,
                                                       graph_input_index, graph_input);
        output = lw_executor_tensor_output_data(plan->session, op->output_index);
        if (input == NULL || output == NULL) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED, "NCHW GELU buffers are unavailable");
            return LW_STATUS_UNSUPPORTED;
        }
        lw_avx2_gelu_f32(input, output, tensor->byte_size / sizeof(float));
        fast_publish_nchw(plan, op->output_index);
        return LW_STATUS_OK;
    }
    status = fast_ensure_nhwc(plan, op->input_index, graph_input_index, graph_input, error);
    if (status != LW_STATUS_OK) return status;
    input = fast_nhwc_pointer(plan, op->input_index);
    output = fast_nhwc_pointer(plan, op->output_index);
    if (input == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "NHWC GELU buffers are unavailable");
        return LW_STATUS_UNSUPPORTED;
    }
    lw_avx2_gelu_f32(input, output, tensor->byte_size / sizeof(float));
    fast_publish_nhwc(plan, op->output_index);
    return LW_STATUS_OK;
}
static lw_status fast_execute_generic_physical_op(lw_x64_rec_fast_plan* plan,
                                                 const lw_x64_physical_op* op,
                                                 uint32_t graph_input_index,
                                                 const float* graph_input,
                                                 lw_error* error) {
    uint32_t consumed = 1u;
    lw_status status;
    if (plan == NULL || op == NULL) return LW_STATUS_INVALID_ARGUMENT;
    status = fast_execute_generic_node(plan, op->semantic_begin, graph_input_index,
                                       graph_input, &consumed, error);
    if (status != LW_STATUS_OK) return status;
    if (consumed != op->semantic_count) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "generic physical op span mismatch");
        return LW_STATUS_UNSUPPORTED;
    }
    return LW_STATUS_OK;
}

static lw_status fast_execute_physical_op(lw_x64_rec_fast_plan* plan,
                                          const lw_x64_physical_op* op,
                                          uint32_t graph_input_index,
                                          const float* graph_input,
                                          lw_nhwc_depthwise_stats* depthwise_stats,
                                          lw_error* error) {
    if (plan == NULL || op == NULL) return LW_STATUS_INVALID_ARGUMENT;
    switch (op->kind) {
    case LW_X64_FAST_NODE_GELU: return fast_execute_gelu(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_BATCH_NORM: return fast_execute_batch_norm(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_CONCAT: return fast_execute_concat(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_REDUCE_MEAN: return fast_execute_reduce_mean(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_POOL: return fast_execute_pool(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_POINTWISE: return fast_execute_pointwise(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_CONTIGUOUS_UNARY:
    case LW_X64_FAST_NODE_CONTIGUOUS_BINARY: return fast_execute_elementwise(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_DENSE: return fast_execute_dense(plan, op, graph_input_index, graph_input, error);
    case LW_X64_FAST_NODE_DEPTHWISE: return fast_execute_depthwise(plan, op, graph_input_index, graph_input, depthwise_stats, error);
    case LW_X64_FAST_NODE_GENERIC: return fast_execute_generic_physical_op(plan, op, graph_input_index, graph_input, error);
    default: return LW_STATUS_UNSUPPORTED;
    }
}

static void fast_profile_add(uint64_t* value, uint64_t amount) {
    if (value == NULL) return;
    if (*value > UINT64_MAX - amount) *value = UINT64_MAX;
    else *value += amount;
}
static lw_status fast_execute_backbone(lw_x64_rec_fast_plan* plan,
                                       const float* input,
                                       lw_x64_fast_profile* profile,
                                       lw_nhwc_depthwise_stats* depthwise_stats,
                                       lw_error* error) {
    if (plan == NULL) return LW_STATUS_INVALID_ARGUMENT;
    for (uint32_t op_index = 0u; op_index < plan->op_count; ++op_index) {
        const lw_x64_physical_op* physical = &plan->ops[op_index];
        uint32_t node_index = physical->semantic_begin;
        uint64_t started = 0u;
        uint64_t finished = 0u;
        lw_status status;
        if (profile != NULL && profile->clock != NULL) {
            started = profile->clock(profile->clock_context);
        }
        status = fast_execute_physical_op(plan, physical, plan->graph_input_index, input,
                                           profile == NULL ? NULL : depthwise_stats, error);
        if (profile != NULL && profile->clock != NULL) {
            finished = profile->clock(profile->clock_context);
            if (finished < started) finished = started;
            fast_profile_add(&profile->total_nanoseconds, finished - started);
            if (node_index < LW_X64_FAST_PROFILE_NODE_CAPACITY) {
                fast_profile_add(&profile->node_nanoseconds[node_index], finished - started);
                fast_profile_add(&profile->node_invocations[node_index], 1u);
            }
            if (physical->kind < LW_X64_FAST_PROFILE_KIND_CAPACITY) {
                fast_profile_add(&profile->kind_nanoseconds[physical->kind], finished - started);
                fast_profile_add(&profile->kind_invocations[physical->kind], 1u);
            }
        }
        if (status != LW_STATUS_OK) return status;
    }
    if (profile != NULL && depthwise_stats != NULL) {
        plan->depthwise_x2_invocations = depthwise_stats->x2_invocations;
        plan->depthwise_x1_invocations = depthwise_stats->x1_invocations;
    }
    return LW_STATUS_OK;
}
lw_status lw_x64_rec_fast_run_profiled(lw_x64_rec_fast_plan* plan, const float* input,
                                       uint64_t input_element_count, float* output,
                                       uint64_t output_element_count,
                                       lw_x64_fast_profile* profile, lw_error* error) {
    const lw_runtime_tensor* input_tensor;
    const lw_runtime_tensor* output_tensor;
    const float* graph_output;
    lw_x64_fast_profile_clock profile_clock_fn = profile == NULL ? NULL : profile->clock;
    void* profile_clock_context = profile == NULL ? NULL : profile->clock_context;
    lw_nhwc_depthwise_stats depthwise_stats = { 0u, 0u };
    lw_status status;
    if (profile != NULL) {
        memset(profile, 0, sizeof(*profile));
        profile->clock = profile_clock_fn;
        profile->clock_context = profile_clock_context;
    }
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
    fast_begin_run(plan, LW_X64_FAST_INPUT_NCHW);
    status = fast_execute_backbone(plan, input, profile, &depthwise_stats, error);
    if (status != LW_STATUS_OK) return status;
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
    if (profile != NULL) {
        profile->conversion_invocations = plan->conversion_count;
        profile->conversion_bytes = plan->conversion_bytes;
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

lw_status lw_x64_rec_fast_run_prepared_nhwc(lw_x64_rec_fast_plan* plan,
                                            lw_x64_fast_profile* profile,
                                            lw_error* error) {
    lw_x64_fast_profile_clock profile_clock_fn = profile == NULL ? NULL : profile->clock;
    void* profile_clock_context = profile == NULL ? NULL : profile->clock_context;
    lw_nhwc_depthwise_stats depthwise_stats = { 0u, 0u };
    lw_status status;
    if (profile != NULL) {
        memset(profile, 0, sizeof(*profile));
        profile->clock = profile_clock_fn;
        profile->clock_context = profile_clock_context;
    }
    if (plan == NULL || plan->session == NULL || plan->session->model == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "prepared NHWC fast run requires a valid plan");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (plan->layout.graph_input_direct_nhwc == 0u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "prepared NHWC fast run requires a direct NHWC plan");
        return LW_STATUS_UNSUPPORTED;
    }
    fast_begin_run(plan, LW_X64_FAST_INPUT_NHWC);
    status = fast_execute_backbone(plan, NULL, profile,
                                   profile == NULL ? NULL : &depthwise_stats, error);
    if (status != LW_STATUS_OK) return status;
    if (profile != NULL) {
        profile->conversion_invocations = plan->conversion_count;
        profile->conversion_bytes = plan->conversion_bytes;
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
lw_status lw_x64_rec_fast_run(lw_x64_rec_fast_plan* plan, const float* input,
                              uint64_t input_element_count, float* output,
                              uint64_t output_element_count, lw_error* error) {
    return lw_x64_rec_fast_run_profiled(plan, input, input_element_count, output,
                                        output_element_count, NULL, error);
}
