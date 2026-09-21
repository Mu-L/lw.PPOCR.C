#include "x64_rec_fast_internal.h"

#include "lwm_read.h"
#include "operator_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static size_t fast_align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}



static int fast_tensor_eligible(const lw_runtime_tensor* tensor) {
    return tensor != NULL && (tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u &&
           tensor->dtype == LW_DTYPE_F32 && tensor->rank == 4u &&
           tensor->dimensions[0] > 0 && tensor->dimensions[1] > 0 &&
           tensor->dimensions[2] > 0 && tensor->dimensions[3] > 0;
}

static const float* fast_constant_f32(const lw_session* session, uint32_t tensor_index) {
    const lw_runtime_tensor* tensor;
    const uint8_t* disk;
    uint64_t data_offset;
    if (session == NULL || session->model == NULL || tensor_index >= session->model->info.tensor_count) {
        return NULL;
    }
    tensor = &session->tensors[tensor_index];
    if (tensor->dtype != LW_DTYPE_F32 ||
        (tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u) return NULL;
    disk = session->model->bytes + (size_t)session->model->tensor_offset +
           (size_t)tensor_index * LWM_V0_TENSOR_SIZE;
    data_offset = lwm_read_u64(disk + 48u);
    if (data_offset > session->model->byte_count ||
        tensor->byte_size > session->model->byte_count - data_offset) return NULL;
    return (const float*)(const void*)(session->model->bytes + (size_t)data_offset);
}

static int fast_same_shape4(const lw_runtime_tensor* left, const lw_runtime_tensor* right) {
    if (!fast_tensor_eligible(left) || !fast_tensor_eligible(right) || left->rank != right->rank) return 0;
    for (uint32_t index = 0u; index < 4u; ++index) {
        if (left->dimensions[index] != right->dimensions[index]) return 0;
    }
    return 1;
}

static uint64_t fast_tensor_elements(const lw_runtime_tensor* tensor) {
    return tensor == NULL || tensor->dtype != LW_DTYPE_F32 || tensor->byte_size % sizeof(float) != 0u
        ? 0u : tensor->byte_size / sizeof(float);
}

static float fast_read_f32(const uint8_t* bytes) {
    float value = 0.0f;
    if (bytes != NULL) memcpy(&value, bytes, sizeof(value));
    return value;
}

static int fast_channel_broadcast(const lw_session* session, uint32_t tensor_index,
                                  const lw_runtime_tensor* full) {
    const lw_runtime_tensor* tensor;
    uint64_t elements;
    if (session == NULL || full == NULL || tensor_index >= session->model->info.tensor_count) return 0;
    tensor = &session->tensors[tensor_index];
    elements = fast_tensor_elements(tensor);
    if (elements == 1u) return 1;
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u &&
        elements == (uint64_t)(uint32_t)full->dimensions[1]) return 1;
    return tensor->rank == 4u && tensor->dtype == LW_DTYPE_F32 &&
           tensor->dimensions[1] == full->dimensions[1] && tensor->dimensions[2] == 1 &&
           tensor->dimensions[3] == 1 &&
           (tensor->dimensions[0] == 1);
}

static int fast_classify_binary(const lw_session* session, uint32_t left_index,
                                uint32_t right_index, uint32_t output_index,
                                uint8_t* broadcast_kind) {
    const lw_runtime_tensor* left;
    const lw_runtime_tensor* right;
    const lw_runtime_tensor* output;
    if (session == NULL || broadcast_kind == NULL || left_index >= session->model->info.tensor_count ||
        right_index >= session->model->info.tensor_count || output_index >= session->model->info.tensor_count) return 0;
    left = &session->tensors[left_index]; right = &session->tensors[right_index];
    output = &session->tensors[output_index];
    if (!fast_tensor_eligible(output)) return 0;
    if (fast_same_shape4(left, output) && fast_same_shape4(right, output)) {
        *broadcast_kind = LW_X64_FAST_BROADCAST_NONE; return 1;
    }
    if (fast_same_shape4(left, output) && fast_tensor_elements(right) == 1u) {
        *broadcast_kind = LW_X64_FAST_BROADCAST_RIGHT_SCALAR; return 1;
    }
    if (fast_same_shape4(right, output) && fast_tensor_elements(left) == 1u) {
        *broadcast_kind = LW_X64_FAST_BROADCAST_LEFT_SCALAR; return 1;
    }
    if (fast_same_shape4(left, output) && fast_channel_broadcast(session, right_index, output)) {
        *broadcast_kind = LW_X64_FAST_BROADCAST_RIGHT_CHANNEL; return 1;
    }
    if (fast_same_shape4(right, output) && fast_channel_broadcast(session, left_index, output)) {
        *broadcast_kind = LW_X64_FAST_BROADCAST_LEFT_CHANNEL; return 1;
    }
    return 0;
}

static int fast_prepare_gelu(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                             lw_x64_fast_node* fast_node) {
    lw_fused_gelu_match match;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        !lw_match_fused_gelu(plan->session, node_index, &match)) return 0;
    if (match.inputs[0][0] >= plan->tensor_count || match.outputs[4] >= plan->tensor_count) return 0;
    input = &plan->session->tensors[match.inputs[0][0]];
    output = &plan->session->tensors[match.outputs[4]];
    if (!fast_same_shape4(input, output)) return 0;
    memset(&fast_node->data.elementwise, 0, sizeof(fast_node->data.elementwise));
    fast_node->data.elementwise.input_index = match.inputs[0][0];
    fast_node->data.elementwise.output_index = match.outputs[4];
    fast_node->data.elementwise.operation = LW_OP_ERF;
    fast_node->kind = LW_X64_FAST_NODE_GELU;
    fast_node->semantic_node_count = 5u;
    fast_node->output_index = match.outputs[4];
    ++plan->unary_node_count;
    return 1;
}
static int fast_prepare_elementwise(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                    lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    const uint8_t* params;
    uint16_t operation;
    uint16_t input_count;
    uint32_t input_index;
    uint32_t rhs_index = UINT32_MAX;
    uint32_t output_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    int is_binary;
    int is_unary;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
    operation = lwm_read_u16(node); input_count = lwm_read_u16(node + 2u);
    input_index = lwm_read_u32(node + 8u); output_index = lwm_read_u32(node + 40u);
    is_binary = operation == LW_OP_ADD || operation == LW_OP_MUL || operation == LW_OP_DIV ||
                operation == LW_OP_SUB || operation == LW_OP_POW;
    is_unary = operation == LW_OP_RELU || operation == LW_OP_SIGMOID || operation == LW_OP_SQRT ||
               operation == LW_OP_ERF || operation == LW_OP_HARD_SIGMOID;
    if ((!is_binary && !is_unary) || input_index >= plan->tensor_count || output_index >= plan->tensor_count) return 0;
    if ((is_unary && input_count != 1u) || (is_binary && input_count != 2u)) return 0;
    input = &plan->session->tensors[input_index]; output = &plan->session->tensors[output_index];
    if (!fast_same_shape4(input, output)) return 0;
    memset(&fast_node->data.elementwise, 0, sizeof(fast_node->data.elementwise));
    fast_node->data.elementwise.input_index = input_index;
    fast_node->data.elementwise.output_index = output_index;
    fast_node->data.elementwise.operation = operation;
    fast_node->data.elementwise.channels = (uint32_t)output->dimensions[1];
    if (is_binary) {
        uint8_t broadcast_kind;
        rhs_index = lwm_read_u32(node + 12u);
        if (!fast_classify_binary(plan->session, input_index, rhs_index, output_index, &broadcast_kind)) return 0;
        fast_node->data.elementwise.rhs_index = rhs_index;
        fast_node->data.elementwise.broadcast_kind = broadcast_kind;
        fast_node->kind = LW_X64_FAST_NODE_CONTIGUOUS_BINARY;
        ++plan->binary_node_count;
    } else {
        fast_node->kind = LW_X64_FAST_NODE_CONTIGUOUS_UNARY;
        if (operation == LW_OP_HARD_SIGMOID) {
            params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
            fast_node->data.elementwise.alpha = fast_read_f32(params + 4u);
            fast_node->data.elementwise.beta = fast_read_f32(params + 8u);
        }
        ++plan->unary_node_count;
        if (operation == LW_OP_RELU) ++plan->relu_node_count;
    }
    return 1;
}

static void fast_record_nhwc_blocker(lw_x64_rec_fast_plan* plan, const uint8_t* node) {
    uint32_t operation;
    if (plan == NULL || node == NULL) return;
    operation = (uint32_t)lwm_read_u16(node);
    if (plan->unsupported_nhwc_node_count != UINT32_MAX) ++plan->unsupported_nhwc_node_count;
    if (operation < LW_X64_FAST_OPERATOR_CAPACITY &&
        plan->unsupported_nhwc_by_operator[operation] != UINT32_MAX) {
        ++plan->unsupported_nhwc_by_operator[operation];
    }
}

static int fast_prepare_reduce_mean(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                    lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    int32_t axis0;
    int32_t axis1;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    if (lwm_read_u16(node) != LW_OP_REDUCE_MEAN || lwm_read_u16(node + 2u) != 1u ||
        lwm_read_u16(node + 4u) != 1u || lwm_read_u16(params + 2u) != 2u || lwm_read_u32(params + 4u) == 0u) return 0;
    input = &plan->session->tensors[lwm_read_u32(node + 8u)]; output = &plan->session->tensors[lwm_read_u32(node + 40u)];
    axis0 = lwm_read_i32(params + 12u); axis1 = lwm_read_i32(params + 16u);
    if (axis0 < 0) axis0 += 4; if (axis1 < 0) axis1 += 4;
    if (!fast_tensor_eligible(input) || !fast_tensor_eligible(output) ||
        !((axis0 == 2 && axis1 == 3) || (axis0 == 3 && axis1 == 2)) ||
        output->dimensions[0] != input->dimensions[0] || output->dimensions[1] != input->dimensions[1] ||
        output->dimensions[2] != 1 || output->dimensions[3] != 1) return 0;
    memset(&fast_node->data.spatial, 0, sizeof(fast_node->data.spatial));
    fast_node->data.spatial.input_index = lwm_read_u32(node + 8u);
    fast_node->data.spatial.output_index = lwm_read_u32(node + 40u);
    fast_node->kind = LW_X64_FAST_NODE_REDUCE_MEAN;
    ++plan->reduce_mean_node_count;
    return 1;
}

static int fast_prepare_pool(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                             lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    if ((lwm_read_u16(node) != LW_OP_AVERAGE_POOL && lwm_read_u16(node) != LW_OP_MAX_POOL) ||
        lwm_read_u16(node + 2u) != 1u || lwm_read_u16(node + 4u) != 1u) return 0;
    input = &plan->session->tensors[lwm_read_u32(node + 8u)]; output = &plan->session->tensors[lwm_read_u32(node + 40u)];
    if (!fast_tensor_eligible(input) || !fast_tensor_eligible(output)) return 0;
    if (lwm_read_i32(params + 8u) <= 0 || lwm_read_i32(params + 12u) <= 0 || lwm_read_i32(params + 16u) <= 0 || lwm_read_i32(params + 20u) <= 0) return 0;
    memset(&fast_node->data.spatial, 0, sizeof(fast_node->data.spatial));
    fast_node->data.spatial.input_index = lwm_read_u32(node + 8u); fast_node->data.spatial.output_index = lwm_read_u32(node + 40u);
    fast_node->data.spatial.kernel_h = (uint32_t)lwm_read_i32(params + 8u); fast_node->data.spatial.kernel_w = (uint32_t)lwm_read_i32(params + 12u);
    fast_node->data.spatial.stride_h = (uint32_t)lwm_read_i32(params + 16u); fast_node->data.spatial.stride_w = (uint32_t)lwm_read_i32(params + 20u);
    fast_node->data.spatial.pad_top = (uint32_t)lwm_read_i32(params + 24u); fast_node->data.spatial.pad_left = (uint32_t)lwm_read_i32(params + 28u);
    fast_node->data.spatial.pad_bottom = (uint32_t)lwm_read_i32(params + 32u); fast_node->data.spatial.pad_right = (uint32_t)lwm_read_i32(params + 36u);
    fast_node->data.spatial.count_include_pad = (uint8_t)lwm_read_u32(params + 44u);
    fast_node->data.spatial.is_max = (uint8_t)(lwm_read_u16(node) == LW_OP_MAX_POOL);
    fast_node->kind = LW_X64_FAST_NODE_POOL;
    ++plan->pool_node_count;
    return 1;
}
static int fast_prepare_concat(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                               lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* output;
    uint32_t count;
    uint32_t slot;
    int32_t axis;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    count = lwm_read_u16(node + 2u);
    axis = lwm_read_i32(params + 4u); if (axis < 0) axis += 4;
    if (lwm_read_u16(node) != LW_OP_CONCAT || count == 0u || count > LWM_V0_MAX_NODE_INPUTS ||
        lwm_read_u16(node + 4u) != 1u || axis != 1) return 0;
    output = &plan->session->tensors[lwm_read_u32(node + 40u)];
    if (!fast_tensor_eligible(output)) return 0;
    memset(&fast_node->data.spatial, 0, sizeof(fast_node->data.spatial));
    fast_node->data.spatial.output_index = lwm_read_u32(node + 40u);
    fast_node->data.spatial.input_count = count;
    for (slot = 0u; slot < count; ++slot) {
        uint32_t input_index = lwm_read_u32(node + 8u + (size_t)slot * sizeof(uint32_t));
        const lw_runtime_tensor* input;
        if (input_index >= plan->tensor_count || !fast_tensor_eligible(input = &plan->session->tensors[input_index]) ||
            input->dimensions[0] != output->dimensions[0] || input->dimensions[2] != output->dimensions[2] ||
            input->dimensions[3] != output->dimensions[3]) return 0;
        fast_node->data.spatial.input_indices[slot] = input_index;
        fast_node->data.spatial.input_channels[slot] = (uint32_t)input->dimensions[1];
    }
    fast_node->kind = LW_X64_FAST_NODE_CONCAT;
    ++plan->concat_node_count;
    return 1;
}
static int fast_prepare_batch_norm(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                   lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    uint32_t input_index;
    uint32_t output_index;
    uint32_t c;
    const float* scale;
    const float* bias;
    const float* mean;
    const float* variance;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
    params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    if (lwm_read_u16(node) != LW_OP_BATCH_NORMALIZATION || lwm_read_u16(node + 2u) < 5u || lwm_read_u16(node + 4u) != 1u) return 0;
    input_index = lwm_read_u32(node + 8u); output_index = lwm_read_u32(node + 40u);
    if (input_index >= plan->tensor_count || output_index >= plan->tensor_count) return 0;
    input = &plan->session->tensors[input_index]; output = &plan->session->tensors[output_index];
    if (!fast_tensor_eligible(input) || !fast_tensor_eligible(output) ||
        input->dimensions[0] != output->dimensions[0] || input->dimensions[1] != output->dimensions[1] ||
        input->dimensions[2] != output->dimensions[2] || input->dimensions[3] != output->dimensions[3]) return 0;
    scale = fast_constant_f32(plan->session, lwm_read_u32(node + 12u));
    bias = fast_constant_f32(plan->session, lwm_read_u32(node + 16u));
    mean = fast_constant_f32(plan->session, lwm_read_u32(node + 20u));
    variance = fast_constant_f32(plan->session, lwm_read_u32(node + 24u));
    c = (uint32_t)input->dimensions[1];
    if (scale == NULL || bias == NULL || mean == NULL || variance == NULL ||
        plan->session->tensors[lwm_read_u32(node + 12u)].byte_size < (uint64_t)c * sizeof(float) ||
        plan->session->tensors[lwm_read_u32(node + 16u)].byte_size < (uint64_t)c * sizeof(float) ||
        plan->session->tensors[lwm_read_u32(node + 20u)].byte_size < (uint64_t)c * sizeof(float) ||
        plan->session->tensors[lwm_read_u32(node + 24u)].byte_size < (uint64_t)c * sizeof(float)) return 0;
    memset(&fast_node->data.spatial, 0, sizeof(fast_node->data.spatial));
    fast_node->data.spatial.input_index = input_index; fast_node->data.spatial.output_index = output_index;
    fast_node->data.spatial.scale = scale; fast_node->data.spatial.bias = bias;
    fast_node->data.spatial.mean = mean; fast_node->data.spatial.variance = variance;
    fast_node->data.spatial.epsilon = fast_read_f32(params + 4u);
    fast_node->kind = LW_X64_FAST_NODE_BATCH_NORM;
    ++plan->batch_norm_node_count;
    return 1;
}
static int fast_conv_output_matches(const lw_runtime_tensor* input, const lw_runtime_tensor* output,
                                    int32_t kh, int32_t kw, int32_t sh, int32_t sw,
                                    int32_t pt, int32_t pl, int32_t pb, int32_t pr) {
    int64_t numerator_h;
    int64_t numerator_w;
    if (input == NULL || output == NULL || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 ||
        pt < 0 || pl < 0 || pb < 0 || pr < 0) return 0;
    numerator_h = (int64_t)input->dimensions[2] + pt + pb - kh;
    numerator_w = (int64_t)input->dimensions[3] + pl + pr - kw;
    if (numerator_h < 0 || numerator_w < 0) return 0;
    return numerator_h / sh + 1 == output->dimensions[2] &&
           numerator_w / sw + 1 == output->dimensions[3];
}
static int fast_prepare_conv(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                             lw_x64_fast_node* fast_node) {
    lw_session* session;
    const uint8_t* node;
    const uint8_t* params;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* weight;
    const lw_runtime_tensor* output;
    lw_x64_fast_conv* conv;
    uint32_t input_index;
    uint32_t weight_index;
    uint32_t bias_index;
    uint32_t group;
    int32_t kh, kw, sh, sw, dh, dw, pt, pl, pb, pr;
    uint64_t packed_count;
    uint64_t scratch_bytes;
    int is_depthwise;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    session = plan->session;
    node = session->model->bytes + (size_t)session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(node) != LW_OP_CONV || lwm_read_u16(node + 2u) < 2u ||
        lwm_read_u16(node + 2u) > 3u || lwm_read_u16(node + 4u) != 1u) return 0;
    params = session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    input_index = lwm_read_u32(node + 8u);
    weight_index = lwm_read_u32(node + 12u);
    bias_index = lwm_read_u16(node + 2u) >= 3u ? lwm_read_u32(node + 16u) : UINT32_MAX;
    if (input_index >= plan->tensor_count || weight_index >= plan->tensor_count ||
        (bias_index != UINT32_MAX && bias_index >= plan->tensor_count)) return 0;
    input = &session->tensors[input_index];
    weight = &session->tensors[weight_index];
    if (lwm_read_u32(node + 40u) >= plan->tensor_count) return 0;
    output = &session->tensors[lwm_read_u32(node + 40u)];
    group = lwm_read_u32(params + 4u);
    kh = lwm_read_i32(params + 8u); kw = lwm_read_i32(params + 12u);
    sh = lwm_read_i32(params + 16u); sw = lwm_read_i32(params + 20u);
    dh = lwm_read_i32(params + 24u); dw = lwm_read_i32(params + 28u);
    pt = lwm_read_i32(params + 32u); pl = lwm_read_i32(params + 36u);
    pb = lwm_read_i32(params + 40u); pr = lwm_read_i32(params + 44u);
    is_depthwise = group == (uint32_t)input->dimensions[1] && group > 1u &&
                  output->dimensions[1] == input->dimensions[1] &&
                  (input->dimensions[1] & 7) == 0u &&
                  weight->dimensions[1] == 1;
    if ((!is_depthwise && group != 1u) || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || dh != 1 || dw != 1 ||
        pt < 0 || pl < 0 || pb < 0 || pr < 0 || !fast_conv_output_matches(input, output, kh, kw, sh, sw, pt, pl, pb, pr) || !fast_tensor_eligible(input) ||
        !fast_tensor_eligible(output) || weight->dtype != LW_DTYPE_F32 || weight->rank != 4u ||
        (weight->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u ||
        weight->dimensions[0] != output->dimensions[1] ||
        (is_depthwise ? weight->dimensions[1] != 1 : weight->dimensions[1] != input->dimensions[1]) ||
        weight->dimensions[2] != kh ||
        weight->dimensions[3] != kw ||
        (is_depthwise ? ((uint32_t)output->dimensions[1] < 8u || ((uint32_t)output->dimensions[1] & 7u) != 0u) :
         ((uint32_t)output->dimensions[1] < 16u || ((uint32_t)output->dimensions[1] % LW_NHWC_OC_BLOCK) != 0u))) return 0;
    conv = &fast_node->data.conv;
    memset(conv, 0, sizeof(*conv));
    conv->input_index = input_index; conv->output_index = lwm_read_u32(node + 40u);
    conv->weight_index = weight_index; conv->bias_index = bias_index;
    conv->input_channels = (uint32_t)input->dimensions[1];
    conv->output_channels = (uint32_t)output->dimensions[1];
    conv->input_height = (uint32_t)input->dimensions[2];
    conv->input_width = (uint32_t)input->dimensions[3];
    conv->output_height = (uint32_t)output->dimensions[2];
    conv->output_width = (uint32_t)output->dimensions[3];
    conv->kernel_h = (uint32_t)kh; conv->kernel_w = (uint32_t)kw;
    conv->stride_h = (uint32_t)sh; conv->stride_w = (uint32_t)sw;
    conv->pad_top = (uint32_t)pt; conv->pad_left = (uint32_t)pl;
    conv->pad_bottom = (uint32_t)pb; conv->pad_right = (uint32_t)pr;
    conv->dense_kc = LW_NHWC_DENSE_KC;
    conv->bias = bias_index == UINT32_MAX ? NULL : fast_constant_f32(session, bias_index);
    if (bias_index != UINT32_MAX && conv->bias == NULL) return 0;
    if (fast_constant_f32(session, weight_index) == NULL ||
        (is_depthwise ? !lw_nhwc_depthwise_packed_weight_count(conv->input_channels,
            conv->kernel_h, conv->kernel_w, &packed_count) :
         !lw_nhwc_dense_packed_weight_count(conv->input_channels, conv->output_channels,
            conv->kernel_h, conv->kernel_w, &packed_count)) ||
        packed_count > SIZE_MAX / sizeof(float)) return 0;
    conv->packed_weights = (float*)malloc((size_t)packed_count * sizeof(float));
    if (conv->packed_weights == NULL) return 0;
    conv->packed_weight_count = packed_count;
    if (is_depthwise) {
        lw_pack_nhwc_depthwise_f32(fast_constant_f32(session, weight_index),
            conv->input_channels, conv->kernel_h, conv->kernel_w, conv->packed_weights);
        fast_node->kind = LW_X64_FAST_NODE_DEPTHWISE;
        ++plan->depthwise_node_count;
        return 1;
    }
    lw_pack_nhwc_dense_f32(fast_constant_f32(session, weight_index), conv->input_channels,
                           conv->output_channels, conv->kernel_h, conv->kernel_w,
                           conv->packed_weights);
    if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0) {
        fast_node->kind = LW_X64_FAST_NODE_POINTWISE;
        ++plan->pointwise_node_count;
    } else {
        lw_nhwc_dense_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = (uint32_t)input->dimensions[0];
        desc.input_channels = conv->input_channels; desc.input_height = conv->input_height;
        desc.input_width = conv->input_width; desc.output_channels = conv->output_channels;
        desc.output_height = conv->output_height; desc.output_width = conv->output_width;
        desc.kernel_h = conv->kernel_h; desc.kernel_w = conv->kernel_w;
        desc.stride_h = conv->stride_h; desc.stride_w = conv->stride_w;
        desc.pad_top = conv->pad_top; desc.pad_left = conv->pad_left;
        desc.pad_bottom = conv->pad_bottom; desc.pad_right = conv->pad_right;
        desc.dense_kc = conv->dense_kc;
        if (!lw_nhwc_dense_scratch_bytes(&desc, &scratch_bytes) || scratch_bytes > SIZE_MAX) {
            free(conv->packed_weights); conv->packed_weights = NULL; return 0;
        }
        conv->scratch_bytes = scratch_bytes;
        if ((size_t)scratch_bytes > plan->scratch_bytes) plan->scratch_bytes = (size_t)scratch_bytes;
        fast_node->kind = LW_X64_FAST_NODE_DENSE;
        ++plan->dense_node_count;
    }
    return 1;
}
static lw_status fast_init_tensor_states(lw_x64_rec_fast_plan* plan, lw_error* error) {
    if (plan == NULL || plan->session == NULL || plan->session->model == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    plan->tensor_count = plan->session->model->info.tensor_count;
    plan->tensors = (lw_x64_fast_tensor_state*)calloc(plan->tensor_count, sizeof(*plan->tensors));
    if (plan->tensors == NULL && plan->tensor_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast tensor states");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t index = 0u; index < plan->tensor_count; ++index) {
        lw_x64_fast_tensor_state* state = &plan->tensors[index];
        state->nhwc_offset = LW_X64_FAST_OFFSET_NONE;
        state->neutral = (uint8_t)lw_layout_tensor_is_neutral(plan->session, index);
        state->available_layouts = state->neutral
            ? (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC)
            : LW_X64_FAST_HAVE_NCHW;
    }
    return LW_STATUS_OK;
}

typedef struct fast_workspace_interval {
    uint32_t tensor_index;
    int32_t birth;
    int32_t last_use;
    size_t bytes;
} fast_workspace_interval;

typedef struct fast_workspace_slot {
    size_t offset;
    size_t bytes;
    int32_t last_use;
} fast_workspace_slot;

static int fast_tensor_needs_nhwc(const lw_x64_rec_fast_plan* plan, uint32_t tensor_index) {
    const lw_runtime_tensor* tensor;
    if (plan == NULL || tensor_index >= plan->tensor_count) return 0;
    tensor = &plan->session->tensors[tensor_index];
    return fast_tensor_eligible(tensor) && plan->tensors[tensor_index].neutral == 0u &&
           (plan->layout.tensor_available_layouts[tensor_index] & LW_LAYOUT_AVAILABLE_NHWC) != 0u;
}

static lw_status fast_allocate_nhwc_workspace(lw_x64_rec_fast_plan* plan, lw_error* error) {
    fast_workspace_interval* intervals = NULL;
    fast_workspace_slot* slots = NULL;
    uint32_t interval_count = 0u;
    uint32_t slot_count = 0u;
    size_t workspace_end = 0u;
    if (plan == NULL) return LW_STATUS_INVALID_ARGUMENT;
    intervals = (fast_workspace_interval*)calloc(plan->tensor_count, sizeof(*intervals));
    slots = (fast_workspace_slot*)calloc(plan->tensor_count, sizeof(*slots));
    if ((intervals == NULL || slots == NULL) && plan->tensor_count != 0u) {
        free(intervals); free(slots);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast workspace intervals");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t index = 0u; index < plan->tensor_count; ++index) {
        const lw_runtime_tensor* tensor = &plan->session->tensors[index];
        if (!fast_tensor_needs_nhwc(plan, index)) continue;
        if (tensor->byte_size > SIZE_MAX - 63u) {
            free(intervals); free(slots);
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast NHWC tensor size overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        intervals[interval_count].tensor_index = index;
        intervals[interval_count].birth = tensor->birth_node < 0 ? 0 : tensor->birth_node;
        intervals[interval_count].last_use = tensor->last_use_node < 0 ? 0 : tensor->last_use_node;
        intervals[interval_count].bytes = fast_align64((size_t)tensor->byte_size);
        ++interval_count;
    }
    for (uint32_t index = 1u; index < interval_count; ++index) {
        fast_workspace_interval value = intervals[index];
        uint32_t position = index;
        while (position > 0u && intervals[position - 1u].birth > value.birth) {
            intervals[position] = intervals[position - 1u];
            --position;
        }
        intervals[position] = value;
    }
    for (uint32_t index = 0u; index < interval_count; ++index) {
        fast_workspace_interval* interval = &intervals[index];
        uint32_t best_slot = UINT32_MAX;
        size_t best_waste = SIZE_MAX;
        for (uint32_t slot_index = 0u; slot_index < slot_count; ++slot_index) {
            fast_workspace_slot* slot = &slots[slot_index];
            if (slot->last_use < interval->birth && slot->bytes >= interval->bytes) {
                size_t waste = slot->bytes - interval->bytes;
                if (waste < best_waste) { best_waste = waste; best_slot = slot_index; }
            }
        }
        if (best_slot != UINT32_MAX) {
            plan->tensors[interval->tensor_index].nhwc_offset = (uint64_t)slots[best_slot].offset;
            slots[best_slot].last_use = interval->last_use;
        } else {
            workspace_end = fast_align64(workspace_end);
            if (interval->bytes > SIZE_MAX - workspace_end) {
                free(intervals); free(slots);
                lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast NHWC workspace size overflows");
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            plan->tensors[interval->tensor_index].nhwc_offset = (uint64_t)workspace_end;
            slots[slot_count].offset = workspace_end;
            slots[slot_count].bytes = interval->bytes;
            slots[slot_count].last_use = interval->last_use;
            ++slot_count;
            workspace_end += interval->bytes;
        }
    }
    free(intervals); free(slots);
    plan->nhwc_workspace_bytes = fast_align64(workspace_end);
    if (plan->nhwc_workspace_bytes != 0u) {
        plan->nhwc_workspace = (uint8_t*)malloc(plan->nhwc_workspace_bytes);
        if (plan->nhwc_workspace == NULL) {
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast NHWC workspace");
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    return LW_STATUS_OK;
}
lw_status lw_x64_rec_fast_plan_create(lw_session* session,
                                      lw_x64_rec_fast_plan** out_plan,
                                      lw_error* error) {
    lw_x64_rec_fast_plan* plan;
    lw_layout_planner_options options;
    lw_status status;
    if (out_plan == NULL || session == NULL || session->model == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "fast plan requires a session");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out_plan = NULL;
    plan = (lw_x64_rec_fast_plan*)calloc(1u, sizeof(*plan));
    if (plan == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast plan");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    plan->session = session;
    plan->node_count = session->model->info.node_count;
    plan->nodes = (lw_x64_fast_node*)calloc(plan->node_count, sizeof(*plan->nodes));
    if (plan->nodes == NULL && plan->node_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast nodes");
        lw_x64_rec_fast_plan_free(plan);
        return LW_STATUS_OUT_OF_MEMORY;
    }
    options.allow_direct_nhwc_graph_input = 0u;
    status = lw_layout_plan_build(session, &options, &plan->layout, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    status = fast_init_tensor_states(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    plan->graph_input_index = lwm_read_u32(session->model->bytes +
                                           (size_t)session->model->input_offset);
    plan->graph_output_index = lwm_read_u32(session->model->bytes +
                                            (size_t)session->model->output_offset);
    for (uint32_t node_index = 0u; node_index < plan->node_count; ++node_index) {
        const uint8_t* node = session->model->bytes + (size_t)session->model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        plan->nodes[node_index].semantic_node_index = node_index;
        plan->nodes[node_index].kind = LW_X64_FAST_NODE_GENERIC;
        plan->nodes[node_index].semantic_node_count = 1u;
        plan->nodes[node_index].output_index = lwm_read_u32(node + 40u);
        if (fast_prepare_gelu(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_elementwise(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_reduce_mean(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_pool(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_concat(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_batch_norm(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_conv(plan, node_index, &plan->nodes[node_index])) {
            ++plan->fast_node_count;
        } else if (plan->layout.node_layout[node_index] == LW_FAST_LAYOUT_NHWC) {
            fast_record_nhwc_blocker(plan, node);
        }
    }
    status = fast_allocate_nhwc_workspace(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    plan->generic_node_count = plan->node_count - plan->fast_node_count;
    if (plan->scratch_bytes != 0u) {
        plan->scratch = (uint8_t*)malloc(plan->scratch_bytes);
        if (plan->scratch == NULL) {
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast scratch");
            lw_x64_rec_fast_plan_free(plan);
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    lw_set_error(error, LW_STATUS_OK, "");
    *out_plan = plan;
    return LW_STATUS_OK;
}

void lw_x64_rec_fast_plan_free(lw_x64_rec_fast_plan* plan) {
    if (plan == NULL) return;
    if (plan->nodes != NULL) {
        uint32_t index;
        for (index = 0u; index < plan->node_count; ++index) {
            if (plan->nodes[index].kind == LW_X64_FAST_NODE_POINTWISE ||
                plan->nodes[index].kind == LW_X64_FAST_NODE_DENSE ||
                plan->nodes[index].kind == LW_X64_FAST_NODE_DEPTHWISE) {
                free(plan->nodes[index].data.conv.packed_weights);
            }
        }
    }
    free(plan->scratch);
    free(plan->nhwc_workspace);
    free(plan->tensors);
    free(plan->nodes);
    lw_layout_plan_free(&plan->layout);
    free(plan);
}
