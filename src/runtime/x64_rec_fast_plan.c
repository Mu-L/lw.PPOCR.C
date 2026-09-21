#include "x64_rec_fast_internal.h"

#include "lwm_read.h"
#include "operator_internal.h"

#include <limits.h>
#include <math.h>
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

static int fast_gelu_temporaries_are_private(const lw_x64_rec_fast_plan* plan,
                                             uint32_t node_index,
                                             const lw_fused_gelu_match* match) {
    uint32_t index;
    if (plan == NULL || match == NULL || plan->session == NULL ||
        node_index > plan->node_count || plan->node_count - node_index < 5u) return 0;
    for (index = 0u; index < 4u; ++index) {
        uint32_t tensor_index = match->outputs[index];
        if (tensor_index >= plan->tensor_count || plan->consumer_count == NULL ||
            plan->consumer_count[tensor_index] != 1u ||
            plan->session->tensors[tensor_index].last_use_node !=
                (int32_t)(node_index + index + 1u)) return 0;
    }
    return 1;
}
static void fast_elide_tensor(lw_x64_rec_fast_plan* plan, uint32_t tensor_index);
static int fast_prepare_gelu(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                             lw_x64_fast_node* fast_node) {
    lw_fused_gelu_match match;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        !lw_match_fused_gelu(plan->session, node_index, &match)) return 0;
    if (!fast_gelu_temporaries_are_private(plan, node_index, &match)) return 0;
    for (uint32_t index = 1u; index < 5u; ++index) {
        if (plan->layout.node_layout[node_index + index] !=
            plan->layout.node_layout[node_index]) return 0;
    }
    if (match.inputs[0][0] >= plan->tensor_count || match.outputs[4] >= plan->tensor_count) return 0;
    input = &plan->session->tensors[match.inputs[0][0]];
    output = &plan->session->tensors[match.outputs[4]];
    if (!fast_same_shape4(input, output)) return 0;
    memset(&fast_node->data.elementwise, 0, sizeof(fast_node->data.elementwise));
    fast_node->data.elementwise.input_index = match.inputs[0][0];
    fast_node->data.elementwise.output_index = match.outputs[4];
    fast_node->data.elementwise.operation = LW_OP_ERF;
    for (uint32_t index = 0u; index < 4u; ++index) fast_elide_tensor(plan, match.outputs[index]);
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
    conv->residual_index = UINT32_MAX;
    conv->activation = LW_NHWC_ACT_NONE;
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
    if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && pb == 0 && pr == 0) {
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

static lw_status fast_build_consumer_counts(lw_x64_rec_fast_plan* plan, lw_error* error) {
    uint32_t node_index;
    if (plan == NULL || plan->session == NULL || plan->session->model == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    plan->consumer_count = (uint32_t*)calloc(plan->tensor_count, sizeof(*plan->consumer_count));
    plan->elided_tensor = (uint8_t*)calloc(plan->tensor_count, sizeof(*plan->elided_tensor));
    if ((plan->consumer_count == NULL || plan->elided_tensor == NULL) && plan->tensor_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast consumer metadata");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (node_index = 0u; node_index < plan->node_count; ++node_index) {
        const uint8_t* node = plan->session->model->bytes +
            (size_t)plan->session->model->node_offset +
            (size_t)node_index * LWM_V0_NODE_SIZE;
        uint32_t input_count = lwm_read_u16(node + 2u);
        for (uint32_t input_slot = 0u; input_slot < input_count; ++input_slot) {
            uint32_t tensor_index = lwm_read_u32(node + 8u +
                (size_t)input_slot * sizeof(uint32_t));
            if (tensor_index >= plan->tensor_count) {
                lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast consumer tensor index out of bounds");
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            if (plan->consumer_count[tensor_index] == UINT32_MAX) {
                lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast consumer count overflow");
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            ++plan->consumer_count[tensor_index];
        }
    }
    return LW_STATUS_OK;
}

static void fast_elide_tensor(lw_x64_rec_fast_plan* plan, uint32_t tensor_index) {
    if (plan != NULL && plan->elided_tensor != NULL && tensor_index < plan->tensor_count &&
        plan->elided_tensor[tensor_index] == 0u) {
        plan->elided_tensor[tensor_index] = 1u;
        if (plan->elided_tensor_count != UINT32_MAX) ++plan->elided_tensor_count;
    }
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
    if (plan->elided_tensor != NULL && plan->elided_tensor[tensor_index] != 0u) return 0;
    return fast_tensor_eligible(tensor) && plan->tensors[tensor_index].neutral == 0u &&
           (plan->layout.tensor_available_layouts[tensor_index] & LW_LAYOUT_AVAILABLE_NHWC) != 0u;
}


static int fast_tensor_needs_nhwc(const lw_x64_rec_fast_plan* plan, uint32_t tensor_index);

static lw_status fast_build_semantic_to_physical(lw_x64_rec_fast_plan* plan, lw_error* error) {
    if (plan == NULL) return LW_STATUS_INVALID_ARGUMENT;
    plan->semantic_to_physical = (uint32_t*)malloc((size_t)plan->node_count * sizeof(*plan->semantic_to_physical));
    if (plan->semantic_to_physical == NULL && plan->node_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate semantic-to-physical map");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t index = 0u; index < plan->node_count; ++index) {
        plan->semantic_to_physical[index] = UINT32_MAX;
    }
    for (uint32_t op_index = 0u; op_index < plan->op_count; ++op_index) {
        const lw_x64_physical_op* op = &plan->ops[op_index];
        uint32_t begin = op->semantic_begin;
        uint32_t count = op->semantic_count == 0u ? 1u : op->semantic_count;
        if (begin >= plan->node_count || count > plan->node_count - begin) {
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "invalid x64 physical semantic span");
            return LW_STATUS_INVALID_SHAPE;
        }
        for (uint32_t offset = 0u; offset < count; ++offset) {
            uint32_t semantic = begin + offset;
            if (plan->semantic_to_physical[semantic] != UINT32_MAX) {
                lw_set_error(error, LW_STATUS_INVALID_SHAPE, "overlapping x64 physical semantic spans");
                return LW_STATUS_INVALID_SHAPE;
            }
            plan->semantic_to_physical[semantic] = op_index;
        }
    }
    for (uint32_t index = 0u; index < plan->node_count; ++index) {
        if (plan->semantic_to_physical[index] == UINT32_MAX) {
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "semantic node is missing from x64 physical plan");
            return LW_STATUS_INVALID_SHAPE;
        }
    }
    return LW_STATUS_OK;
}

static lw_status fast_build_physical_lifetimes(lw_x64_rec_fast_plan* plan, lw_error* error) {
    if (plan == NULL || plan->session == NULL || plan->semantic_to_physical == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    plan->physical_lifetimes = (lw_x64_fast_tensor_lifetime*)calloc(
        plan->tensor_count, sizeof(*plan->physical_lifetimes));
    if (plan->physical_lifetimes == NULL && plan->tensor_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate physical tensor lifetimes");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t index = 0u; index < plan->tensor_count; ++index) {
        const lw_runtime_tensor* tensor = &plan->session->tensors[index];
        lw_x64_fast_tensor_lifetime* lifetime = &plan->physical_lifetimes[index];
        lifetime->birth_op = 0;
        lifetime->last_use_op = 0;
        if (tensor->birth_node >= 0) {
            uint32_t semantic = (uint32_t)tensor->birth_node;
            if (semantic >= plan->node_count || plan->semantic_to_physical[semantic] >= plan->op_count) {
                lw_set_error(error, LW_STATUS_INVALID_SHAPE, "tensor birth semantic node is not in physical plan");
                return LW_STATUS_INVALID_SHAPE;
            }
            lifetime->birth_op = (int32_t)plan->semantic_to_physical[semantic];
        }
        if (tensor->last_use_node >= 0) {
            uint32_t semantic = (uint32_t)tensor->last_use_node;
            if (semantic == plan->node_count && plan->op_count != 0u) {
                /* The session planner uses node_count as the graph-output sentinel. */
                lifetime->last_use_op = (int32_t)(plan->op_count - 1u);
            } else {
                if (semantic >= plan->node_count || plan->semantic_to_physical[semantic] >= plan->op_count) {
                    lw_set_error(error, LW_STATUS_INVALID_SHAPE, "tensor last-use semantic node is not in physical plan");
                    return LW_STATUS_INVALID_SHAPE;
                }
                lifetime->last_use_op = (int32_t)plan->semantic_to_physical[semantic];
            }
        }
        if (lifetime->last_use_op < lifetime->birth_op) lifetime->last_use_op = lifetime->birth_op;
    }
    return LW_STATUS_OK;
}

static int fast_lifetimes_overlap(const lw_x64_fast_tensor_lifetime* left,
                                  const lw_x64_fast_tensor_lifetime* right) {
    return left != NULL && right != NULL &&
           left->birth_op <= right->last_use_op &&
           right->birth_op <= left->last_use_op;
}

static int fast_ranges_overlap(uint64_t left_offset, uint64_t left_bytes,
                               uint64_t right_offset, uint64_t right_bytes) {
    if (left_bytes == 0u || right_bytes == 0u) return 0;
    return left_offset < right_offset
        ? left_bytes > right_offset - left_offset
        : right_bytes > left_offset - right_offset;
}

static int fast_validate_workspace_aliasing(const lw_x64_rec_fast_plan* plan) {
    if (plan == NULL || plan->physical_lifetimes == NULL) return 1;
    for (uint32_t left = 0u; left < plan->tensor_count; ++left) {
        if (!fast_tensor_needs_nhwc(plan, left)) continue;
        for (uint32_t right = left + 1u; right < plan->tensor_count; ++right) {
            if (!fast_tensor_needs_nhwc(plan, right) ||
                !fast_lifetimes_overlap(&plan->physical_lifetimes[left],
                                        &plan->physical_lifetimes[right])) continue;
            if (fast_ranges_overlap(plan->tensors[left].nhwc_offset,
                                     plan->session->tensors[left].byte_size,
                                     plan->tensors[right].nhwc_offset,
                                     plan->session->tensors[right].byte_size)) return 0;
        }
    }
    return 1;
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
        if (plan->physical_lifetimes == NULL) {
            free(intervals); free(slots);
            lw_set_error(error, LW_STATUS_INVALID_SHAPE, "physical tensor lifetimes are unavailable");
            return LW_STATUS_INVALID_SHAPE;
        }
        intervals[interval_count].birth = plan->physical_lifetimes[index].birth_op;
        intervals[interval_count].last_use = plan->physical_lifetimes[index].last_use_op;
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

static int fast_conv_epilogue_supported(uint16_t kind) {
    return kind == LW_X64_FAST_NODE_POINTWISE || kind == LW_X64_FAST_NODE_DENSE;
}

static int fast_conv_bn_fold_supported(uint16_t kind) {
    return kind == LW_X64_FAST_NODE_POINTWISE ||
           kind == LW_X64_FAST_NODE_DENSE ||
           kind == LW_X64_FAST_NODE_DEPTHWISE;
}

static int fast_scale_conv_weights(lw_x64_fast_conv* conv, uint16_t kind,
                                   const float* scale, const float* bn_bias,
                                   const float* mean, const float* variance,
                                   float epsilon) {
    uint32_t c;
    float* folded_bias;
    float* mul;
    if (conv == NULL || scale == NULL || bn_bias == NULL || mean == NULL || variance == NULL ||
        conv->packed_weights == NULL || conv->output_channels == 0u) return 0;
    folded_bias = (float*)malloc((size_t)conv->output_channels * sizeof(float));
    mul = (float*)malloc((size_t)conv->output_channels * sizeof(float));
    if (folded_bias == NULL || mul == NULL) {
        free(folded_bias);
        free(mul);
        return 0;
    }
    for (c = 0u; c < conv->output_channels; ++c) {
        float variance_epsilon = variance[c] + epsilon;
        float old_bias = conv->bias == NULL ? 0.0f : conv->bias[c];
        if (!(variance_epsilon > 0.0f) || !isfinite(variance_epsilon)) {
            free(folded_bias);
            free(mul);
            return 0;
        }
        mul[c] = scale[c] / sqrtf(variance_epsilon);
        if (!isfinite(mul[c])) {
            free(folded_bias);
            free(mul);
            return 0;
        }
        folded_bias[c] = old_bias * mul[c] + bn_bias[c] - mean[c] * mul[c];
    }
    if (kind == LW_X64_FAST_NODE_DEPTHWISE) {
        uint32_t taps = conv->kernel_h * conv->kernel_w;
        uint32_t blocks = (conv->output_channels + LW_NHWC_DEPTHWISE_BLOCK - 1u) /
                          LW_NHWC_DEPTHWISE_BLOCK;
        for (uint32_t block = 0u; block < blocks; ++block) {
            for (uint32_t tap = 0u; tap < taps; ++tap) {
                float* weights = conv->packed_weights +
                    ((size_t)block * taps + tap) * LW_NHWC_DEPTHWISE_BLOCK;
                for (uint32_t lane = 0u; lane < LW_NHWC_DEPTHWISE_BLOCK; ++lane) {
                    uint32_t channel = block * LW_NHWC_DEPTHWISE_BLOCK + lane;
                    if (channel < conv->output_channels) weights[lane] *= mul[channel];
                }
            }
        }
    } else {
        uint64_t k_total = (uint64_t)conv->input_channels * conv->kernel_h * conv->kernel_w;
        uint32_t blocks = conv->output_channels / LW_NHWC_OC_BLOCK;
        for (uint32_t block = 0u; block < blocks; ++block) {
            for (uint64_t k = 0u; k < k_total; ++k) {
                float* weights = conv->packed_weights +
                    ((size_t)block * k_total + (size_t)k) * LW_NHWC_OC_BLOCK;
                for (uint32_t lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                    uint32_t channel = block * LW_NHWC_OC_BLOCK + lane;
                    weights[lane] *= mul[channel];
                }
            }
        }
    }
    free(mul);
    free(conv->owned_bias);
    conv->owned_bias = folded_bias;
    conv->bias = folded_bias;
    return 1;
}
static void fast_set_physical_output(lw_x64_physical_op* op, uint32_t tensor_index) {
    if (op == NULL) return;
    op->output_index = tensor_index;
    if (op->kind == LW_X64_FAST_NODE_POINTWISE ||
        op->kind == LW_X64_FAST_NODE_DENSE ||
        op->kind == LW_X64_FAST_NODE_DEPTHWISE) {
        op->data.conv.output_index = tensor_index;
    }
}

static int fast_fuse_conv_bn(lw_x64_rec_fast_plan* plan, lw_x64_physical_op* op,
                             uint32_t bn_node_index) {
    const uint8_t* node;
    const uint8_t* params;
    uint32_t conv_output;
    uint32_t output_index;
    uint32_t scale_index;
    uint32_t bias_index;
    uint32_t mean_index;
    uint32_t variance_index;
    const lw_runtime_tensor* conv_tensor;
    const lw_runtime_tensor* output_tensor;
    uint32_t channels;
    const float* scale;
    const float* bias;
    const float* mean;
    const float* variance;
    if (plan == NULL || op == NULL || !fast_conv_bn_fold_supported(op->kind) ||
        bn_node_index >= plan->node_count || op->data.conv.output_index >= plan->tensor_count ||
        plan->consumer_count == NULL || plan->consumer_count[op->data.conv.output_index] != 1u) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)bn_node_index * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(node) != LW_OP_BATCH_NORMALIZATION || lwm_read_u16(node + 2u) < 5u ||
        lwm_read_u16(node + 4u) != 1u || lwm_read_u32(node + 8u) != op->data.conv.output_index) return 0;
    conv_output = op->data.conv.output_index;
    output_index = lwm_read_u32(node + 40u);
    if (output_index >= plan->tensor_count) return 0;
    conv_tensor = &plan->session->tensors[conv_output];
    output_tensor = &plan->session->tensors[output_index];
    if (!fast_same_shape4(conv_tensor, output_tensor) || plan->layout.node_layout[bn_node_index] !=
        plan->layout.node_layout[op->semantic_begin]) return 0;
    scale_index = lwm_read_u32(node + 12u);
    bias_index = lwm_read_u32(node + 16u);
    mean_index = lwm_read_u32(node + 20u);
    variance_index = lwm_read_u32(node + 24u);
    channels = (uint32_t)conv_tensor->dimensions[1];
    scale = fast_constant_f32(plan->session, scale_index);
    bias = fast_constant_f32(plan->session, bias_index);
    mean = fast_constant_f32(plan->session, mean_index);
    variance = fast_constant_f32(plan->session, variance_index);
    params = plan->session->model->bytes + (size_t)lwm_read_u64(node + 56u);
    if (scale == NULL || bias == NULL || mean == NULL || variance == NULL ||
        scale_index >= plan->tensor_count || bias_index >= plan->tensor_count ||
        mean_index >= plan->tensor_count || variance_index >= plan->tensor_count ||
        plan->session->tensors[scale_index].byte_size < (uint64_t)channels * sizeof(float) ||
        plan->session->tensors[bias_index].byte_size < (uint64_t)channels * sizeof(float) ||
        plan->session->tensors[mean_index].byte_size < (uint64_t)channels * sizeof(float) ||
        plan->session->tensors[variance_index].byte_size < (uint64_t)channels * sizeof(float)) return 0;
    if (!fast_scale_conv_weights(&op->data.conv, op->kind, scale, bias, mean,
                                 variance, fast_read_f32(params + 4u))) return 0;
    fast_set_physical_output(op, output_index);
    ++op->semantic_count;
    fast_elide_tensor(plan, conv_output);
    return 1;
}

static int fast_fuse_conv_add(lw_x64_rec_fast_plan* plan, lw_x64_physical_op* op,
                              uint32_t add_node_index) {
    const uint8_t* node;
    uint32_t left;
    uint32_t right;
    uint32_t residual;
    uint32_t output_index;
    uint32_t conv_output;
    if (plan == NULL || op == NULL || !fast_conv_epilogue_supported(op->kind) ||
        add_node_index >= plan->node_count || op->data.conv.output_index >= plan->tensor_count ||
        plan->consumer_count == NULL || plan->consumer_count[op->data.conv.output_index] != 1u) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)add_node_index * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(node) != LW_OP_ADD || lwm_read_u16(node + 2u) != 2u ||
        lwm_read_u16(node + 4u) != 1u) return 0;
    conv_output = op->data.conv.output_index;
    left = lwm_read_u32(node + 8u);
    right = lwm_read_u32(node + 12u);
    if (left == op->data.conv.output_index) residual = right;
    else if (right == op->data.conv.output_index) residual = left;
    else return 0;
    output_index = lwm_read_u32(node + 40u);
    if (output_index >= plan->tensor_count || residual >= plan->tensor_count ||
        plan->consumer_count[output_index] != 1u ||
        (plan->session->tensors[residual].flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u ||
        !fast_same_shape4(&plan->session->tensors[op->data.conv.output_index],
                          &plan->session->tensors[residual]) ||
        !fast_same_shape4(&plan->session->tensors[op->data.conv.output_index],
                          &plan->session->tensors[output_index]) ||
        plan->layout.node_layout[add_node_index] != plan->layout.node_layout[op->semantic_begin]) return 0;
    op->data.conv.residual_index = residual;
    fast_set_physical_output(op, output_index);
    ++op->semantic_count;
    fast_elide_tensor(plan, conv_output);
    return 1;
}

static int fast_fuse_conv_relu(lw_x64_rec_fast_plan* plan, lw_x64_physical_op* op,
                               uint32_t relu_node_index) {
    const uint8_t* node;
    uint32_t input_index;
    uint32_t output_index;
    if (plan == NULL || op == NULL || !fast_conv_epilogue_supported(op->kind) ||
        relu_node_index >= plan->node_count || op->data.conv.output_index >= plan->tensor_count ||
        plan->consumer_count == NULL || plan->consumer_count[op->data.conv.output_index] != 1u) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)relu_node_index * LWM_V0_NODE_SIZE;
    if (lwm_read_u16(node) != LW_OP_RELU || lwm_read_u16(node + 2u) != 1u ||
        lwm_read_u16(node + 4u) != 1u) return 0;
    input_index = lwm_read_u32(node + 8u);
    output_index = lwm_read_u32(node + 40u);
    if (input_index != op->data.conv.output_index || output_index >= plan->tensor_count ||
        !fast_same_shape4(&plan->session->tensors[input_index], &plan->session->tensors[output_index]) ||
        plan->layout.node_layout[relu_node_index] != plan->layout.node_layout[op->semantic_begin]) return 0;
    op->data.conv.activation = LW_NHWC_ACT_RELU;
    fast_set_physical_output(op, output_index);
    ++op->semantic_count;
    fast_elide_tensor(plan, input_index);
    return 1;
}
static int fast_physical_op_owns_conv(uint16_t kind) {
    return kind == LW_X64_FAST_NODE_POINTWISE ||
           kind == LW_X64_FAST_NODE_DENSE ||
           kind == LW_X64_FAST_NODE_DEPTHWISE;
}

static void fast_move_node_to_physical(const lw_x64_fast_node* node,
                                       lw_x64_physical_op* op) {
    if (node == NULL || op == NULL) return;
    memset(op, 0, sizeof(*op));
    op->semantic_begin = node->semantic_node_index;
    op->semantic_count = node->semantic_node_count == 0u ? 1u : node->semantic_node_count;
    op->kind = node->kind;
    op->output_index = node->output_index;
    memcpy(&op->data, &node->data, sizeof(op->data));
}

static void fast_release_conv(lw_x64_fast_conv* conv) {
    if (conv == NULL) return;
    free(conv->packed_weights);
    free(conv->owned_bias);
    free(conv->dense_tap_offsets);
    free(conv->dense_patch_offsets);
    conv->packed_weights = NULL;
    conv->owned_bias = NULL;
    conv->dense_tap_offsets = NULL;
    conv->dense_patch_offsets = NULL;
}

static lw_status fast_compile_physical_ops(lw_x64_rec_fast_plan* plan, lw_error* error) {
    uint32_t node_index;
    if (plan == NULL) return LW_STATUS_INVALID_ARGUMENT;
    plan->op_capacity = plan->node_count;
    plan->ops = (lw_x64_physical_op*)calloc(plan->op_capacity, sizeof(*plan->ops));
    if (plan->ops == NULL && plan->op_capacity != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate x64 physical ops");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (node_index = 0u; node_index < plan->node_count;) {
        lw_x64_fast_node* node = &plan->nodes[node_index];
        lw_x64_physical_op* op;
        uint32_t consumed = node->semantic_node_count == 0u ? 1u : node->semantic_node_count;
        if (plan->op_count >= plan->op_capacity) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "x64 physical op count overflow");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        op = &plan->ops[plan->op_count++];
        fast_move_node_to_physical(node, op);
        if (op->kind == LW_X64_FAST_NODE_GENERIC && plan->physical_generic_op_count != UINT32_MAX) ++plan->physical_generic_op_count;
        if (fast_physical_op_owns_conv(op->kind)) {
            node->data.conv.packed_weights = NULL;
            node->data.conv.owned_bias = NULL;
            node->data.conv.dense_tap_offsets = NULL;
            node->data.conv.dense_patch_offsets = NULL;
            {
                uint32_t start = op->semantic_begin;
                int fused_bn = 0;
                int fused_add = 0;
                int fused_relu = 0;
                if (start + 1u < plan->node_count) {
                    fused_bn = fast_fuse_conv_bn(plan, op, start + 1u);
                    if (!fused_bn) fused_add = fast_fuse_conv_add(plan, op, start + 1u);
                }
                if (start + op->semantic_count < plan->node_count) {
                    fused_relu = fast_fuse_conv_relu(plan, op, start + op->semantic_count);
                }
                if (fused_bn && fused_relu) {
                    if (plan->fused_conv_bn_relu_count != UINT32_MAX) ++plan->fused_conv_bn_relu_count;
                } else if (fused_bn) {
                    if (plan->fused_conv_bn_count != UINT32_MAX) ++plan->fused_conv_bn_count;
                } else if (fused_add && fused_relu) {
                    if (plan->fused_conv_add_relu_count != UINT32_MAX) ++plan->fused_conv_add_relu_count;
                } else if (fused_add) {
                    if (plan->fused_conv_add_count != UINT32_MAX) ++plan->fused_conv_add_count;
                } else if (fused_relu) {
                    if (plan->fused_conv_relu_count != UINT32_MAX) ++plan->fused_conv_relu_count;
                }
            }
            consumed = op->semantic_count;
        }
        if (consumed > plan->node_count - node_index) consumed = 1u;
        node_index += consumed;
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
    status = fast_build_consumer_counts(plan, error);
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
    status = fast_compile_physical_ops(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    status = fast_build_semantic_to_physical(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    status = fast_build_physical_lifetimes(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    status = fast_allocate_nhwc_workspace(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    if (!fast_validate_workspace_aliasing(plan)) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "x64 physical workspace alias detected");
        lw_x64_rec_fast_plan_free(plan);
        return LW_STATUS_INVALID_SHAPE;
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
            if (fast_physical_op_owns_conv(plan->nodes[index].kind)) {
                fast_release_conv(&plan->nodes[index].data.conv);
            }
        }
    }
    if (plan->ops != NULL) {
        uint32_t index;
        for (index = 0u; index < plan->op_count; ++index) {
            if (fast_physical_op_owns_conv(plan->ops[index].kind)) {
                fast_release_conv(&plan->ops[index].data.conv);
            }
        }
    }
    free(plan->scratch);
    free(plan->nhwc_workspace);
    free(plan->tensors);
    free(plan->consumer_count);
    free(plan->elided_tensor);
    free(plan->physical_lifetimes);
    free(plan->semantic_to_physical);
    free(plan->nodes);
    free(plan->ops);
    lw_layout_plan_free(&plan->layout);
    free(plan);
}
