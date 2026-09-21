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

static int fast_prepare_elementwise(lw_x64_rec_fast_plan* plan, uint32_t node_index,
                                    lw_x64_fast_node* fast_node) {
    const uint8_t* node;
    uint16_t operation;
    uint16_t input_count;
    uint32_t input_index;
    uint32_t rhs_index = UINT32_MAX;
    uint32_t output_index;
    const lw_runtime_tensor* input;
    const lw_runtime_tensor* output;
    int is_binary;
    if (plan == NULL || fast_node == NULL || node_index >= plan->node_count ||
        plan->layout.node_layout[node_index] != LW_FAST_LAYOUT_NHWC) return 0;
    node = plan->session->model->bytes + (size_t)plan->session->model->node_offset +
           (size_t)node_index * LWM_V0_NODE_SIZE;
    operation = lwm_read_u16(node);
    input_count = lwm_read_u16(node + 2u);
    input_index = lwm_read_u32(node + 8u);
    output_index = lwm_read_u32(node + 40u);
    is_binary = operation == LW_OP_ADD || operation == LW_OP_MUL || operation == LW_OP_DIV || operation == LW_OP_SUB;
    if ((!is_binary && operation != LW_OP_RELU) || input_index >= plan->tensor_count ||
        output_index >= plan->tensor_count) return 0;
    if (operation == LW_OP_DIV && lw_execution_plan_is_fused_gelu_start(plan->session, node_index)) return 0;
    if ((operation == LW_OP_RELU && input_count != 1u) || (is_binary && input_count != 2u)) return 0;
    if (is_binary) {
        rhs_index = lwm_read_u32(node + 12u);
        if (rhs_index >= plan->tensor_count ||
            (plan->session->tensors[rhs_index].flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u ||
            !fast_same_shape4(&plan->session->tensors[input_index], &plan->session->tensors[rhs_index])) return 0;
    }
    input = &plan->session->tensors[input_index];
    output = &plan->session->tensors[output_index];
    if (!fast_same_shape4(input, output)) return 0;
    fast_node->data.elementwise.input_index = input_index;
    fast_node->data.elementwise.rhs_index = rhs_index;
    fast_node->data.elementwise.output_index = output_index;
    fast_node->data.elementwise.operation = operation;
    fast_node->kind = operation == LW_OP_RELU ? LW_X64_FAST_NODE_CONTIGUOUS_UNARY
                                               : LW_X64_FAST_NODE_CONTIGUOUS_BINARY;
    if (operation == LW_OP_RELU) ++plan->relu_node_count;
    else ++plan->binary_node_count;
    return 1;
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
    output = &session->tensors[lwm_read_u32(node + 40u)];
    group = lwm_read_u32(params + 4u);
    kh = lwm_read_i32(params + 8u); kw = lwm_read_i32(params + 12u);
    sh = lwm_read_i32(params + 16u); sw = lwm_read_i32(params + 20u);
    dh = lwm_read_i32(params + 24u); dw = lwm_read_i32(params + 28u);
    pt = lwm_read_i32(params + 32u); pl = lwm_read_i32(params + 36u);
    pb = lwm_read_i32(params + 40u); pr = lwm_read_i32(params + 44u);
    is_depthwise = group == (uint32_t)input->dimensions[1] && group > 1u &&
                  output->dimensions[1] == input->dimensions[1] &&
                  (input->dimensions[1] % LW_NHWC_DEPTHWISE_BLOCK) == 0u &&
                  weight->dimensions[1] == 1;
    if ((!is_depthwise && group != 1u) || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || dh != 1 || dw != 1 ||
        pt < 0 || pl < 0 || pt != pb || pl != pr || !fast_tensor_eligible(input) ||
        !fast_tensor_eligible(output) || weight->dtype != LW_DTYPE_F32 || weight->rank != 4u ||
        (weight->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u ||
        weight->dimensions[0] != output->dimensions[1] ||
        (is_depthwise ? weight->dimensions[1] != 1 : weight->dimensions[1] != input->dimensions[1]) ||
        weight->dimensions[2] != kh ||
        weight->dimensions[3] != kw || output->dimensions[1] < 16 ||
        ((uint32_t)output->dimensions[1] % LW_NHWC_OC_BLOCK) != 0u) return 0;
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
        desc.pad_top = conv->pad_top; desc.pad_left = conv->pad_left; desc.dense_kc = conv->dense_kc;
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
static lw_status fast_allocate_tensor_twins(lw_x64_rec_fast_plan* plan, lw_error* error) {
    size_t total = 0u;
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
        const lw_runtime_tensor* tensor = &plan->session->tensors[index];
        lw_x64_fast_tensor_state* state = &plan->tensors[index];
        state->nhwc_offset = LW_X64_FAST_OFFSET_NONE;
        state->neutral = (uint8_t)lw_layout_tensor_is_neutral(plan->session, index);
        if (!fast_tensor_eligible(tensor) || state->neutral != 0u) {
            state->available_layouts = state->neutral != 0u
                ? (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC)
                : LW_X64_FAST_HAVE_NCHW;
            continue;
        }
        total = fast_align64(total);
        if (tensor->byte_size > SIZE_MAX - total) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast NHWC workspace size overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        state->nhwc_offset = (uint64_t)total;
        total += (size_t)tensor->byte_size;
        state->available_layouts = LW_X64_FAST_HAVE_NCHW;
    }
    plan->nhwc_workspace_bytes = fast_align64(total);
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
    status = fast_allocate_tensor_twins(plan, error);
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
        if (fast_prepare_elementwise(plan, node_index, &plan->nodes[node_index]) ||
            fast_prepare_conv(plan, node_index, &plan->nodes[node_index])) {
            ++plan->fast_node_count;
        }
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
            free(plan->nodes[index].data.conv.packed_weights);
        }
    }
    free(plan->scratch);
    free(plan->nhwc_workspace);
    free(plan->tensors);
    free(plan->nodes);
    lw_layout_plan_free(&plan->layout);
    free(plan);
}
