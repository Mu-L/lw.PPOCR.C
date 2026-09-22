#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"
#include "packed_matmul_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static lw_status make_shape_session(const lw_model* model, uint32_t width, lw_session** out,
                                    lw_error* error) {
    lw_tensor_desc input;
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = 1;
    input.dimensions[1] = 3;
    input.dimensions[2] = 48;
    input.dimensions[3] = (int32_t)width;
    return lw_session_create(model, &input, 1u, NULL, out, error);
}

static const uint8_t* node_bytes(const lw_model* model, uint32_t index) {
    return model->bytes + (size_t)model->node_offset + (size_t)index * LWM_V0_NODE_SIZE;
}

static const float* constant_f32(const lw_model* model, uint32_t index) {
    const uint8_t* tensor = model->bytes + (size_t)model->tensor_offset +
                            (size_t)index * LWM_V0_TENSOR_SIZE;
    return (const float*)(const void*)(model->bytes + (size_t)lwm_read_u64(tensor + 48u));
}

static float read_f32(const uint8_t* p) {
    uint32_t bits = lwm_read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void fill_value(const lw_session* session, uint32_t i, lw_x64_rec_value* value,
                       lw_x64_rec_compile_strategy strategy) {
    const lw_runtime_tensor* tensor = &session->tensors[i];
    uint32_t j;
    memset(value, 0, sizeof(*value));
    value->bytes = tensor->byte_size;
    value->rank = tensor->rank > 4u ? 4u : tensor->rank;
    if (tensor->rank == 4u) {
        if (strategy == LW_X64_REC_COMPILE_NCHW) {
            value->dimensions[0] = tensor->dimensions[0];
            value->dimensions[1] = tensor->dimensions[1];
            value->dimensions[2] = tensor->dimensions[2];
            value->dimensions[3] = tensor->dimensions[3];
            value->layout = LW_X64_REC_LAYOUT_NCHW;
        } else {
            value->dimensions[0] = tensor->dimensions[0];
            value->dimensions[1] = tensor->dimensions[2];
            value->dimensions[2] = tensor->dimensions[3];
            value->dimensions[3] = tensor->dimensions[1];
            value->layout = LW_X64_REC_LAYOUT_NHWC;
        }
    } else {
        for (j = 0u; j < value->rank; ++j) value->dimensions[j] = tensor->dimensions[j];
        value->layout = LW_X64_REC_LAYOUT_SCALAR;
    }
    value->producer = -1;
    value->last_use = -1;
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) value->constant_data = constant_f32(session->model, i);
}

static int add_constant(lw_x64_rec_program* program, void* data, uint64_t bytes) {
    lw_x64_rec_constant* next;
    if (data == NULL || bytes == 0u) return 0;
    next = (lw_x64_rec_constant*)realloc(program->constants,
        (size_t)(program->packed_constant_count + 1u) * sizeof(*next));
    if (next == NULL) return 0;
    program->constants = next;
    program->constants[program->packed_constant_count].data = data;
    program->constants[program->packed_constant_count].bytes = bytes;
    ++program->packed_constant_count;
    return 1;
}

static void* alloc_constant(lw_x64_rec_program* program, uint64_t bytes) {
    void* data;
    if (bytes == 0u || bytes > SIZE_MAX) return NULL;
    data = malloc((size_t)bytes);
    if (data == NULL || !add_constant(program, data, bytes)) {
        free(data);
        return NULL;
    }
    return data;
}

static int detect_ctc(const lw_model* model, const lw_session* session, lw_x64_rec_ctc_tail* tail) {
    uint32_t n = model->info.node_count;
    const uint8_t* matmul;
    const uint8_t* add;
    const uint8_t* softmax;
    uint32_t activation;
    uint32_t weights;
    uint32_t output;
    uint32_t add_output;
    if (n < 3u) return 0;
    matmul = node_bytes(model, n - 3u);
    add = node_bytes(model, n - 2u);
    softmax = node_bytes(model, n - 1u);
    if (lwm_read_u16(matmul) != LW_OP_MATMUL || lwm_read_u16(add) != LW_OP_ADD ||
        lwm_read_u16(softmax) != LW_OP_SOFTMAX || lwm_read_u16(matmul + 2u) != 2u ||
        lwm_read_u16(add + 2u) != 2u || lwm_read_u16(softmax + 2u) != 1u) return 0;
    activation = lwm_read_u32(matmul + 8u);
    weights = lwm_read_u32(matmul + 12u);
    output = lwm_read_u32(matmul + 40u);
    add_output = lwm_read_u32(add + 40u);
    if (lwm_read_u32(add + 8u) != output || session->tensors[weights].rank != 2u ||
        add_output != lwm_read_u32(softmax + 8u)) return 0;
    tail->enabled = 1u;
    tail->activation_value = activation;
    tail->weight_tensor = weights;
    tail->bias_tensor = lwm_read_u32(add + 12u);
    tail->classes = (uint32_t)session->tensors[weights].dimensions[1];
    tail->inner = (uint32_t)session->tensors[weights].dimensions[0];
    tail->rows = (uint32_t)session->tensors[activation].dimensions[session->tensors[activation].rank - 2u];
    return 1;
}

static int value_is_constant(const lw_x64_rec_value* value) { return value->constant_data != NULL; }
static uint64_t value_elements(const lw_x64_rec_value* value) { return value->bytes / sizeof(float); }

static lw_x64_rec_op* push_op(lw_x64_rec_program* program, uint32_t* count, uint16_t kind,
                              uint32_t semantic_begin, uint16_t semantic_count) {
    lw_x64_rec_op* op = &program->ops[*count];
    memset(op, 0, sizeof(*op));
    op->kind = kind;
    op->semantic_begin = semantic_begin;
    op->semantic_count = semantic_count;
    ++*count;
    return op;
}

static uint8_t choose_pointwise_kernel(uint32_t pixels, uint32_t output_channels) {
    if (pixels <= 1u && output_channels % 32u == 0u) return LW_X64_REC_PW_2X32;
    if (pixels <= 1u && output_channels % 16u == 0u) return LW_X64_REC_PW_4X16;
    return LW_X64_REC_PW_6X16;
}

static int fused_gelu_temporaries_private(const lw_model* model, uint32_t node_index,
                                           const lw_fused_gelu_match* match) {
    uint32_t temporary;
    if (model == NULL || match == NULL || node_index > model->info.node_count ||
        model->info.node_count - node_index < 5u) {
        return 0;
    }
    for (temporary = 0u; temporary < 4u; ++temporary) {
        uint32_t tensor = match->outputs[temporary];
        uint32_t node;
        int32_t last_use = -1;
        for (node = 0u; node < model->info.node_count; ++node) {
            const uint8_t* current = node_bytes(model, node);
            uint16_t input_count = lwm_read_u16(current + 2u);
            uint16_t input_index;
            for (input_index = 0u; input_index < input_count; ++input_index) {
                if (lwm_read_u32(current + 8u + (size_t)input_index * sizeof(uint32_t)) == tensor) {
                    last_use = (int32_t)node;
                }
            }
        }
        for (node = 0u; node < model->info.output_count; ++node) {
            if (lwm_read_u32(model->bytes + (size_t)model->output_offset +
                             (size_t)node * sizeof(uint32_t)) == tensor) {
                last_use = (int32_t)model->info.node_count;
            }
        }
        if (last_use != (int32_t)(node_index + temporary + 1u)) {
            return 0;
        }
    }
    return 1;
}

static lw_status compile_conv(const lw_model* model, const lw_session* session,
                              lw_x64_rec_program* program, const uint8_t* node,
                              lw_x64_rec_op* op, lw_error* error) {
    const lw_runtime_tensor* input = &session->tensors[lwm_read_u32(node + 8u)];
    const lw_runtime_tensor* weight = &session->tensors[lwm_read_u32(node + 12u)];
    const lw_runtime_tensor* output = &session->tensors[lwm_read_u32(node + 40u)];
    const uint8_t* params = model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint32_t groups = lwm_read_u32(params + 4u);
    uint32_t kh = (uint32_t)lwm_read_i32(params + 8u);
    uint32_t kw = (uint32_t)lwm_read_i32(params + 12u);
    uint32_t sh = (uint32_t)lwm_read_i32(params + 16u);
    uint32_t sw = (uint32_t)lwm_read_i32(params + 20u);
    uint64_t count;
    float* packed;
    op->data.conv.input_channels = (uint32_t)input->dimensions[1];
    op->data.conv.input_offset = program->values[lwm_read_u32(node + 8u)].offset;
    op->data.conv.output_offset = program->values[lwm_read_u32(node + 40u)].offset;
    op->data.conv.output_channels = (uint32_t)output->dimensions[1];
    op->data.conv.input_height = (uint32_t)input->dimensions[2];
    op->data.conv.input_width = (uint32_t)input->dimensions[3];
    op->data.conv.output_height = (uint32_t)output->dimensions[2];
    op->data.conv.output_width = (uint32_t)output->dimensions[3];
    op->data.conv.kernel_h = kh; op->data.conv.kernel_w = kw;
    op->data.conv.weight_h = (uint32_t)weight->dimensions[2]; op->data.conv.weight_w = (uint32_t)weight->dimensions[3];
    op->data.conv.stride_h = sh; op->data.conv.stride_w = sw;
    op->data.conv.pad_top = (uint32_t)lwm_read_i32(params + 32u);
    op->data.conv.pad_left = (uint32_t)lwm_read_i32(params + 36u);
    op->data.conv.pad_bottom = (uint32_t)lwm_read_i32(params + 40u);
    op->data.conv.pad_right = (uint32_t)lwm_read_i32(params + 44u);
    op->data.conv.groups = groups;
    if (program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
        const uint32_t weight_index = lwm_read_u32(node + 12u);
        const float* weights = constant_f32(model, weight_index);
        if (weights == NULL) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "NCHW Conv weights are not constant");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        op->data.conv.original_weights = weights;
        if (lwm_read_u16(node + 2u) >= 3u) {
            op->data.conv.bias = constant_f32(model, lwm_read_u32(node + 16u));
        }
        if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u) {
            if (!lw_packed_conv1x1_weight_count(op->data.conv.input_channels,
                                                op->data.conv.output_channels, &count)) {
                lw_set_error(error, LW_STATUS_UNSUPPORTED, "NCHW pointwise shape is unsupported");
                return LW_STATUS_UNSUPPORTED;
            }
            packed = (float*)alloc_constant(program, count * sizeof(float));
            if (packed == NULL) {
                lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "NCHW pointwise pack allocation failed");
                return LW_STATUS_OUT_OF_MEMORY;
            }
            lw_pack_conv1x1_weights_f32(weights, op->data.conv.input_channels,
                                        op->data.conv.output_channels, packed);
            op->kind = LW_X64_REC_OP_POINTWISE_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_POINTWISE;
            op->data.conv.packed_weights = packed;
            return LW_STATUS_OK;
        }
        if (groups == 1u && kh == 3u && kw == 3u && sh == 2u && sw == 2u &&
            op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u &&
            lw_packed_conv3x3_stride2_weight_count(op->data.conv.input_channels,
                                                   op->data.conv.output_channels, &count)) {
            packed = (float*)alloc_constant(program, count * sizeof(float));
            if (packed == NULL) {
                lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "NCHW stem pack allocation failed");
                return LW_STATUS_OUT_OF_MEMORY;
            }
            lw_pack_conv3x3_stride2_weights_f32(weights, op->data.conv.input_channels,
                                                op->data.conv.output_channels, packed);
            op->kind = LW_X64_REC_OP_STEM_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_DENSE;
            op->data.conv.packed_weights = packed;
            return LW_STATUS_OK;
        }
        if (groups == op->data.conv.input_channels &&
            groups == op->data.conv.output_channels &&
            weight->dimensions[1] == 1u &&
            ((kh == 3u && kw == 3u &&
              op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
              op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u &&
              ((sh == 1u && sw == 1u) || (sh == 2u && sw == 1u))) ||
             (kh == 1u && kw == 5u && sh == 1u && sw == 1u &&
              op->data.conv.pad_top == 0u && op->data.conv.pad_left == 2u &&
              op->data.conv.pad_bottom == 0u && op->data.conv.pad_right == 2u))) {
            op->kind = LW_X64_REC_OP_DEPTHWISE_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_DEPTHWISE;
            return LW_STATUS_OK;
        }
        char message[192]; (void)snprintf(message, sizeof(message), "Tiny NCHW Conv shape is unsupported: groups=%u in=%u out=%u kernel=%ux%u stride=%ux%u pad=%u,%u,%u,%u", groups, op->data.conv.input_channels, op->data.conv.output_channels, kh, kw, sh, sw, op->data.conv.pad_top, op->data.conv.pad_left, op->data.conv.pad_bottom, op->data.conv.pad_right); lw_set_error(error, LW_STATUS_UNSUPPORTED, message);
        return LW_STATUS_UNSUPPORTED;
    }
    if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u &&
        lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                          op->data.conv.output_channels, 1u, 1u, &count)) {
        op->kind = LW_X64_REC_OP_POINTWISE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_POINTWISE;
        op->data.conv.pointwise_kernel = choose_pointwise_kernel(
            op->data.conv.input_height * op->data.conv.input_width,
            op->data.conv.output_channels);
    } else if (groups == op->data.conv.input_channels && groups == op->data.conv.output_channels &&
               weight->dimensions[1] == 1 &&
               lw_nhwc_depthwise_packed_weight_count(op->data.conv.input_channels, kh, kw, &count)) {
        op->kind = LW_X64_REC_OP_DEPTHWISE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_DEPTHWISE;
    } else if (groups == 1u &&
               lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                                 op->data.conv.output_channels, kh, kw, &count)) {
        op->kind = LW_X64_REC_OP_DENSE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_DENSE;
    } else {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "Tiny REC Conv shape is not NHWC-lowerable");
        return LW_STATUS_UNSUPPORTED;
    }
    packed = (float*)alloc_constant(program, count * sizeof(float));
    if (packed == NULL) { lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "Conv packed weight allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    if (op->data.conv.kernel_kind == LW_X64_REC_CONV_DEPTHWISE)
        lw_pack_nhwc_depthwise_f32(constant_f32(model, lwm_read_u32(node + 12u)), op->data.conv.input_channels, kh, kw, packed);
    else
        lw_pack_nhwc_dense_f32(constant_f32(model, lwm_read_u32(node + 12u)), op->data.conv.input_channels,
                               op->data.conv.output_channels, kh, kw, packed);
    op->data.conv.packed_weights = packed;
    op->data.conv.original_weights = constant_f32(model, lwm_read_u32(node + 12u));
    if (lwm_read_u16(node + 2u) >= 3u) op->data.conv.bias = constant_f32(model, lwm_read_u32(node + 16u));
    op->data.conv.scratch_bytes = 0u;
    if (op->kind == LW_X64_REC_OP_DENSE) {
        op->data.conv.dense_kc = LW_NHWC_DENSE_KC;
        lw_nhwc_dense_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u; desc.input_channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height; desc.input_width = op->data.conv.input_width;
        desc.output_channels = op->data.conv.output_channels; desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width; desc.kernel_h = kh; desc.kernel_w = kw;
        desc.stride_h = sh; desc.stride_w = sw; desc.pad_top = op->data.conv.pad_top; desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom; desc.pad_right = op->data.conv.pad_right;
        desc.dense_kc = op->data.conv.dense_kc;
        if (!lw_nhwc_dense_scratch_bytes(&desc, &op->data.conv.scratch_bytes)) {
            op->data.conv.scalar_fallback = 1u;
            op->data.conv.scratch_bytes = 0u;
        }
        if (op->data.conv.scratch_bytes > program->scratch_bytes) program->scratch_bytes = op->data.conv.scratch_bytes;
    }
    return LW_STATUS_OK;
}

static lw_status compile_node(const lw_model* model, const lw_session* session,
                              lw_x64_rec_program* program, uint32_t node_index,
                              lw_x64_rec_op* op, lw_error* error) {
    const uint8_t* node = node_bytes(model, node_index);
    const uint16_t semantic = lwm_read_u16(node);
    uint32_t input_index = lwm_read_u32(node + 8u);
    uint32_t output_index = lwm_read_u32(node + 40u);
    const lw_runtime_tensor* input = &session->tensors[input_index];
    const lw_runtime_tensor* output = &session->tensors[output_index];
    const lw_x64_rec_value* in_value = &program->values[input_index];
    lw_x64_rec_value* out_value = &program->values[output_index];
    const uint8_t* params = lwm_read_u64(node + 56u) == 0u ? NULL : model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint64_t elements = value_elements(out_value);
    op->semantic_begin = node_index; op->semantic_count = 1u;
    switch (semantic) {
    case LW_OP_CONV: return compile_conv(model, session, program, node, op, error);
    case LW_OP_RELU: op->kind = LW_X64_REC_OP_RELU; break;
    case LW_OP_ERF: op->kind = LW_X64_REC_OP_ERF; break;
    case LW_OP_HARD_SIGMOID: op->kind = LW_X64_REC_OP_HARD_SIGMOID; op->data.unary.alpha = read_f32(params + 4u); op->data.unary.beta = read_f32(params + 8u); break;
    case LW_OP_SIGMOID: op->kind = LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    case LW_OP_SQRT: op->kind = LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    case LW_OP_BATCH_NORMALIZATION: {
        uint32_t c = (uint32_t)input->dimensions[1];
        float* mul = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
        float* add = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
        uint32_t j;
        if (mul == NULL || add == NULL) return LW_STATUS_OUT_OF_MEMORY;
        for (j = 0u; j < c; ++j) { float m = constant_f32(model,lwm_read_u32(node+12u))[j]; float b = constant_f32(model,lwm_read_u32(node+16u))[j]; float mean=constant_f32(model,lwm_read_u32(node+20u))[j]; float var=constant_f32(model,lwm_read_u32(node+24u))[j]; float scale=m/sqrtf(var+read_f32(params+4u)); mul[j]=scale; add[j]=b-mean*scale; }
        op->kind = LW_X64_REC_OP_AFFINE; op->data.affine.input_offset=in_value->offset; op->data.affine.output_offset=out_value->offset; op->data.affine.mul=mul; op->data.affine.add=add; op->data.affine.scale=constant_f32(model,lwm_read_u32(node+12u)); op->data.affine.bias=constant_f32(model,lwm_read_u32(node+16u)); op->data.affine.mean=constant_f32(model,lwm_read_u32(node+20u)); op->data.affine.variance=constant_f32(model,lwm_read_u32(node+24u)); op->data.affine.epsilon=read_f32(params+4u); op->data.affine.channel_major=(uint8_t)(program->backend_layout == LW_X64_REC_BACKEND_NCHW || input->rank != 4u); op->data.affine.pixels=(uint32_t)(value_elements(in_value)/c); op->data.affine.channels=c; break;
    }
    case LW_OP_ADD: case LW_OP_MUL: case LW_OP_DIV: case LW_OP_SUB: case LW_OP_POW:
        op->kind = semantic == LW_OP_ADD ? LW_X64_REC_OP_ADD : semantic == LW_OP_MUL ? LW_X64_REC_OP_MUL : semantic == LW_OP_DIV ? LW_X64_REC_OP_DIV : LW_X64_REC_OP_GENERIC_UNSUPPORTED; op->data.binary.operation=semantic; op->data.binary.element_count=elements; op->data.binary.output_offset=out_value->offset; op->data.binary.left_offset=program->values[input_index].offset; op->data.binary.left_constant=program->values[input_index].constant_data; op->data.binary.right_offset=program->values[lwm_read_u32(node+12u)].offset; op->data.binary.right_constant=program->values[lwm_read_u32(node+12u)].constant_data; op->data.binary.broadcast_kind=(program->values[lwm_read_u32(node+12u)].bytes==sizeof(float))?LW_X64_REC_BROADCAST_RIGHT_SCALAR:program->values[lwm_read_u32(node+12u)].bytes==out_value->bytes?LW_X64_REC_BROADCAST_SAME:((program->values[lwm_read_u32(node+12u)].bytes == (uint64_t)out_value->dimensions[3] * sizeof(float))?LW_X64_REC_BROADCAST_RIGHT_CHANNEL:((program->values[input_index].bytes == (uint64_t)out_value->dimensions[3] * sizeof(float))?LW_X64_REC_BROADCAST_LEFT_CHANNEL:LW_X64_REC_BROADCAST_GENERAL)); memcpy(op->data.binary.dimensions, out_value->dimensions, sizeof(op->data.binary.dimensions)); op->data.binary.channels = out_value->rank > 0u ? (uint32_t)out_value->dimensions[out_value->rank - 1u] : 1u; op->data.binary.pixels = op->data.binary.channels != 0u ? (uint32_t)(elements / op->data.binary.channels) : 0u;
        if (program->backend_layout == LW_X64_REC_BACKEND_NCHW && out_value->rank == 4u) {
            op->data.binary.channels = (uint32_t)out_value->dimensions[1];
            op->data.binary.pixels = op->data.binary.channels != 0u ?
                (uint32_t)(elements / op->data.binary.channels) : 0u;
            if (program->values[lwm_read_u32(node + 12u)].bytes ==
                (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_REC_BROADCAST_RIGHT_CHANNEL;
            } else if (program->values[input_index].bytes ==
                       (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_REC_BROADCAST_LEFT_CHANNEL;
            }
        }
        break;
    case LW_OP_REDUCE_MEAN: op->kind=LW_X64_REC_OP_REDUCE_MEAN; op->data.reduce.input_offset=in_value->offset; op->data.reduce.output_offset=out_value->offset; op->data.reduce.batch=(uint32_t)input->dimensions[0]; op->data.reduce.height=(uint32_t)input->dimensions[2]; op->data.reduce.width=(uint32_t)input->dimensions[3]; op->data.reduce.channels=(uint32_t)input->dimensions[1]; break;
    case LW_OP_AVERAGE_POOL: case LW_OP_MAX_POOL: op->kind=semantic==LW_OP_MAX_POOL?LW_X64_REC_OP_MAX_POOL:LW_X64_REC_OP_AVG_POOL; op->data.pool.input_offset=in_value->offset; op->data.pool.output_offset=out_value->offset; op->data.pool.input_dimensions[0]=input->dimensions[0];
        op->data.pool.input_dimensions[1]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[1] : input->dimensions[2];
        op->data.pool.input_dimensions[2]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[2] : input->dimensions[3];
        op->data.pool.input_dimensions[3]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[3] : input->dimensions[1];
        op->data.pool.output_dimensions[0]=output->dimensions[0];
        op->data.pool.output_dimensions[1]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[1] : output->dimensions[2];
        op->data.pool.output_dimensions[2]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[2] : output->dimensions[3];
        op->data.pool.output_dimensions[3]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[3] : output->dimensions[1]; op->data.pool.kernel[0]=lwm_read_i32(params+8u); op->data.pool.kernel[1]=lwm_read_i32(params+12u); op->data.pool.strides[0]=lwm_read_i32(params+16u); op->data.pool.strides[1]=lwm_read_i32(params+20u); op->data.pool.pads[0]=lwm_read_i32(params+24u); op->data.pool.pads[1]=lwm_read_i32(params+28u); op->data.pool.pads[2]=lwm_read_i32(params+32u); op->data.pool.pads[3]=lwm_read_i32(params+36u); op->data.pool.count_include_pad=(uint8_t)lwm_read_u32(params+44u); op->data.pool.is_max=(uint8_t)(semantic==LW_OP_MAX_POOL); break;
    case LW_OP_MATMUL: { const lw_runtime_tensor* w=&session->tensors[lwm_read_u32(node+12u)]; uint64_t count; op->kind=LW_X64_REC_OP_MATMUL; op->data.matmul.input_offset=in_value->offset; op->data.matmul.output_offset=out_value->offset; op->data.matmul.weights=constant_f32(model,lwm_read_u32(node+12u)); op->data.matmul.batch=1u; op->data.matmul.rows=(uint32_t)input->dimensions[input->rank-2u]; op->data.matmul.inner=(uint32_t)input->dimensions[input->rank-1u]; op->data.matmul.columns=(uint32_t)w->dimensions[1]; if(lw_packed_matmul_weight_count(op->data.matmul.inner,op->data.matmul.columns,&count)){op->data.matmul.packed_weights=(float*)alloc_constant(program,count*sizeof(float)); if(op->data.matmul.packed_weights==NULL)return LW_STATUS_OUT_OF_MEMORY; lw_pack_matmul_weights_f32(op->data.matmul.weights,op->data.matmul.inner,op->data.matmul.columns,op->data.matmul.packed_weights);} break; }
    case LW_OP_TRANSPOSE: op->kind=LW_X64_REC_OP_TRANSPOSE; op->data.transpose.input_offset=in_value->offset; op->data.transpose.output_offset=out_value->offset; op->data.transpose.rank=input->rank; op->data.transpose.input_dimensions[0]=input->dimensions[0]; op->data.transpose.input_dimensions[1]=input->dimensions[1]; op->data.transpose.input_dimensions[2]=input->dimensions[2]; op->data.transpose.input_dimensions[3]=input->dimensions[3]; op->data.transpose.output_dimensions[0]=output->dimensions[0]; op->data.transpose.output_dimensions[1]=output->dimensions[1]; op->data.transpose.output_dimensions[2]=output->dimensions[2]; op->data.transpose.output_dimensions[3]=output->dimensions[3]; for(uint32_t j=0;j<input->rank;++j)op->data.transpose.permutation[j]=lwm_read_i32(params+4u+j*4u); break;
    case LW_OP_CONCAT: case LW_OP_RESIZE: op->kind=LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    default: op->kind=LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    }
    if (op->kind == LW_X64_REC_OP_RELU || op->kind == LW_X64_REC_OP_ERF || op->kind == LW_X64_REC_OP_HARD_SIGMOID) { op->data.unary.input_offset=in_value->offset; op->data.unary.output_offset=out_value->offset; op->data.unary.element_count=elements; op->data.unary.rank=out_value->rank; memcpy(op->data.unary.dimensions,out_value->dimensions,sizeof(op->data.unary.dimensions)); }
    return LW_STATUS_OK;
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile_ex(const lw_model* model, uint32_t target_width,
                                                      lw_x64_rec_compile_strategy strategy,
                                                      lw_x64_rec_program** out_program, lw_error* error) {
    lw_session* session=NULL; lw_x64_rec_program* program=NULL; lw_model_info info; lw_status status; uint64_t cursor=0u; uint32_t i, physical=0u, limit;
    if(out_program==NULL||model==NULL||target_width==0u||target_width>INT32_MAX){lw_set_error(error,LW_STATUS_INVALID_ARGUMENT,"REC backend arguments are invalid");return LW_X64_REC_COMPILE_INVALID_GRAPH;} *out_program=NULL; lw_model_info_init(&info); status=lw_model_get_info(model,&info); if(status!=LW_STATUS_OK)return LW_X64_REC_COMPILE_INVALID_GRAPH; status=make_shape_session(model,target_width,&session,error); if(status!=LW_STATUS_OK)return status==LW_STATUS_OUT_OF_MEMORY?LW_X64_REC_COMPILE_OUT_OF_MEMORY:LW_X64_REC_COMPILE_INVALID_GRAPH; program=(lw_x64_rec_program*)calloc(1u,sizeof(*program)); if(program==NULL){lw_session_free(session);lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"REC program allocation failed");return LW_X64_REC_COMPILE_OUT_OF_MEMORY;} program->model=model;program->cpu=lw_get_cpu_capabilities();program->model_signature=info.content_checksum;program->target_width=target_width;program->backend_layout=(uint8_t)strategy;program->value_count=info.tensor_count;program->direct_nhwc=(uint8_t)(strategy == LW_X64_REC_COMPILE_NHWC); if(!lw_simd_level_is_avx2(program->cpu.simd)||!program->cpu.has_avx2_fma){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_UNSUPPORTED,"standalone x64 REC backend requires AVX2 and FMA");return LW_X64_REC_COMPILE_UNSUPPORTED;} program->ctc_fused=detect_ctc(model,session,&program->ctc)?1u:0u; limit=program->ctc_fused?info.node_count-3u:info.node_count; program->ops=(lw_x64_rec_op*)calloc(limit,sizeof(*program->ops));program->values=(lw_x64_rec_value*)calloc(program->value_count,sizeof(*program->values));if(program->ops==NULL||program->values==NULL){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"REC program tables allocation failed");return LW_X64_REC_COMPILE_OUT_OF_MEMORY;}
    for(i=0u;i<program->value_count;++i){fill_value(session,i,&program->values[i],strategy);if((session->tensors[i].flags&LWM_V0_TENSOR_FLAG_CONSTANT)==0u){status=lw_x64_rec_arena_alloc(&cursor,session->tensors[i].byte_size,64u,&program->values[i].offset,error);if(status!=LW_STATUS_OK){lw_session_free(session);lw_x64_rec_program_free(program);return LW_X64_REC_COMPILE_INVALID_GRAPH;}if((session->tensors[i].flags&LWM_V0_TENSOR_FLAG_INPUT)!=0u)program->input_value=i;}}
    for(i=0u;i<limit;++i){const uint8_t* node=node_bytes(model,i);uint16_t semantic=lwm_read_u16(node);
    {
        lw_fused_gelu_match match;
        if (i + 5u <= limit && lw_match_fused_gelu(session, i, &match) &&
            fused_gelu_temporaries_private(model, i, &match) && physical < limit) {
            lw_x64_rec_op* op = &program->ops[physical];
            const lw_x64_rec_value* source = &program->values[match.inputs[0][0]];
            const lw_x64_rec_value* output = &program->values[match.outputs[4]];
            memset(op, 0, sizeof(*op));
            op->kind = LW_X64_REC_OP_GELU;
            op->semantic_begin = i;
            op->semantic_count = 5u;
            op->data.unary.input_offset = source->offset;
            op->data.unary.output_offset = output->offset;
            op->data.unary.element_count = output->bytes / sizeof(float);
            op->data.unary.rank = output->rank;
            memcpy(op->data.unary.dimensions, output->dimensions,
                   sizeof(op->data.unary.dimensions));
            program->values[match.outputs[4]].producer = (int32_t)physical;
            for (uint32_t fused_node = i; fused_node < i + 5u; ++fused_node) {
                const uint8_t* fused_bytes = node_bytes(model, fused_node);
                uint16_t fused_inputs = lwm_read_u16(fused_bytes + 2u);
                for (uint16_t input_index = 0u; input_index < fused_inputs; ++input_index) {
                    uint32_t input = lwm_read_u32(
                        fused_bytes + 8u + (size_t)input_index * sizeof(uint32_t));
                    if (program->values[input].last_use < (int32_t)physical) {
                        program->values[input].last_use = (int32_t)physical;
                    }
                }
            }
            ++physical;
            program->semantic_consumed += 5u;
            i += 4u;
            continue;
        }
    }
    if (program->backend_layout == LW_X64_REC_BACKEND_NCHW &&
        (semantic == LW_OP_RESHAPE || semantic == LW_OP_SQUEEZE ||
         semantic == LW_OP_UNSQUEEZE)) {
        uint32_t in = lwm_read_u32(node + 8u);
        uint32_t out = lwm_read_u32(node + 40u);
        program->values[out].alias = 1u;
        program->values[out].alias_of = in;
        program->values[out].offset = program->values[in].offset;
        ++program->semantic_elided;
        program->values[out].producer = (int32_t)i;
        continue;
    }
    if(semantic==LW_OP_RESHAPE||semantic==LW_OP_SQUEEZE||semantic==LW_OP_UNSQUEEZE){uint32_t in=lwm_read_u32(node+8u),out=lwm_read_u32(node+40u); const lw_runtime_tensor* in_tensor=&session->tensors[in]; const lw_runtime_tensor* out_tensor=&session->tensors[out]; if(in_tensor->rank==3u && out_tensor->rank==4u && physical<limit){ lw_x64_rec_op* repack=&program->ops[physical++]; memset(repack,0,sizeof(*repack)); repack->kind=LW_X64_REC_OP_TRANSPOSE; repack->semantic_begin=i; repack->semantic_count=1u; repack->data.transpose.input_offset=program->values[in].offset; repack->data.transpose.output_offset=program->values[out].offset; repack->data.transpose.rank=4u; repack->data.transpose.input_dimensions[0]=in_tensor->dimensions[0]; repack->data.transpose.input_dimensions[1]=in_tensor->dimensions[1]; repack->data.transpose.input_dimensions[2]=out_tensor->dimensions[2]; repack->data.transpose.input_dimensions[3]=out_tensor->dimensions[3]; repack->data.transpose.output_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.output_dimensions[1]=out_tensor->dimensions[2]; repack->data.transpose.output_dimensions[2]=out_tensor->dimensions[3]; repack->data.transpose.output_dimensions[3]=out_tensor->dimensions[1]; repack->data.transpose.permutation[0]=0; repack->data.transpose.permutation[1]=2; repack->data.transpose.permutation[2]=3; repack->data.transpose.permutation[3]=1; program->values[out].producer=(int32_t)(physical-1u); program->semantic_consumed++; continue; } if(in_tensor->rank==4u && out_tensor->rank==3u && physical<limit){ lw_x64_rec_op* repack=&program->ops[physical++]; memset(repack,0,sizeof(*repack)); repack->kind=LW_X64_REC_OP_TRANSPOSE; repack->semantic_begin=i; repack->semantic_count=1u; repack->data.transpose.input_offset=program->values[in].offset; repack->data.transpose.output_offset=program->values[out].offset; repack->data.transpose.rank=4u; repack->data.transpose.input_dimensions[0]=in_tensor->dimensions[0]; repack->data.transpose.input_dimensions[1]=in_tensor->dimensions[2]; repack->data.transpose.input_dimensions[2]=in_tensor->dimensions[3]; repack->data.transpose.input_dimensions[3]=in_tensor->dimensions[1]; repack->data.transpose.output_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.output_dimensions[1]=in_tensor->dimensions[1]; repack->data.transpose.output_dimensions[2]=in_tensor->dimensions[2]; repack->data.transpose.output_dimensions[3]=in_tensor->dimensions[3]; repack->data.transpose.permutation[0]=0; repack->data.transpose.permutation[1]=3; repack->data.transpose.permutation[2]=1; repack->data.transpose.permutation[3]=2; program->values[out].producer=(int32_t)(physical-1u); program->semantic_consumed++; continue; } program->values[out].alias=1u;program->values[out].alias_of=in;program->values[out].offset=program->values[in].offset;++program->semantic_elided;program->values[out].producer=(int32_t)i;continue;} if(physical>=limit){lw_session_free(session);lw_x64_rec_program_free(program);return LW_X64_REC_COMPILE_INVALID_GRAPH;} {lw_x64_rec_op* op=&program->ops[physical];memset(op,0,sizeof(*op));status=compile_node(model,session,program,i,op,error);if(status!=LW_STATUS_OK||op->kind==LW_X64_REC_OP_GENERIC_UNSUPPORTED){++program->unsupported_nodes;if (status == LW_STATUS_OK || error == NULL || error->message[0] == 0) { char message[128]; (void)snprintf(message,sizeof(message),"REC node %u (op %u) lowering failed (status %d, kind %u)",(unsigned)i,(unsigned)semantic,(int)status,(unsigned)op->kind); lw_set_error(error,status!=LW_STATUS_OK?status:LW_STATUS_UNSUPPORTED,message); }lw_session_free(session);lw_x64_rec_program_free(program);return LW_X64_REC_COMPILE_UNSUPPORTED;}op->semantic_begin=i;op->semantic_count=1u;program->semantic_consumed++;program->values[lwm_read_u32(node+40u)].producer=(int32_t)physical;for(uint32_t j=0u;j<lwm_read_u16(node+2u);++j){uint32_t in=lwm_read_u32(node+8u+j*4u);if(program->values[in].last_use<(int32_t)physical)program->values[in].last_use=(int32_t)physical;}++physical;}}
    program->op_count=physical;program->semantic_consumed+=program->semantic_elided; if(program->ctc_fused){status=lw_x64_rec_ctc_prepare(model,session,&program->ctc,error);if(status!=LW_STATUS_OK){lw_session_free(session);lw_x64_rec_program_free(program);return status==LW_STATUS_OUT_OF_MEMORY?LW_X64_REC_COMPILE_OUT_OF_MEMORY:LW_X64_REC_COMPILE_INVALID_GRAPH;}}
    program->class_count=program->ctc.classes;program->time_steps=program->ctc.rows;program->output_value=info.output_count?lwm_read_u32(model->bytes+(size_t)model->output_offset):0u;program->arena_bytes=cursor; if(cursor==0u){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_INVALID_SHAPE,"REC graph has no runtime arena");return LW_X64_REC_COMPILE_INVALID_GRAPH;} lw_session_free(session);lw_set_error(error,LW_STATUS_OK,"");*out_program=program;return LW_X64_REC_COMPILE_OK;
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model, uint32_t target_width,
    lw_x64_rec_program** out_program, lw_error* error) {
    return lw_x64_rec_backend_compile_ex(model, target_width, LW_X64_REC_COMPILE_NHWC,
                                         out_program, error);
}

void lw_x64_rec_program_free(lw_x64_rec_program* program){uint32_t i;if(program==NULL)return;lw_x64_rec_ctc_free(&program->ctc);for(i=0u;i<program->packed_constant_count;++i)free(program->constants[i].data);free(program->constants);free(program->ops);free(program->values);free(program);}
