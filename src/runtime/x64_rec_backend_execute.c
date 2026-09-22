#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "operator_internal.h"
#include "../simd/simd_kernels.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_MSC_VER)
#include <malloc.h>
static void* rec_aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
static void rec_aligned_free(void* p) { _aligned_free(p); }
#else
static void* rec_aligned_alloc(size_t alignment, size_t size) {
    void* p = NULL;
    return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
}
static void rec_aligned_free(void* p) { free(p); }
#endif

static float* offset_ptr(lw_x64_rec_instance* instance, uint64_t offset) {
    if (instance == NULL || instance->arena == NULL || offset > SIZE_MAX) return NULL;
    return (float*)(void*)(instance->arena + (size_t)offset);
}

static lw_scalar_binary_op binary_operation(uint16_t operation) {
    switch (operation) {
    case LW_OP_ADD: return LW_SCALAR_BINARY_ADD;
    case LW_OP_MUL: return LW_SCALAR_BINARY_MUL;
    case LW_OP_DIV: return LW_SCALAR_BINARY_DIV;
    case LW_OP_SUB: return LW_SCALAR_BINARY_SUB;
    default: return LW_SCALAR_BINARY_POW;
    }
}

static void scalar_nhwc_conv(const lw_x64_rec_conv_op* conv, const float* input,
                             float* output) {
    uint32_t oy, ox, oc, ky, kx, ic;
    for (oy = 0u; oy < conv->output_height; ++oy) {
        for (ox = 0u; ox < conv->output_width; ++ox) {
            for (oc = 0u; oc < conv->output_channels; ++oc) {
                float sum = conv->bias == NULL ? 0.0f : conv->bias[oc];
                for (ky = 0u; ky < conv->kernel_h; ++ky) {
                    int32_t iy = (int32_t)(oy * conv->stride_h + ky) - (int32_t)conv->pad_top;
                    if (iy < 0 || iy >= (int32_t)conv->input_height) continue;
                    for (kx = 0u; kx < conv->kernel_w; ++kx) {
                        int32_t ix = (int32_t)(ox * conv->stride_w + kx) - (int32_t)conv->pad_left;
                        if (ix < 0 || ix >= (int32_t)conv->input_width) continue;
                        for (ic = 0u; ic < conv->input_channels; ++ic) {
                            size_t input_index = ((size_t)iy * conv->input_width + (size_t)ix) * conv->input_channels + ic;
                            size_t weight_index;
                            if (conv->groups == conv->input_channels) {
                                if (ic != oc) continue;
                                weight_index = (((size_t)oc * conv->kernel_h + ky) * conv->kernel_w) + kx;
                            } else {
                                weight_index = ((((size_t)oc * conv->input_channels + ic) * conv->kernel_h + ky) * conv->kernel_w) + kx;
                            }
                            sum += input[input_index] * conv->original_weights[weight_index];                        }
                    }
                }
                output[((size_t)oy * conv->output_width + ox) * conv->output_channels + oc] = sum;
            }
        }
    }
}
static void scalar_nhwc_batch_norm(const lw_x64_rec_affine_op* affine, const float* input, float* output) {
    uint32_t pixel;
    for (pixel = 0u; pixel < affine->pixels; ++pixel) {
        uint32_t channel;
        for (channel = 0u; channel < affine->channels; ++channel) {
            float factor = affine->scale[channel] / sqrtf(affine->variance[channel] + affine->epsilon);
            size_t index = affine->channel_major
                              ? (size_t)channel * affine->pixels + pixel
                              : (size_t)pixel * affine->channels + channel;
            float value = input[index];
            output[index] = (value - affine->mean[channel]) * factor + affine->bias[channel];
        }
    }
}

static lw_status execute_op(lw_x64_rec_instance* instance, const lw_x64_rec_op* op,
                            lw_error* error) {
    (void)error;
    float* input;
    float* output;
    if (op == NULL) return LW_STATUS_INVALID_ARGUMENT;
    switch (op->kind) {
    case LW_X64_REC_OP_POINTWISE: {
        lw_nhwc_epilogue ep = { NULL, NULL, op->data.conv.activation, 0u, 0.0f, 0.0f };
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        ep.bias = op->data.conv.bias;
        lw_avx2_fma_nhwc_pointwise_f32(input, op->data.conv.packed_weights, &ep, output,
                                       op->data.conv.input_height * op->data.conv.input_width,
                                       op->data.conv.input_channels, op->data.conv.output_channels);
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_DENSE: {
        lw_nhwc_dense_desc desc;
        lw_nhwc_epilogue ep;
        lw_status status;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.input_channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height;
        desc.input_width = op->data.conv.input_width;
        desc.output_channels = op->data.conv.output_channels;
        desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width;
        desc.kernel_h = op->data.conv.kernel_h;
        desc.kernel_w = op->data.conv.kernel_w;
        desc.stride_h = op->data.conv.stride_h;
        desc.stride_w = op->data.conv.stride_w;
        desc.pad_top = op->data.conv.pad_top;
        desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom;
        desc.pad_right = op->data.conv.pad_right;
        desc.dense_kc = op->data.conv.dense_kc;
        memset(&ep, 0, sizeof(ep));
        ep.bias = op->data.conv.bias;
        ep.activation = op->data.conv.activation;
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        status = lw_avx2_fma_nhwc_dense_f32(input, op->data.conv.packed_weights, &ep, output,
                                             &desc, instance->scratch, op->data.conv.scratch_bytes);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        scalar_nhwc_conv(&op->data.conv, input, output);
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_DEPTHWISE: {
        lw_nhwc_depthwise_desc desc;
        lw_status status;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height;
        desc.input_width = op->data.conv.input_width;
        desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width;
        desc.kernel_h = op->data.conv.kernel_h;
        desc.kernel_w = op->data.conv.kernel_w;
        desc.stride_h = op->data.conv.stride_h;
        desc.stride_w = op->data.conv.stride_w;
        desc.pad_top = op->data.conv.pad_top;
        desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom;
        desc.pad_right = op->data.conv.pad_right;
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        status = lw_avx2_fma_nhwc_depthwise_f32(input, op->data.conv.packed_weights,
                                                 op->data.conv.bias, output, &desc);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        scalar_nhwc_conv(&op->data.conv, input, output);
        return LW_STATUS_OK;
    }
    case LW_X64_REC_OP_AFFINE:
        input = offset_ptr(instance, op->data.affine.input_offset); output = offset_ptr(instance, op->data.affine.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        scalar_nhwc_batch_norm(&op->data.affine, input, output); return LW_STATUS_OK;
    case LW_X64_REC_OP_ADD: case LW_X64_REC_OP_MUL: case LW_X64_REC_OP_DIV:
        input = op->data.binary.left_constant != NULL ? (float*)(uintptr_t)op->data.binary.left_constant : offset_ptr(instance, op->data.binary.left_offset);
        output = offset_ptr(instance, op->data.binary.output_offset);
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_LEFT_CHANNEL) {
            float* full = offset_ptr(instance, op->data.binary.right_offset);
            float* channel = op->data.binary.left_constant != NULL ? (float*)(uintptr_t)op->data.binary.left_constant : offset_ptr(instance, op->data.binary.left_offset);
            if (full == NULL || channel == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), full, channel, output, op->data.binary.pixels, op->data.binary.channels, 1);
            return LW_STATUS_OK;
        }
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_SCALAR && op->data.binary.right_constant != NULL) {
            lw_avx2_binary_right_scalar_f32(binary_operation(op->data.binary.operation), input,
                                             op->data.binary.right_constant[0], output, op->data.binary.element_count);
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_SAME) {
            float* right = op->data.binary.right_constant != NULL ? (float*)(uintptr_t)op->data.binary.right_constant : offset_ptr(instance, op->data.binary.right_offset);
            if (right == NULL) return LW_STATUS_INVALID_ARGUMENT;
            lw_avx2_binary_contiguous_f32(binary_operation(op->data.binary.operation), input, right, output, op->data.binary.element_count);
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_RIGHT_CHANNEL) {
            float* channel = op->data.binary.right_constant != NULL ? (float*)(uintptr_t)op->data.binary.right_constant : offset_ptr(instance, op->data.binary.right_offset);
            if (channel == NULL) return LW_STATUS_INVALID_ARGUMENT;
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), input, channel, output, op->data.binary.pixels, op->data.binary.channels, 0);
            return LW_STATUS_OK;
        }
        if (op->data.binary.right_constant != NULL && op->data.binary.channels > 1u) {
            uint64_t index;
            const float* right = op->data.binary.right_constant;
            for (index = 0u; index < op->data.binary.element_count; ++index) {
                float lhs = input[index];
                float rhs = right[index % op->data.binary.channels];
                switch (op->data.binary.operation) {
                case LW_OP_ADD: output[index] = lhs + rhs; break;
                case LW_OP_MUL: output[index] = lhs * rhs; break;
                case LW_OP_DIV: output[index] = lhs / rhs; break;
                default: output[index] = lhs - rhs; break;
                }
            }
            return LW_STATUS_OK;
        }
        return LW_STATUS_UNSUPPORTED;
    case LW_X64_REC_OP_RELU: case LW_X64_REC_OP_ERF: case LW_X64_REC_OP_GELU: case LW_X64_REC_OP_HARD_SIGMOID:
        input = offset_ptr(instance, op->data.unary.input_offset); output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->kind == LW_X64_REC_OP_RELU) lw_avx2_relu_contiguous_f32(input, output, op->data.unary.element_count);
        else if (op->kind == LW_X64_REC_OP_ERF) lw_avx2_erf_f32(input, output, op->data.unary.element_count);
        else if (op->kind == LW_X64_REC_OP_GELU) lw_avx2_gelu_f32(input, output, op->data.unary.element_count);
        else lw_avx2_hard_sigmoid_contiguous_f32(input, output, op->data.unary.element_count, op->data.unary.alpha, op->data.unary.beta);
        return LW_STATUS_OK;
    case LW_X64_REC_OP_REDUCE_MEAN:
        input = offset_ptr(instance, op->data.reduce.input_offset); output = offset_ptr(instance, op->data.reduce.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        lw_avx2_nhwc_reduce_mean_hw_f32(input, output, op->data.reduce.batch, op->data.reduce.height, op->data.reduce.width, op->data.reduce.channels); return LW_STATUS_OK;
    case LW_X64_REC_OP_AVG_POOL: case LW_X64_REC_OP_MAX_POOL:
        input = offset_ptr(instance, op->data.pool.input_offset); output = offset_ptr(instance, op->data.pool.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        lw_avx2_nhwc_pool_f32(input, output, (uint32_t)op->data.pool.input_dimensions[0], (uint32_t)op->data.pool.input_dimensions[1], (uint32_t)op->data.pool.input_dimensions[2], (uint32_t)op->data.pool.output_dimensions[1], (uint32_t)op->data.pool.output_dimensions[2], (uint32_t)op->data.pool.input_dimensions[3], (uint32_t)op->data.pool.kernel[0], (uint32_t)op->data.pool.kernel[1], (uint32_t)op->data.pool.strides[0], (uint32_t)op->data.pool.strides[1], (uint32_t)op->data.pool.pads[0], (uint32_t)op->data.pool.pads[1], op->data.pool.count_include_pad, op->data.pool.is_max); return LW_STATUS_OK;
    case LW_X64_REC_OP_TRANSPOSE:
        input = offset_ptr(instance, op->data.transpose.input_offset); output = offset_ptr(instance, op->data.transpose.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_transpose_f32(input, output, op->data.transpose.rank, op->data.transpose.input_dimensions, op->data.transpose.rank, op->data.transpose.permutation, op->data.transpose.output_dimensions);
    case LW_X64_REC_OP_MATMUL:
        input = offset_ptr(instance, op->data.matmul.input_offset); output = offset_ptr(instance, op->data.matmul.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.matmul.packed_weights != NULL) lw_avx2_packed_matmul_shared_f32(input, op->data.matmul.packed_weights, output, op->data.matmul.batch, op->data.matmul.rows, op->data.matmul.inner, op->data.matmul.columns);
        else return lw_scalar_matmul_shared_f32(input, op->data.matmul.weights, output, op->data.matmul.batch, op->data.matmul.rows, op->data.matmul.inner, op->data.matmul.columns);
        return LW_STATUS_OK;
    default: return LW_STATUS_UNSUPPORTED;
    }
}

lw_status lw_x64_rec_instance_create(const lw_x64_rec_program* program, lw_x64_rec_instance** out, lw_error* error) {
    lw_x64_rec_instance* instance;
    if (out == NULL || program == NULL) { lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance requires a compiled program"); return LW_STATUS_INVALID_ARGUMENT; }
    *out = NULL; instance = (lw_x64_rec_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) { lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    instance->program = program;
    if (program->arena_bytes > 0u) instance->arena = (uint8_t*)rec_aligned_alloc(64u, (size_t)program->arena_bytes);
    if (program->scratch_bytes > 0u) instance->scratch = (uint8_t*)rec_aligned_alloc(64u, (size_t)program->scratch_bytes);
    if ((program->arena_bytes > 0u && instance->arena == NULL) || (program->scratch_bytes > 0u && instance->scratch == NULL)) { lw_x64_rec_instance_free(instance); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance workspace allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
    if (program->ctc.enabled) { uint64_t rows = program->ctc.rows, classes = program->ctc.classes; if (rows == 0u || classes == 0u || rows > SIZE_MAX / classes || rows * classes > SIZE_MAX / sizeof(float)) { lw_x64_rec_instance_free(instance); return LW_STATUS_INVALID_SHAPE; } instance->ctc_logits = (float*)rec_aligned_alloc(64u, (size_t)(rows * classes * sizeof(float))); instance->best_indices = (uint32_t*)rec_aligned_alloc(64u, (size_t)(rows * sizeof(uint32_t))); instance->best_probabilities = (float*)rec_aligned_alloc(64u, (size_t)(rows * sizeof(float))); if (instance->ctc_logits == NULL || instance->best_indices == NULL || instance->best_probabilities == NULL) { lw_x64_rec_instance_free(instance); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC CTC workspace allocation failed"); return LW_STATUS_OUT_OF_MEMORY; } }
    *out = instance; lw_set_error(error, LW_STATUS_OK, ""); return LW_STATUS_OK;
}

void lw_x64_rec_instance_free(lw_x64_rec_instance* instance) { if (instance == NULL) return; rec_aligned_free(instance->ctc_logits); rec_aligned_free(instance->best_indices); rec_aligned_free(instance->best_probabilities); rec_aligned_free(instance->scratch); rec_aligned_free(instance->arena); free(instance); }

float* lw_x64_rec_instance_input(lw_x64_rec_instance* instance, uint64_t* element_count) { const lw_x64_rec_value* value; if (element_count != NULL) *element_count = 0u; if (instance == NULL || instance->program == NULL || instance->program->input_value >= instance->program->value_count) return NULL; value=&instance->program->values[instance->program->input_value]; if (element_count != NULL) *element_count=value->bytes/sizeof(float); return offset_ptr(instance,value->offset); }

static lw_status run_backbone_ops(lw_x64_rec_instance* instance, lw_error* error) {
    uint32_t i;
    lw_status status;
    if (instance == NULL || instance->program == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0u; i < instance->program->op_count; ++i) {
        status = execute_op(instance, &instance->program->ops[i], error);
        if (status != LW_STATUS_OK) {
            if (error != NULL && error->message[0] == '\0') {
                char message[96];
                (void)snprintf(message, sizeof(message), "REC physical op %u kind %u failed", (unsigned)i, (unsigned)instance->program->ops[i].kind);
                lw_set_error(error, status, message);
            }
            return status;
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_x64_rec_instance_run_op(lw_x64_rec_instance* instance, uint32_t op_index, lw_error* error) {
    if (instance == NULL || instance->program == NULL || op_index >= instance->program->op_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC physical op index is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return execute_op(instance, &instance->program->ops[op_index], error);
}
lw_status lw_x64_rec_instance_run_backbone(lw_x64_rec_instance* instance, lw_error* error) {
    lw_status status = run_backbone_ops(instance, error);
    if (status == LW_STATUS_OK) lw_set_error(error, LW_STATUS_OK, "");
    return status;
}

lw_status lw_x64_rec_instance_run(lw_x64_rec_instance* instance, lw_error* error) {
    lw_status status;
    float* activation;
    status = run_backbone_ops(instance, error);
    if (status != LW_STATUS_OK) return status;
    if (instance->program->ctc.enabled) {
        activation = offset_ptr(instance, instance->program->values[instance->program->ctc.activation_value].offset);
        status = lw_x64_rec_ctc_execute(instance->program, instance, activation, error);
        if (status != LW_STATUS_OK) return status;
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
