#include "x64_det_backend_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"
#include "parallel_internal.h"
#include "rec_backend_kernels_internal.h"
#include "../simd/simd_kernels.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_MSC_VER)
#include <malloc.h>
static void* det_aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
static void det_aligned_free(void* p) { _aligned_free(p); }
#else
static void* det_aligned_alloc(size_t alignment, size_t size) {
    void* p = NULL;
    return posix_memalign(&p, alignment, size) == 0 ? p : NULL;
}
static void det_aligned_free(void* p) { free(p); }
#endif

static float* offset_ptr(lw_x64_det_instance* instance, uint64_t offset) {
    if (instance == NULL || instance->arena == NULL || offset > SIZE_MAX) return NULL;
    return (float*)(void*)(instance->arena + (size_t)offset);
}

/* Defined below; the shard machinery sits between these and the op loop. */
static lw_scalar_binary_op binary_operation(uint16_t operation);
static void scalar_nhwc_conv_row_range(const lw_x64_det_conv_op* conv, const float* input,
                                       float* output, uint32_t row_begin, uint32_t row_end,
                                       const float* post_bias, const float* residual);
static void scalar_nhwc_conv(const lw_x64_det_conv_op* conv, const float* input, float* output,
                             const float* post_bias, const float* residual);
static void scalar_nhwc_batch_norm(const lw_x64_det_affine_op* affine, const float* input,
                                   float* output);

/* Threaded-run state carried through the op loop. NULL keeps the proven
 * serial path (contract driver, test hooks). */
typedef struct lw_x64_det_run_state {
    lw_x64_det_instance* instance;
    lw_thread_pool* pool;
    uint32_t worker_count;
} lw_x64_det_run_state;

/* Reference sharding thresholds (SimdPaddleOCR): dense/pointwise shard at 2M
 * MACs, depthwise and the 1-channel probability map at 1M. Workers cap at the
 * pool size (16). */
#define LW_X64_DET_CONV_SHARD_MACS UINT64_C(2000000)
#define LW_X64_DET_DEPTHWISE_SHARD_MACS UINT64_C(1000000)
#define LW_X64_DET_ELEMENTWISE_SHARD_ELEMS UINT64_C(131072)
#define LW_X64_DET_CONCAT_SHARD_PIXELS UINT64_C(8192)

static uint64_t conv_macs(const lw_x64_det_conv_op* conv) {
    uint64_t macs = (uint64_t)conv->output_channels * conv->input_channels *
                    conv->kernel_h * conv->kernel_w * conv->output_height * conv->output_width;
    /* Depthwise per-pixel work does not scale with the channel count. */
    if (conv->groups == conv->input_channels && conv->groups == conv->output_channels &&
        conv->groups != 1u) {
        macs /= conv->input_channels;
    }
    return macs;
}

static uint32_t det_shard_workers(const lw_x64_det_run_state* state, uint64_t macs,
                                  uint64_t threshold) {
    if (state == NULL || state->pool == NULL || state->worker_count <= 1u ||
        macs < threshold) {
        return 1u;
    }
    return state->worker_count > LW_PARALLEL_MAX_WORKERS
        ? LW_PARALLEL_MAX_WORKERS : state->worker_count;
}

static lw_scalar_binary_op binary_operation(uint16_t operation);

/* Shard worker context: one physical op, per-kind subranges. Whole-tile and
 * whole-row partitioning keeps every output element computed by exactly one
 * worker in the same summation order as the serial run (bit-identical). */
typedef enum lw_x64_det_shard_kind {
    LW_X64_DET_SHARD_POINTWISE_SPATIAL = 0,
    LW_X64_DET_SHARD_DENSE_ROWS = 2,
    LW_X64_DET_SHARD_DEPTHWISE_ROWS = 3,
    LW_X64_DET_SHARD_CT16_ROWS = 4,
    LW_X64_DET_SHARD_CTC1_ROWS = 5,
    LW_X64_DET_SHARD_BINARY_SAME = 6,
    LW_X64_DET_SHARD_BINARY_SCALAR = 7,
    LW_X64_DET_SHARD_BINARY_CHANNEL = 8,
    LW_X64_DET_SHARD_SIGMOID = 9,
    LW_X64_DET_SHARD_CONCAT_NHWC = 10,
    LW_X64_DET_SHARD_POOL_NCHW_CHANNELS = 11,
    LW_X64_DET_SHARD_POOL_NHWC_ROWS = 12,
    LW_X64_DET_SHARD_REDUCE_CHANNELS = 13
} lw_x64_det_shard_kind;

typedef struct lw_x64_det_shard_ctx {
    uint16_t kind;
    uint16_t pointwise_kernel;
    const lw_x64_det_op* op;
    lw_x64_det_instance* instance;
    const float* input;
    const float* input2;
    float* output_base;
    uint8_t binary_left_is_channel;
    lw_nhwc_epilogue epilogue;
    uint32_t items;
    uint32_t dense_kc;
    uint8_t* scratch_base;
    uint64_t scratch_bytes;
    lw_status statuses[LW_PARALLEL_MAX_WORKERS];
} lw_x64_det_shard_ctx;

static void det_shard_worker(void* opaque, uint32_t worker_index, uint32_t worker_count) {
    lw_x64_det_shard_ctx* ctx = (lw_x64_det_shard_ctx*)opaque;
    const lw_x64_det_conv_op* conv = &ctx->op->data.conv;
    uint32_t begin;
    uint32_t end;
    ctx->statuses[worker_index] = LW_STATUS_OK;
    switch (ctx->kind) {
    case LW_X64_DET_SHARD_POINTWISE_SPATIAL: {
        lw_nhwc_epilogue ep = ctx->epilogue;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        /* The pointwise kernels index the residual by slice-local pixel, so
         * advance it to this worker's first pixel (same offset as output). */
        if (ep.residual != NULL) ep.residual += (size_t)begin * conv->output_channels;
        switch (ctx->pointwise_kernel) {
        case LW_X64_DET_PW_4X16:
            lw_avx2_fma_nhwc_pointwise_4x16_f32(
                ctx->input + (size_t)begin * conv->input_channels, conv->packed_weights,
                &ep, ctx->output_base + (size_t)begin * conv->output_channels,
                end - begin, conv->input_channels, conv->output_channels);
            break;
        case LW_X64_DET_PW_3X32:
            lw_avx2_fma_nhwc_pointwise_3x32_f32(
                ctx->input + (size_t)begin * conv->input_channels, conv->packed_weights,
                &ep, ctx->output_base + (size_t)begin * conv->output_channels,
                end - begin, conv->input_channels, conv->output_channels);
            break;
        case LW_X64_DET_PW_2X32:
            lw_avx2_fma_nhwc_pointwise_2x32_f32(
                ctx->input + (size_t)begin * conv->input_channels, conv->packed_weights,
                &ep, ctx->output_base + (size_t)begin * conv->output_channels,
                end - begin, conv->input_channels, conv->output_channels);
            break;
        default:
            lw_avx2_fma_nhwc_pointwise_f32(
                ctx->input + (size_t)begin * conv->input_channels, conv->packed_weights,
                &ep, ctx->output_base + (size_t)begin * conv->output_channels,
                end - begin, conv->input_channels, conv->output_channels);
            break;
        }
        break;
    }
    case LW_X64_DET_SHARD_DENSE_ROWS: {
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        lw_nhwc_dense_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.input_channels = conv->input_channels;
        desc.input_height = conv->input_height;
        desc.input_width = conv->input_width;
        desc.output_channels = conv->output_channels;
        desc.output_height = end - begin;
        desc.output_width = conv->output_width;
        desc.kernel_h = conv->kernel_h;
        desc.kernel_w = conv->kernel_w;
        desc.stride_h = conv->stride_h;
        desc.stride_w = conv->stride_w;
        desc.pad_top = conv->pad_top;
        desc.pad_left = conv->pad_left;
        desc.pad_bottom = conv->pad_bottom;
        desc.pad_right = conv->pad_right;
        desc.dense_kc = ctx->dense_kc;
        desc.output_row_offset = begin;
        {
            lw_nhwc_epilogue ep = ctx->epilogue;
            /* The dense kernel indexes the residual by slice-local row, so
             * advance it to this worker's first row (same offset as output). */
            if (ep.residual != NULL) {
                ep.residual += (size_t)begin * conv->output_width * conv->output_channels;
            }
            ctx->statuses[worker_index] = lw_avx2_fma_nhwc_dense_f32(
                ctx->input, conv->packed_weights, &ep,
                ctx->output_base + (size_t)begin * conv->output_width * conv->output_channels,
                &desc, ctx->scratch_base + (size_t)worker_index * ctx->scratch_bytes,
                ctx->scratch_bytes);
        }
        if (ctx->statuses[worker_index] != LW_STATUS_OK) {
            scalar_nhwc_conv_row_range(conv, ctx->input, ctx->output_base, begin, end,
                                       ctx->epilogue.post_bias, ctx->epilogue.residual);
            ctx->statuses[worker_index] = LW_STATUS_OK;
        }
        break;
    }
    case LW_X64_DET_SHARD_DEPTHWISE_ROWS: {
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        lw_nhwc_depthwise_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.channels = conv->input_channels;
        desc.input_height = conv->input_height;
        desc.input_width = conv->input_width;
        desc.output_height = end - begin;
        desc.output_width = conv->output_width;
        desc.kernel_h = conv->kernel_h;
        desc.kernel_w = conv->kernel_w;
        desc.stride_h = conv->stride_h;
        desc.stride_w = conv->stride_w;
        desc.pad_top = conv->pad_top;
        desc.pad_left = conv->pad_left;
        desc.pad_bottom = conv->pad_bottom;
        desc.pad_right = conv->pad_right;
        desc.output_row_offset = begin;
        ctx->statuses[worker_index] = lw_avx2_fma_nhwc_depthwise_f32(
            ctx->input, conv->packed_weights, ctx->epilogue.bias,
            ctx->output_base + (size_t)begin * conv->output_width * conv->output_channels,
            &desc);
        if (ctx->statuses[worker_index] != LW_STATUS_OK) {
            scalar_nhwc_conv_row_range(conv, ctx->input, ctx->output_base, begin, end,
                                       NULL, NULL);
            ctx->statuses[worker_index] = LW_STATUS_OK;
        }
        break;
    }
    case LW_X64_DET_SHARD_CT16_ROWS:
    case LW_X64_DET_SHARD_CTC1_ROWS: {
        const lw_x64_det_conv_transpose_op* ct = &ctx->op->data.conv_transpose;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        lw_nhwc_convtranspose_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.input_channels = (uint32_t)ct->input_dimensions[1];
        desc.input_height = end - begin;
        desc.input_width = (uint32_t)ct->input_dimensions[3];
        desc.output_channels = (uint32_t)ct->output_dimensions[1];
        desc.output_height = (end - begin) * 2u;
        desc.output_width = (uint32_t)ct->output_dimensions[3];
        if (ctx->kind == LW_X64_DET_SHARD_CTC1_ROWS) {
            ctx->statuses[worker_index] = lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32(
                ctx->input + (size_t)begin * desc.input_width * desc.input_channels,
                ct->weights, &ctx->epilogue,
                ctx->output_base + (size_t)(begin * 2u) * desc.output_width *
                    desc.output_channels, &desc);
        } else {
            ctx->statuses[worker_index] = lw_avx2_fma_nhwc_convtranspose2x2_s2_f32(
                ctx->input + (size_t)begin * desc.input_width * desc.input_channels,
                ct->packed_weights, &ctx->epilogue,
                ctx->output_base + (size_t)(begin * 2u) * desc.output_width *
                    desc.output_channels, &desc);
        }
        break;
    }
    case LW_X64_DET_SHARD_BINARY_SAME:
    case LW_X64_DET_SHARD_BINARY_SCALAR:
    case LW_X64_DET_SHARD_SIGMOID: {
        uint64_t element_begin = ((uint64_t)ctx->items * worker_index) / worker_count;
        uint64_t element_end = ((uint64_t)ctx->items * (worker_index + 1u)) / worker_count;
        if (element_begin >= element_end) return;
        if (ctx->kind == LW_X64_DET_SHARD_SIGMOID) {
            ctx->statuses[worker_index] = lw_scalar_sigmoid_f32(
                ctx->input + element_begin, ctx->output_base + element_begin,
                element_end - element_begin);
        } else if (ctx->kind == LW_X64_DET_SHARD_BINARY_SCALAR) {
            lw_avx2_binary_right_scalar_f32(
                binary_operation(ctx->op->data.binary.operation),
                ctx->input + element_begin, ctx->op->data.binary.right_constant[0],
                ctx->output_base + element_begin, element_end - element_begin);
        } else {
            lw_avx2_binary_contiguous_f32(
                binary_operation(ctx->op->data.binary.operation),
                ctx->input + element_begin, ctx->input2 + element_begin,
                ctx->output_base + element_begin, element_end - element_begin);
        }
        break;
    }
    case LW_X64_DET_SHARD_BINARY_CHANNEL: {
        uint32_t channels = ctx->op->data.binary.channels;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        lw_avx2_binary_channel_f32(binary_operation(ctx->op->data.binary.operation),
                                   ctx->input + (size_t)begin * channels, ctx->input2,
                                   ctx->output_base + (size_t)begin * channels,
                                   (uint64_t)(end - begin), channels,
                                   ctx->binary_left_is_channel);
        break;
    }
    case LW_X64_DET_SHARD_CONCAT_NHWC: {
        const float* inputs[LWM_V0_MAX_NODE_INPUTS];
        uint32_t input_slot;
        uint32_t output_channels = 0u;
        uint64_t pixel;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        for (input_slot = 0u; input_slot < ctx->op->data.concat.input_count; ++input_slot) {
            inputs[input_slot] =
                offset_ptr(ctx->instance, ctx->op->data.concat.input_offsets[input_slot]);
            if (inputs[input_slot] == NULL) {
                ctx->statuses[worker_index] = LW_STATUS_INVALID_ARGUMENT;
                return;
            }
            output_channels += (uint32_t)ctx->op->data.concat.input_dimensions[input_slot][1];
        }
        for (pixel = begin; pixel < end; ++pixel) {
            float* destination = ctx->output_base + (size_t)pixel * output_channels;
            for (input_slot = 0u; input_slot < ctx->op->data.concat.input_count; ++input_slot) {
                uint32_t channels =
                    (uint32_t)ctx->op->data.concat.input_dimensions[input_slot][1];
                memcpy(destination, inputs[input_slot] + (size_t)pixel * channels,
                       (size_t)channels * sizeof(float));
                destination += channels;
            }
        }
        break;
    }
    case LW_X64_DET_SHARD_POOL_NCHW_CHANNELS: {
        int32_t input_dims[4];
        int32_t output_dims[4];
        size_t input_slice;
        size_t output_slice;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        memcpy(input_dims, ctx->op->data.pool.input_dimensions, sizeof(input_dims));
        memcpy(output_dims, ctx->op->data.pool.output_dimensions, sizeof(output_dims));
        input_slice = (size_t)input_dims[2] * (size_t)input_dims[3];
        output_slice = (size_t)output_dims[2] * (size_t)output_dims[3];
        input_dims[1] = (int32_t)(end - begin);
        output_dims[1] = (int32_t)(end - begin);
        ctx->statuses[worker_index] = ctx->op->kind == LW_X64_DET_OP_MAX_POOL
            ? lw_scalar_max_pool2d_f32(ctx->input + (size_t)begin * input_slice,
                                       ctx->output_base + (size_t)begin * output_slice,
                                       input_dims, output_dims, ctx->op->data.pool.kernel,
                                       ctx->op->data.pool.strides, ctx->op->data.pool.pads, 0u)
            : lw_scalar_average_pool2d_f32(ctx->input + (size_t)begin * input_slice,
                                           ctx->output_base + (size_t)begin * output_slice,
                                           input_dims, output_dims, ctx->op->data.pool.kernel,
                                           ctx->op->data.pool.strides, ctx->op->data.pool.pads, 0u,
                                           ctx->op->data.pool.count_include_pad);
        break;
    }
    case LW_X64_DET_SHARD_POOL_NHWC_ROWS: {
        uint32_t channels = (uint32_t)ctx->op->data.pool.input_dimensions[3];
        uint32_t input_width = (uint32_t)ctx->op->data.pool.input_dimensions[2];
        uint32_t output_width = (uint32_t)ctx->op->data.pool.output_dimensions[2];
        uint32_t stride_h = (uint32_t)ctx->op->data.pool.strides[0];
        uint32_t pad_top = (uint32_t)ctx->op->data.pool.pads[0];
        const float* input = ctx->input;
        uint32_t input_height = (uint32_t)ctx->op->data.pool.input_dimensions[1];
        uint32_t pad = pad_top;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        if (begin > 0u) {
            uint32_t row_offset = begin * stride_h - pad_top;
            input = ctx->input + (size_t)row_offset * input_width * channels;
            input_height -= row_offset;
            pad = 0u;
        }
        lw_avx2_nhwc_pool_f32(
            input, ctx->output_base + (size_t)begin * output_width * channels, 1u,
            input_height, input_width, end - begin, output_width, channels,
            (uint32_t)ctx->op->data.pool.kernel[0], (uint32_t)ctx->op->data.pool.kernel[1],
            stride_h, (uint32_t)ctx->op->data.pool.strides[1], pad,
            (uint32_t)ctx->op->data.pool.pads[1], ctx->op->data.pool.count_include_pad,
            ctx->op->data.pool.is_max);
        break;
    }
    case LW_X64_DET_SHARD_REDUCE_CHANNELS: {
        uint32_t channels = ctx->op->data.reduce.channels;
        uint32_t height = ctx->op->data.reduce.height;
        uint32_t width = ctx->op->data.reduce.width;
        begin = (uint32_t)(((uint64_t)ctx->items * worker_index) / worker_count);
        end = (uint32_t)(((uint64_t)ctx->items * (worker_index + 1u)) / worker_count);
        if (begin >= end) return;
        if (ctx->op->data.reduce.layout == LW_X64_DET_LAYOUT_NCHW) {
            uint32_t channel;
            for (channel = begin; channel < end; ++channel) {
                const float* source = ctx->input + (size_t)channel * height * width;
                float sum = 0.0f;
                uint32_t spatial;
                for (spatial = 0u; spatial < height * width; ++spatial) {
                    sum += source[spatial];
                }
                ctx->output_base[channel] = sum / (float)(height * width);
            }
        } else {
            lw_avx2_nhwc_reduce_mean_hw_strided_f32(
                ctx->input + begin, ctx->output_base + begin,
                ctx->op->data.reduce.batch, height, width, end - begin,
                channels, channels);
        }
        break;
    }
    default:
        ctx->statuses[worker_index] = LW_STATUS_UNSUPPORTED;
        break;
    }
}

static lw_status run_sharded_conv(lw_x64_det_instance* instance, const lw_x64_det_op* op,
                                  const lw_x64_det_run_state* state,
                                  uint16_t shard_kind, uint32_t items,
                                  lw_nhwc_epilogue epilogue) {
    lw_x64_det_shard_ctx ctx;
    uint64_t macs;
    uint64_t threshold = LW_X64_DET_CONV_SHARD_MACS;
    uint32_t workers;
    uint32_t w;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = shard_kind;
    ctx.op = op;
    ctx.epilogue = epilogue;
    ctx.items = items;
    ctx.dense_kc = op->data.conv.dense_kc;
    ctx.pointwise_kernel = op->data.conv.pointwise_kernel;
    switch (shard_kind) {
    case LW_X64_DET_SHARD_POINTWISE_SPATIAL:
    case LW_X64_DET_SHARD_DENSE_ROWS:
        macs = conv_macs(&op->data.conv);
        ctx.input = offset_ptr(instance, op->data.conv.input_offset);
        ctx.output_base = offset_ptr(instance, op->data.conv.output_offset);
        ctx.scratch_bytes = op->data.conv.scratch_bytes;
        ctx.scratch_base = instance->shard_scratch;
        break;
    case LW_X64_DET_SHARD_DEPTHWISE_ROWS:
        threshold = LW_X64_DET_DEPTHWISE_SHARD_MACS;
        macs = conv_macs(&op->data.conv);
        ctx.input = offset_ptr(instance, op->data.conv.input_offset);
        ctx.output_base = offset_ptr(instance, op->data.conv.output_offset);
        break;
    case LW_X64_DET_SHARD_CT16_ROWS:
    case LW_X64_DET_SHARD_CTC1_ROWS:
        macs = (uint64_t)op->data.conv_transpose.input_dimensions[1] *
               op->data.conv_transpose.output_dimensions[1] * 4u *
               op->data.conv_transpose.output_dimensions[2] *
               op->data.conv_transpose.output_dimensions[3];
        threshold = shard_kind == LW_X64_DET_SHARD_CTC1_ROWS
            ? LW_X64_DET_DEPTHWISE_SHARD_MACS : LW_X64_DET_CONV_SHARD_MACS;
        ctx.input = offset_ptr(instance, op->data.conv_transpose.input_offset);
        ctx.output_base = offset_ptr(instance, op->data.conv_transpose.output_offset);
        break;
    default:
        return LW_STATUS_UNSUPPORTED;
    }
    if (ctx.input == NULL || ctx.output_base == NULL) return LW_STATUS_INVALID_ARGUMENT;
    if (shard_kind == LW_X64_DET_SHARD_DENSE_ROWS &&
        (ctx.scratch_base == NULL || ctx.scratch_bytes == 0u)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    workers = det_shard_workers(state, macs, threshold);
    if (workers <= 1u) return LW_STATUS_UNSUPPORTED;
    lw_thread_pool_run(state->pool, workers, det_shard_worker, &ctx);
    for (w = 0u; w < workers; ++w) {
        if (ctx.statuses[w] != LW_STATUS_OK) return ctx.statuses[w];
    }
    return LW_STATUS_OK;
}

/* Elementwise-style sharding: every output element is computed independently
 * by the same per-element kernel as the serial path, so splitting the
 * contiguous range keeps results bit-identical. Returns LW_STATUS_UNSUPPORTED
 * when the op is too small or no pool is available and the caller must run
 * the serial kernel instead. */
static lw_status run_sharded_elementwise(lw_x64_det_instance* instance,
                                         const lw_x64_det_op* op,
                                         const lw_x64_det_run_state* state,
                                         uint16_t shard_kind, const float* input,
                                         const float* input2, float* output, uint64_t items,
                                         uint64_t threshold, int binary_left_is_channel) {
    lw_x64_det_shard_ctx ctx;
    uint32_t workers;
    uint32_t w;
    if (items > UINT32_MAX) return LW_STATUS_UNSUPPORTED;
    workers = det_shard_workers(state, items, threshold);
    if (workers <= 1u) return LW_STATUS_UNSUPPORTED;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = shard_kind;
    ctx.op = op;
    ctx.instance = instance;
    ctx.input = input;
    ctx.input2 = input2;
    ctx.output_base = output;
    ctx.items = (uint32_t)items;
    ctx.binary_left_is_channel = (uint8_t)binary_left_is_channel;
    lw_thread_pool_run(state->pool, workers, det_shard_worker, &ctx);
    for (w = 0u; w < workers; ++w) {
        if (ctx.statuses[w] != LW_STATUS_OK) return ctx.statuses[w];
    }
    return LW_STATUS_OK;
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

static void scalar_nhwc_conv_row_range(const lw_x64_det_conv_op* conv, const float* input,
                                       float* output, uint32_t row_begin, uint32_t row_end,
                                       const float* post_bias, const float* residual) {
    uint32_t oy, ox, oc, ky, kx, ic;
    for (oy = row_begin; oy < row_end; ++oy) {
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
                            sum += input[input_index] * conv->original_weights[weight_index];
                        }
                    }
                }
                if (post_bias != NULL) sum += post_bias[oc];
                if (residual != NULL) {
                    sum += residual[((size_t)oy * conv->output_width + ox) *
                                    conv->output_channels + oc];
                }
                if (conv->activation == LW_NHWC_ACT_RELU && sum < 0.0f) sum = 0.0f;
                else if (conv->activation == LW_NHWC_ACT_GELU) sum = lw_nhwc_gelu_scalar_exact_f32(sum);
                output[((size_t)oy * conv->output_width + ox) * conv->output_channels + oc] = sum;
            }
        }
    }
}

static void scalar_nhwc_conv(const lw_x64_det_conv_op* conv, const float* input,
                             float* output, const float* post_bias, const float* residual) {
    scalar_nhwc_conv_row_range(conv, input, output, 0u, conv->output_height,
                               post_bias, residual);
}

static void scalar_nhwc_batch_norm(const lw_x64_det_affine_op* affine, const float* input,
                                   float* output) {
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

static lw_status run_sharded_conv(lw_x64_det_instance* instance, const lw_x64_det_op* op,
                                  const lw_x64_det_run_state* state,
                                  uint16_t shard_kind, uint32_t items,
                                  lw_nhwc_epilogue epilogue);

static lw_status execute_op(lw_x64_det_instance* instance, const lw_x64_det_op* op,
                            const lw_x64_det_run_state* state, lw_error* error) {
    (void)error;
    float* input;
    float* output;
    if (op == NULL) return LW_STATUS_INVALID_ARGUMENT;
    switch (op->kind) {
    case LW_X64_DET_OP_CONV_NCHW: {
        int32_t input_dimensions[4] = {
            1, (int32_t)op->data.conv.input_channels,
            (int32_t)op->data.conv.input_height,
            (int32_t)op->data.conv.input_width
        };
        int32_t output_dimensions[4] = {
            1, (int32_t)op->data.conv.output_channels,
            (int32_t)op->data.conv.output_height,
            (int32_t)op->data.conv.output_width
        };
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        switch (op->data.conv.nchw_conv_kernel) {
        case LW_X64_DET_NCHW_CONV_POINTWISE_PACKED:
            if (op->data.conv.nchw_pointwise_kernel == LW_X64_DET_NCHW_PW_FMA8) {
                lw_avx2_fma_packed_conv1x1_8x8_f32(
                    input, op->data.conv.packed_weights, op->data.conv.bias,
                    output, input_dimensions, output_dimensions);
            } else {
                lw_avx2_fma_packed_conv1x1_f32(
                    input, op->data.conv.packed_weights, op->data.conv.bias,
                    output, input_dimensions, output_dimensions);
            }
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_3X3S2:
            lw_avx2_conv3x3_stride2_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_3X3_UNIT:
            lw_avx2_conv3x3_unit_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_2X2_PADEND1:
            lw_avx2_conv2x2_unit_pad_end1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_DEPTHWISE3X3:
            lw_avx2_depthwise_conv3x3_unit_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_DEPTHWISE3X3_S2X1:
            lw_avx2_depthwise_conv3x3_stride2x1_pad1_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions, output_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_DEPTHWISE5X5:
            lw_avx2_depthwise_conv5x5_unit_pad2_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                output, input_dimensions);
            return LW_STATUS_OK;
        case LW_X64_DET_NCHW_CONV_SCALAR:
        default: {
            int32_t weight_dimensions[4] = {
                (int32_t)op->data.conv.output_channels,
                op->data.conv.groups == op->data.conv.input_channels
                    ? 1 : (int32_t)op->data.conv.input_channels,
                (int32_t)op->data.conv.weight_h, (int32_t)op->data.conv.weight_w
            };
            int32_t kernel[2] = { (int32_t)op->data.conv.kernel_h, (int32_t)op->data.conv.kernel_w };
            int32_t strides[2] = { (int32_t)op->data.conv.stride_h, (int32_t)op->data.conv.stride_w };
            int32_t dilations[2] = { 1, 1 };
            int32_t pads[4] = { (int32_t)op->data.conv.pad_top, (int32_t)op->data.conv.pad_left,
                                (int32_t)op->data.conv.pad_bottom, (int32_t)op->data.conv.pad_right };
            lw_status fallback_status = lw_scalar_conv2d_f32(
                input, op->data.conv.original_weights, op->data.conv.bias,
                op->data.conv.bias != NULL ? op->data.conv.output_channels : 0u, output,
                input_dimensions, weight_dimensions, output_dimensions, kernel, strides,
                dilations, pads, op->data.conv.groups);
            if (fallback_status != LW_STATUS_OK) {
                lw_set_error(error, fallback_status, "NCHW conv scalar fallback failed");
                return fallback_status;
            }
            return LW_STATUS_OK;
        }
        }
    }
    case LW_X64_DET_OP_POINTWISE: {
        lw_nhwc_epilogue ep = { NULL, NULL, op->data.conv.activation, 0u, 0.0f, 0.0f, NULL };
        uint32_t pixels = op->data.conv.input_height * op->data.conv.input_width;
        uint32_t workers = det_shard_workers(state, conv_macs(&op->data.conv),
                                             LW_X64_DET_CONV_SHARD_MACS);
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        ep.bias = op->data.conv.bias;
        ep.post_bias = op->data.conv.post_bias;
        if (op->data.conv.has_residual) {
            ep.residual = offset_ptr(instance, op->data.conv.residual_offset);
            if (ep.residual == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        if (workers > 1u) {
            /* The pointwise kernels write tightly packed pixel rows. Shard by
             * pixels so each worker keeps the full NHWC output-channel stride. */
            lw_status status = run_sharded_conv(
                instance, op, state, LW_X64_DET_SHARD_POINTWISE_SPATIAL, pixels, ep);
            if (status != LW_STATUS_OK) return status;
            return LW_STATUS_OK;
        }
#if defined(LW_WASM_COMPILED_DET)
        lw_rec_backend_kernels_current()->pointwise(
            input, op->data.conv.packed_weights, &ep, output, pixels,
            op->data.conv.input_channels, op->data.conv.output_channels);
#else
        switch (op->data.conv.pointwise_kernel) {
        case LW_X64_DET_PW_4X16:
            lw_avx2_fma_nhwc_pointwise_4x16_f32(
                input, op->data.conv.packed_weights, &ep, output, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
            break;
        case LW_X64_DET_PW_3X32:
            lw_avx2_fma_nhwc_pointwise_3x32_f32(
                input, op->data.conv.packed_weights, &ep, output, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
            break;
        case LW_X64_DET_PW_2X32:
            lw_avx2_fma_nhwc_pointwise_2x32_f32(
                input, op->data.conv.packed_weights, &ep, output, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
            break;
        default:
            lw_avx2_fma_nhwc_pointwise_f32(
                input, op->data.conv.packed_weights, &ep, output, pixels,
                op->data.conv.input_channels, op->data.conv.output_channels);
            break;
        }
#endif
        return LW_STATUS_OK;
    }
    case LW_X64_DET_OP_DENSE: {
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
        ep.post_bias = op->data.conv.post_bias;
        input = offset_ptr(instance, op->data.conv.input_offset);
        output = offset_ptr(instance, op->data.conv.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.conv.has_residual) {
            ep.residual = offset_ptr(instance, op->data.conv.residual_offset);
            if (ep.residual == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        if (det_shard_workers(state, conv_macs(&op->data.conv),
                              LW_X64_DET_CONV_SHARD_MACS) > 1u &&
            op->data.conv.output_height > 1u) {
            return run_sharded_conv(instance, op, state, LW_X64_DET_SHARD_DENSE_ROWS,
                                    op->data.conv.output_height, ep);
        }
        status = lw_avx2_fma_nhwc_dense_f32(
            input, op->data.conv.packed_weights, &ep, output, &desc,
            instance->scratch, op->data.conv.scratch_bytes);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        scalar_nhwc_conv(&op->data.conv, input, output, ep.post_bias, ep.residual);
        return LW_STATUS_OK;
    }
    case LW_X64_DET_OP_DEPTHWISE: {
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
        if (det_shard_workers(state, conv_macs(&op->data.conv),
                              LW_X64_DET_DEPTHWISE_SHARD_MACS) > 1u &&
            op->data.conv.output_height > 1u) {
            lw_nhwc_epilogue ep = { op->data.conv.bias, NULL, 0u, 0u, 0.0f, 0.0f, NULL };
            return run_sharded_conv(instance, op, state, LW_X64_DET_SHARD_DEPTHWISE_ROWS,
                                    op->data.conv.output_height, ep);
        }
        status = lw_avx2_fma_nhwc_depthwise_f32(input, op->data.conv.packed_weights,
                                                op->data.conv.bias, output, &desc);
        if (status == LW_STATUS_OK) return LW_STATUS_OK;
        scalar_nhwc_conv(&op->data.conv, input, output, NULL, NULL);
        return LW_STATUS_OK;
    }
    case LW_X64_DET_OP_LAYOUT_CONVERT:
        input = offset_ptr(instance, op->data.convert.input_offset);
        output = offset_ptr(instance, op->data.convert.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.convert.to_nhwc) {
            lw_x64_fast_nchw_to_nhwc(input, output, op->data.convert.batch,
                                     op->data.convert.channels, op->data.convert.height,
                                     op->data.convert.width);
        } else {
            lw_x64_fast_nhwc_to_nchw(input, output, op->data.convert.batch,
                                     op->data.convert.channels, op->data.convert.height,
                                     op->data.convert.width);
        }
        return LW_STATUS_OK;
    case LW_X64_DET_OP_CONV_TRANSPOSE_NCHW:
    case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC:
    case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1:
        input = offset_ptr(instance, op->data.conv_transpose.input_offset);
        output = offset_ptr(instance, op->data.conv_transpose.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NCHW) {
            lw_scalar_conv_transpose2x2_stride2_f32(
                input, op->data.conv_transpose.weights, op->data.conv_transpose.bias,
                output, op->data.conv_transpose.input_dimensions,
                op->data.conv_transpose.output_dimensions);
            return LW_STATUS_OK;
        }
        {
            lw_nhwc_convtranspose_desc desc;
            lw_nhwc_epilogue epilogue;
            uint64_t ct_macs = (uint64_t)op->data.conv_transpose.input_dimensions[1] *
                               op->data.conv_transpose.output_dimensions[1] * 4u *
                               op->data.conv_transpose.output_dimensions[2] *
                               op->data.conv_transpose.output_dimensions[3];
            memset(&epilogue, 0, sizeof(epilogue));
            epilogue.bias = op->data.conv_transpose.bias;
            if (det_shard_workers(
                    state, ct_macs,
                    op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1
                        ? LW_X64_DET_DEPTHWISE_SHARD_MACS : LW_X64_DET_CONV_SHARD_MACS) > 1u &&
                op->data.conv_transpose.input_dimensions[2] > 1) {
                return run_sharded_conv(
                    instance, op, state,
                    op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1
                        ? LW_X64_DET_SHARD_CTC1_ROWS : LW_X64_DET_SHARD_CT16_ROWS,
                    (uint32_t)op->data.conv_transpose.input_dimensions[2], epilogue);
            }
            memset(&desc, 0, sizeof(desc));
            desc.batch = (uint32_t)op->data.conv_transpose.input_dimensions[0];
            desc.input_channels = (uint32_t)op->data.conv_transpose.input_dimensions[1];
            desc.input_height = (uint32_t)op->data.conv_transpose.input_dimensions[2];
            desc.input_width = (uint32_t)op->data.conv_transpose.input_dimensions[3];
            desc.output_channels = (uint32_t)op->data.conv_transpose.output_dimensions[1];
            desc.output_height = (uint32_t)op->data.conv_transpose.output_dimensions[2];
            desc.output_width = (uint32_t)op->data.conv_transpose.output_dimensions[3];
            if (op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1) {
#if defined(LW_WASM_COMPILED_DET)
                return lw_wasm128_nhwc_convtranspose2x2_s2_c1_f32(
                    input, op->data.conv_transpose.weights, &epilogue, output, &desc);
#else
                return lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32(
                    input, op->data.conv_transpose.weights, &epilogue, output, &desc);
#endif
            }
#if defined(LW_WASM_COMPILED_DET)
            return lw_wasm128_nhwc_convtranspose2x2_s2_f32(
                input, op->data.conv_transpose.packed_weights, &epilogue, output, &desc);
#else
            return lw_avx2_fma_nhwc_convtranspose2x2_s2_f32(
                input, op->data.conv_transpose.packed_weights, &epilogue, output, &desc);
#endif
        }
    case LW_X64_DET_OP_RESIZE_NCHW:
        input = offset_ptr(instance, op->data.resize.input_offset);
        output = offset_ptr(instance, op->data.resize.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        return lw_scalar_resize_nearest_f32(
            input, output, op->data.resize.rank, op->data.resize.input_dimensions,
            op->data.resize.output_dimensions, op->data.resize.scales);
    case LW_X64_DET_OP_RESIZE_NHWC:
        input = offset_ptr(instance, op->data.resize.input_offset);
        output = offset_ptr(instance, op->data.resize.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        lw_avx2_nhwc_resize_nearest_f32(
            input, output, (uint32_t)op->data.resize.input_dimensions[0],
            (uint32_t)op->data.resize.input_dimensions[1],
            (uint32_t)op->data.resize.input_dimensions[2],
            (uint32_t)op->data.resize.input_dimensions[3],
            (uint32_t)op->data.resize.output_dimensions[2],
            (uint32_t)op->data.resize.output_dimensions[3]);
        return LW_STATUS_OK;
    case LW_X64_DET_OP_SIGMOID_NCHW:
        input = offset_ptr(instance, op->data.unary.input_offset);
        output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (state != NULL) {
            lw_status sharded = run_sharded_elementwise(
                instance, op, state, LW_X64_DET_SHARD_SIGMOID, input, NULL, output,
                op->data.unary.element_count, LW_X64_DET_ELEMENTWISE_SHARD_ELEMS, 0);
            if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
        }
        return lw_scalar_sigmoid_f32(input, output, op->data.unary.element_count);
    case LW_X64_DET_OP_CONCAT_NCHW: {
        const float* inputs[LWM_V0_MAX_NODE_INPUTS];
        uint32_t input_slot;
        for (input_slot = 0u; input_slot < op->data.concat.input_count; ++input_slot) {
            inputs[input_slot] = offset_ptr(instance, op->data.concat.input_offsets[input_slot]);
            if (inputs[input_slot] == NULL) return LW_STATUS_INVALID_ARGUMENT;
        }
        output = offset_ptr(instance, op->data.concat.output_offset);
        if (output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        {
            const int32_t* dimensions[LWM_V0_MAX_NODE_INPUTS];
            uint32_t ranks[LWM_V0_MAX_NODE_INPUTS];
            for (input_slot = 0u; input_slot < op->data.concat.input_count; ++input_slot) {
                ranks[input_slot] = op->data.concat.input_ranks[input_slot];
                dimensions[input_slot] = op->data.concat.input_dimensions[input_slot];
            }
            return lw_scalar_concat_f32(inputs, op->data.concat.input_count, ranks, dimensions,
                                        output, op->data.concat.output_rank,
                                        op->data.concat.output_dimensions, op->data.concat.axis);
        }
    }
    case LW_X64_DET_OP_CONCAT_NHWC: {
        /* Channel-axis concat in NHWC is per-pixel concatenated copies; all
         * inputs share the spatial shape, so take it from the first input. */
        uint64_t pixel;
        uint32_t input_slot;
        const float* inputs[LWM_V0_MAX_NODE_INPUTS];
        uint64_t pixel_count;
        uint32_t output_channels = 0u;
        for (input_slot = 0u; input_slot < op->data.concat.input_count; ++input_slot) {
            inputs[input_slot] = offset_ptr(instance, op->data.concat.input_offsets[input_slot]);
            if (inputs[input_slot] == NULL) return LW_STATUS_INVALID_ARGUMENT;
            output_channels += (uint32_t)op->data.concat.input_dimensions[input_slot][1];
        }
        if (op->data.concat.input_count == 0u) return LW_STATUS_INVALID_ARGUMENT;
        pixel_count = (uint64_t)(uint32_t)op->data.concat.input_dimensions[0][0] *
                      (uint64_t)(uint32_t)op->data.concat.input_dimensions[0][2] *
                      (uint64_t)(uint32_t)op->data.concat.input_dimensions[0][3];
        output = offset_ptr(instance, op->data.concat.output_offset);
        if (output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (state != NULL) {
            lw_status sharded = run_sharded_elementwise(
                instance, op, state, LW_X64_DET_SHARD_CONCAT_NHWC, NULL, NULL, output,
                pixel_count, LW_X64_DET_CONCAT_SHARD_PIXELS, 0);
            if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
        }
        for (pixel = 0u; pixel < pixel_count; ++pixel) {
            float* destination = output + (size_t)pixel * output_channels;
            for (input_slot = 0u; input_slot < op->data.concat.input_count; ++input_slot) {
                uint32_t channels =
                    (uint32_t)op->data.concat.input_dimensions[input_slot][1];
                memcpy(destination, inputs[input_slot] + (size_t)pixel * channels,
                       (size_t)channels * sizeof(float));
                destination += channels;
            }
        }
        return LW_STATUS_OK;
    }
    case LW_X64_DET_OP_AFFINE:
        input = offset_ptr(instance, op->data.affine.input_offset);
        output = offset_ptr(instance, op->data.affine.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.affine.channel_major && op->data.affine.pixels > 1u) {
            lw_avx2_nchw_affine_f32(input, op->data.affine.mul, op->data.affine.add,
                                    output, op->data.affine.channels, op->data.affine.pixels);
        } else if (op->data.affine.channel_major) {
            scalar_nhwc_batch_norm(&op->data.affine, input, output);
        } else {
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_affine_nhwc_f32(input, op->data.affine.mul, op->data.affine.add,
                                       output, op->data.affine.pixels,
                                       op->data.affine.channels);
#else
            lw_avx2_nhwc_affine_f32(input, op->data.affine.mul, op->data.affine.add,
                                    output, op->data.affine.pixels, op->data.affine.channels);
#endif
        }
        return LW_STATUS_OK;
    case LW_X64_DET_OP_ADD: case LW_X64_DET_OP_MUL: case LW_X64_DET_OP_DIV:
        input = op->data.binary.left_constant != NULL
            ? (float*)(uintptr_t)op->data.binary.left_constant
            : offset_ptr(instance, op->data.binary.left_offset);
        output = offset_ptr(instance, op->data.binary.output_offset);
        if (op->data.binary.layout == LW_X64_DET_LAYOUT_NCHW &&
            (op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_LEFT_CHANNEL ||
             op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_RIGHT_CHANNEL)) {
            float* full = op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_LEFT_CHANNEL
                ? offset_ptr(instance, op->data.binary.right_offset) : input;
            float* channel = op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_LEFT_CHANNEL
                ? (op->data.binary.left_constant != NULL
                    ? (float*)(uintptr_t)op->data.binary.left_constant
                    : offset_ptr(instance, op->data.binary.left_offset))
                : (op->data.binary.right_constant != NULL
                    ? (float*)(uintptr_t)op->data.binary.right_constant
                    : offset_ptr(instance, op->data.binary.right_offset));
            if (full == NULL || channel == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
            lw_avx2_binary_channel_nchw_f32(
                binary_operation(op->data.binary.operation), full, channel, output,
                op->data.binary.pixels, op->data.binary.channels,
                op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_LEFT_CHANNEL);
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_LEFT_CHANNEL) {
            float* full = offset_ptr(instance, op->data.binary.right_offset);
            float* channel = op->data.binary.left_constant != NULL
                ? (float*)(uintptr_t)op->data.binary.left_constant
                : offset_ptr(instance, op->data.binary.left_offset);
            if (full == NULL || channel == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
            if (state != NULL && op->data.binary.channels > 0u) {
                uint64_t ch_threshold = LW_X64_DET_ELEMENTWISE_SHARD_ELEMS /
                                        op->data.binary.channels;
                lw_status sharded = run_sharded_elementwise(
                    instance, op, state, LW_X64_DET_SHARD_BINARY_CHANNEL, full, channel, output,
                    op->data.binary.pixels, ch_threshold < 2u ? 2u : ch_threshold, 1);
                if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
            }
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_channel_nhwc_f32(binary_operation(op->data.binary.operation),
                full, channel, output, op->data.binary.pixels,
                op->data.binary.channels, 1);
#else
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), full, channel,
                                       output, op->data.binary.pixels, op->data.binary.channels, 1);
#endif
            return LW_STATUS_OK;
        }
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_RIGHT_SCALAR &&
            op->data.binary.right_constant != NULL) {
            if (state != NULL) {
                lw_status sharded = run_sharded_elementwise(
                    instance, op, state, LW_X64_DET_SHARD_BINARY_SCALAR, input, NULL, output,
                    op->data.binary.element_count, LW_X64_DET_ELEMENTWISE_SHARD_ELEMS, 0);
                if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
            }
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_scalar_f32(binary_operation(op->data.binary.operation), input,
                op->data.binary.right_constant[0], output,
                op->data.binary.element_count, 0);
#else
            lw_avx2_binary_right_scalar_f32(binary_operation(op->data.binary.operation), input,
                                            op->data.binary.right_constant[0], output,
                                            op->data.binary.element_count);
#endif
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_SAME) {
            float* right = op->data.binary.right_constant != NULL
                ? (float*)(uintptr_t)op->data.binary.right_constant
                : offset_ptr(instance, op->data.binary.right_offset);
            if (right == NULL) return LW_STATUS_INVALID_ARGUMENT;
            if (state != NULL) {
                lw_status sharded = run_sharded_elementwise(
                    instance, op, state, LW_X64_DET_SHARD_BINARY_SAME, input, right, output,
                    op->data.binary.element_count, LW_X64_DET_ELEMENTWISE_SHARD_ELEMS, 0);
                if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
            }
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_contiguous_f32(binary_operation(op->data.binary.operation),
                input, right, output, op->data.binary.element_count);
#else
            lw_avx2_binary_contiguous_f32(binary_operation(op->data.binary.operation), input,
                                          right, output, op->data.binary.element_count);
#endif
            return LW_STATUS_OK;
        }
        if (op->data.binary.broadcast_kind == LW_X64_DET_BROADCAST_RIGHT_CHANNEL) {
            float* channel = op->data.binary.right_constant != NULL
                ? (float*)(uintptr_t)op->data.binary.right_constant
                : offset_ptr(instance, op->data.binary.right_offset);
            if (channel == NULL) return LW_STATUS_INVALID_ARGUMENT;
            if (state != NULL && op->data.binary.channels > 0u) {
                uint64_t ch_threshold = LW_X64_DET_ELEMENTWISE_SHARD_ELEMS /
                                        op->data.binary.channels;
                lw_status sharded = run_sharded_elementwise(
                    instance, op, state, LW_X64_DET_SHARD_BINARY_CHANNEL, input, channel, output,
                    op->data.binary.pixels, ch_threshold < 2u ? 2u : ch_threshold, 0);
                if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
            }
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_binary_channel_nhwc_f32(binary_operation(op->data.binary.operation),
                input, channel, output, op->data.binary.pixels,
                op->data.binary.channels, 0);
#else
            lw_avx2_binary_channel_f32(binary_operation(op->data.binary.operation), input, channel,
                                       output, op->data.binary.pixels, op->data.binary.channels, 0);
#endif
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
    case LW_X64_DET_OP_RELU: case LW_X64_DET_OP_ERF: case LW_X64_DET_OP_GELU:
    case LW_X64_DET_OP_HARD_SIGMOID:
        input = offset_ptr(instance, op->data.unary.input_offset);
        output = offset_ptr(instance, op->data.unary.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->kind == LW_X64_DET_OP_RELU) {
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_relu_f32(input, output, op->data.unary.element_count);
#else
            lw_avx2_relu_contiguous_f32(input, output, op->data.unary.element_count);
#endif
        } else if (op->kind == LW_X64_DET_OP_ERF) {
            lw_avx2_erf_f32(input, output, op->data.unary.element_count);
        } else if (op->kind == LW_X64_DET_OP_GELU) {
            lw_avx2_gelu_f32(input, output, op->data.unary.element_count);
        } else {
            lw_avx2_hard_sigmoid_contiguous_f32(input, output, op->data.unary.element_count,
                                                op->data.unary.alpha, op->data.unary.beta);
        }
        return LW_STATUS_OK;
    case LW_X64_DET_OP_REDUCE_MEAN:
        input = offset_ptr(instance, op->data.reduce.input_offset);
        output = offset_ptr(instance, op->data.reduce.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (state != NULL && op->data.reduce.channels > 1u &&
            (uint64_t)op->data.reduce.batch * op->data.reduce.channels *
                    op->data.reduce.height * op->data.reduce.width >=
                LW_X64_DET_ELEMENTWISE_SHARD_ELEMS) {
            lw_status sharded = run_sharded_elementwise(
                instance, op, state, LW_X64_DET_SHARD_REDUCE_CHANNELS, input, NULL,
                output, op->data.reduce.channels, 2u, 0);
            if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
        }
        if (op->data.reduce.layout == LW_X64_DET_LAYOUT_NCHW) {
            uint32_t channel;
            for (channel = 0u; channel < op->data.reduce.channels; ++channel) {
                const float* source = input + (size_t)channel * op->data.reduce.height *
                                              op->data.reduce.width;
                float sum = 0.0f;
                uint32_t spatial;
                for (spatial = 0u; spatial < op->data.reduce.height * op->data.reduce.width;
                     ++spatial) {
                    sum += source[spatial];
                }
                output[channel] = sum / (float)(op->data.reduce.height * op->data.reduce.width);
            }
        } else {
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
            lw_wasm128_reduce_mean_hw_f32(input, output, op->data.reduce.batch,
                op->data.reduce.height, op->data.reduce.width,
                op->data.reduce.channels);
#else
            lw_avx2_nhwc_reduce_mean_hw_f32(input, output, op->data.reduce.batch,
                                            op->data.reduce.height, op->data.reduce.width,
                                            op->data.reduce.channels);
#endif
        }
        return LW_STATUS_OK;
    case LW_X64_DET_OP_AVG_POOL: case LW_X64_DET_OP_MAX_POOL:
        input = offset_ptr(instance, op->data.pool.input_offset);
        output = offset_ptr(instance, op->data.pool.output_offset);
        if (input == NULL || output == NULL) return LW_STATUS_INVALID_ARGUMENT;
        if (op->data.pool.layout == LW_X64_DET_LAYOUT_NCHW) {
            if (state != NULL && op->data.pool.input_dimensions[0] == 1 &&
                op->data.pool.input_dimensions[1] > 1 &&
                (uint64_t)op->data.pool.input_dimensions[1] *
                        (uint64_t)op->data.pool.input_dimensions[2] *
                        (uint64_t)op->data.pool.input_dimensions[3] >=
                    LW_X64_DET_ELEMENTWISE_SHARD_ELEMS) {
                lw_status sharded = run_sharded_elementwise(
                    instance, op, state, LW_X64_DET_SHARD_POOL_NCHW_CHANNELS, input, NULL,
                    output, (uint64_t)op->data.pool.input_dimensions[1], 2u, 0);
                if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
            }
            return op->kind == LW_X64_DET_OP_MAX_POOL
                ? lw_scalar_max_pool2d_f32(input, output, op->data.pool.input_dimensions,
                                           op->data.pool.output_dimensions, op->data.pool.kernel,
                                           op->data.pool.strides, op->data.pool.pads, 0u)
                : lw_scalar_average_pool2d_f32(input, output, op->data.pool.input_dimensions,
                                               op->data.pool.output_dimensions, op->data.pool.kernel,
                                               op->data.pool.strides, op->data.pool.pads, 0u,
                                               op->data.pool.count_include_pad);
        }
        if (state != NULL && op->data.pool.input_dimensions[0] == 1 &&
            op->data.pool.strides[0] >= op->data.pool.pads[0] &&
            op->data.pool.output_dimensions[1] > 1 &&
            (uint64_t)op->data.pool.input_dimensions[1] *
                    (uint64_t)op->data.pool.input_dimensions[2] *
                    (uint64_t)op->data.pool.input_dimensions[3] >=
                LW_X64_DET_ELEMENTWISE_SHARD_ELEMS) {
            lw_status sharded = run_sharded_elementwise(
                instance, op, state, LW_X64_DET_SHARD_POOL_NHWC_ROWS, input, NULL, output,
                (uint64_t)op->data.pool.output_dimensions[1], 2u, 0);
            if (sharded != LW_STATUS_UNSUPPORTED) return sharded;
        }
#if defined(LW_WASM_COMPILED_DET) && defined(LW_WASM_REC_SIMD_KERNELS)
        lw_wasm128_pool_nhwc_f32(input, output,
            (uint32_t)op->data.pool.input_dimensions[0],
            (uint32_t)op->data.pool.input_dimensions[1],
            (uint32_t)op->data.pool.input_dimensions[2],
            (uint32_t)op->data.pool.output_dimensions[1],
            (uint32_t)op->data.pool.output_dimensions[2],
            (uint32_t)op->data.pool.input_dimensions[3],
            (uint32_t)op->data.pool.kernel[0],
            (uint32_t)op->data.pool.kernel[1],
            (uint32_t)op->data.pool.strides[0],
            (uint32_t)op->data.pool.strides[1],
            (uint32_t)op->data.pool.pads[0],
            (uint32_t)op->data.pool.pads[1],
            op->data.pool.count_include_pad, op->data.pool.is_max);
#else
        lw_avx2_nhwc_pool_f32(input, output, (uint32_t)op->data.pool.input_dimensions[0],
                              (uint32_t)op->data.pool.input_dimensions[1],
                              (uint32_t)op->data.pool.input_dimensions[2],
                              (uint32_t)op->data.pool.output_dimensions[1],
                              (uint32_t)op->data.pool.output_dimensions[2],
                              (uint32_t)op->data.pool.input_dimensions[3],
                              (uint32_t)op->data.pool.kernel[0], (uint32_t)op->data.pool.kernel[1],
                              (uint32_t)op->data.pool.strides[0], (uint32_t)op->data.pool.strides[1],
                              (uint32_t)op->data.pool.pads[0], (uint32_t)op->data.pool.pads[1],
                              op->data.pool.count_include_pad, op->data.pool.is_max);
#endif
        return LW_STATUS_OK;
    default: return LW_STATUS_UNSUPPORTED;
    }
}

lw_status lw_x64_det_instance_create(const lw_x64_det_program* program,
                                     lw_x64_det_instance** out, lw_error* error) {
    lw_x64_det_instance* instance;
    if (out == NULL || program == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET instance requires a compiled program");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    instance = (lw_x64_det_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET instance allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    instance->program = program;
    if (program->arena_bytes > 0u) {
        instance->arena = (uint8_t*)det_aligned_alloc(64u, (size_t)program->arena_bytes);
    }
    if (program->scratch_bytes > 0u) {
        instance->scratch = (uint8_t*)det_aligned_alloc(64u, (size_t)program->scratch_bytes);
        instance->shard_scratch_slice_bytes = program->scratch_bytes;
        instance->shard_scratch = (uint8_t*)det_aligned_alloc(
            64u, (size_t)program->scratch_bytes * LW_PARALLEL_MAX_WORKERS);
    }
    if ((program->arena_bytes > 0u && instance->arena == NULL) ||
        (program->scratch_bytes > 0u &&
         (instance->scratch == NULL || instance->shard_scratch == NULL))) {
        lw_x64_det_instance_free(instance);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET instance workspace allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    *out = instance;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_x64_det_instance_free(lw_x64_det_instance* instance) {
    if (instance == NULL) return;
    det_aligned_free(instance->shard_scratch);
    det_aligned_free(instance->scratch);
    det_aligned_free(instance->arena);
    free(instance);
}

float* lw_x64_det_instance_input(lw_x64_det_instance* instance, uint64_t* element_count) {
    const lw_x64_det_value* value;
    if (element_count != NULL) *element_count = 0u;
    if (instance == NULL || instance->program == NULL ||
        instance->program->input_value >= instance->program->value_count) {
        return NULL;
    }
    value = &instance->program->values[instance->program->input_value];
    if (element_count != NULL) *element_count = value->bytes / sizeof(float);
    return offset_ptr(instance, value->offset);
}

static uint32_t det_conv_class(const lw_model* model, uint32_t node_index) {
    const uint8_t* node = model->bytes + (size_t)model->node_offset +
                          (size_t)node_index * LWM_V0_NODE_SIZE;
    uint64_t param_offset = lwm_read_u64(node + 56u);
    const uint8_t* params;
    int32_t kernel_height;
    int32_t kernel_width;
    int32_t stride_height;
    int32_t stride_width;
    uint32_t group;
    if (param_offset == 0u) return LW_EXECUTION_PROFILE_CONV_OTHER;
    params = model->bytes + (size_t)param_offset;
    kernel_height = lwm_read_i32(params + 8u);
    kernel_width = lwm_read_i32(params + 12u);
    stride_height = lwm_read_i32(params + 16u);
    stride_width = lwm_read_i32(params + 20u);
    group = lwm_read_u32(params + 4u);
    if (kernel_height == 1 && kernel_width == 1) return LW_EXECUTION_PROFILE_CONV_1X1;
    if (kernel_height == 3 && kernel_width == 3 && group > 1u) {
        return LW_EXECUTION_PROFILE_CONV_DEPTHWISE_3X3;
    }
    if (kernel_height == 3 && kernel_width == 3 && stride_height == 2 && stride_width == 2) {
        return LW_EXECUTION_PROFILE_CONV_STRIDE2_3X3;
    }
    if (kernel_height == 3 && kernel_width == 3) return LW_EXECUTION_PROFILE_CONV_3X3;
    return LW_EXECUTION_PROFILE_CONV_OTHER;
}

/* Attribute one physical op's elapsed time to its semantic span, mirroring
 * the canonical executor's accounting (fused GELU work lands on the Erf node
 * with 1-ns placeholders for the other four). */
static void profile_physical_op(const lw_model* model, lw_execution_profile* profile,
                                const lw_x64_det_op* op, uint64_t elapsed) {
    uint32_t offset;
    uint32_t begin = op->semantic_begin;
    uint32_t count = op->semantic_count;
    if (profile == NULL || count == 0u) return;
    if (count == 5u && begin + 5u <= model->info.node_count) {
        for (offset = 0u; offset < 5u; ++offset) {
            uint32_t node_index = begin + offset;
            const uint8_t* node = model->bytes + (size_t)model->node_offset +
                                  (size_t)node_index * LWM_V0_NODE_SIZE;
            uint32_t operation = (uint32_t)lwm_read_u16(node);
            uint64_t node_elapsed = offset == 1u ? (elapsed == 0u ? 1u : elapsed) : 1u;
            if (operation < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY &&
                profile->operator_nanoseconds[operation] <= UINT64_MAX - node_elapsed &&
                profile->operator_invocations[operation] != UINT64_MAX) {
                profile->operator_nanoseconds[operation] += node_elapsed;
                profile->operator_invocations[operation] += 1u;
            }
            if (node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
                profile->node_nanoseconds[node_index] <= UINT64_MAX - node_elapsed &&
                profile->node_invocations[node_index] != UINT64_MAX) {
                profile->node_nanoseconds[node_index] += node_elapsed;
                profile->node_invocations[node_index] += 1u;
            }
        }
        return;
    }
    for (offset = 0u; offset < count && begin + offset < model->info.node_count; ++offset) {
        uint32_t node_index = begin + offset;
        const uint8_t* node = model->bytes + (size_t)model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        uint32_t operation = (uint32_t)lwm_read_u16(node);
        if (operation < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY &&
            profile->operator_nanoseconds[operation] <= UINT64_MAX - elapsed &&
            profile->operator_invocations[operation] != UINT64_MAX) {
            profile->operator_nanoseconds[operation] += elapsed;
            profile->operator_invocations[operation] += 1u;
        }
        if (node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY &&
            profile->node_nanoseconds[node_index] <= UINT64_MAX - elapsed &&
            profile->node_invocations[node_index] != UINT64_MAX) {
            profile->node_nanoseconds[node_index] += elapsed;
            profile->node_invocations[node_index] += 1u;
        }
        if (operation == LW_OP_CONV && node_index < LW_EXECUTION_PROFILE_NODE_CAPACITY) {
            uint32_t conv_class = det_conv_class(model, node_index);
            if (profile->conv_class_nanoseconds[conv_class] <= UINT64_MAX - elapsed &&
                profile->conv_class_invocations[conv_class] != UINT64_MAX) {
                profile->conv_class_nanoseconds[conv_class] += elapsed;
                profile->conv_class_invocations[conv_class] += 1u;
            }
        }
    }
}

static lw_status run_backbone_ops(lw_x64_det_instance* instance,
                                  const lw_x64_det_run_state* state, lw_error* error) {
    uint32_t i;
    lw_status status;
    if (instance == NULL || instance->program == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET instance is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0u; i < instance->program->op_count; ++i) {
        status = execute_op(instance, &instance->program->ops[i], state, error);
        if (status != LW_STATUS_OK) {
            if (error != NULL && error->message[0] == '\0') {
                char message[96];
                (void)snprintf(message, sizeof(message), "DET physical op %u kind %u failed",
                               (unsigned)i, (unsigned)instance->program->ops[i].kind);
                lw_set_error(error, status, message);
            }
            return status;
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_x64_det_instance_run_op(lw_x64_det_instance* instance, uint32_t op_index,
                                     lw_error* error) {
    if (instance == NULL || instance->program == NULL ||
        op_index >= instance->program->op_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET physical op index is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return execute_op(instance, &instance->program->ops[op_index], NULL, error);
}

lw_status lw_x64_det_instance_run_op_ex(lw_x64_det_instance* instance, uint32_t op_index,
                                        lw_thread_pool* pool, uint32_t worker_count,
                                        lw_error* error) {
    lw_x64_det_run_state state;
    if (instance == NULL || instance->program == NULL ||
        op_index >= instance->program->op_count) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET physical op index is invalid");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    state.instance = instance;
    state.pool = pool;
    state.worker_count = worker_count;
    return execute_op(instance, &instance->program->ops[op_index], &state, error);
}

lw_status lw_x64_det_instance_run(lw_x64_det_instance* instance, lw_error* error) {
    lw_status status = run_backbone_ops(instance, NULL, error);
    if (status == LW_STATUS_OK) lw_set_error(error, LW_STATUS_OK, "");
    return status;
}

lw_status lw_x64_det_instance_run_profiled(lw_x64_det_instance* instance, float* output,
                                           uint64_t output_element_count,
                                           lw_execution_profile* profile, lw_error* error) {
    return lw_x64_det_instance_run_profiled_ex(instance, output, output_element_count, NULL, 1u,
                                               profile, error);
}

/* Workers actually used by a physical op, recomputed exactly like the
 * dispatch so the thread histograms match the executed shards. */
static uint32_t det_op_workers(const lw_x64_det_run_state* state, const lw_x64_det_op* op) {
    switch (op->kind) {
    case LW_X64_DET_OP_POINTWISE:
    case LW_X64_DET_OP_DENSE:
        return det_shard_workers(state, conv_macs(&op->data.conv),
                                 LW_X64_DET_CONV_SHARD_MACS);
    case LW_X64_DET_OP_DEPTHWISE:
        return det_shard_workers(state, conv_macs(&op->data.conv),
                                 LW_X64_DET_DEPTHWISE_SHARD_MACS);
    case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC:
        return det_shard_workers(
            state, (uint64_t)op->data.conv_transpose.input_dimensions[1] *
                       op->data.conv_transpose.output_dimensions[1] * 4u *
                       op->data.conv_transpose.output_dimensions[2] *
                       op->data.conv_transpose.output_dimensions[3],
            LW_X64_DET_CONV_SHARD_MACS);
    case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1:
        return det_shard_workers(
            state, (uint64_t)op->data.conv_transpose.input_dimensions[1] *
                       op->data.conv_transpose.output_dimensions[1] * 4u *
                       op->data.conv_transpose.output_dimensions[2] *
                       op->data.conv_transpose.output_dimensions[3],
            LW_X64_DET_DEPTHWISE_SHARD_MACS);
    default:
        return 1u;
    }
}

lw_status lw_x64_det_instance_run_profiled_ex(lw_x64_det_instance* instance, float* output,
                                              uint64_t output_element_count,
                                              lw_thread_pool* pool, uint32_t worker_count,
                                              lw_execution_profile* profile, lw_error* error) {
    const lw_x64_det_program* program;
    const lw_x64_det_value* output_value;
    lw_x64_det_run_state state;
    uint64_t total_ns = 0u;
    lw_status status;
    uint32_t i;
    if (instance == NULL || instance->program == NULL || output == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET instance, program and output are required");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    program = instance->program;
    if (program->output_value >= program->value_count) {
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "DET program has no graph output");
        return LW_STATUS_INVALID_SHAPE;
    }
    output_value = &program->values[program->output_value];
    if (output_element_count < output_value->bytes / sizeof(float)) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET output buffer is too small");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    state.instance = instance;
    state.pool = pool;
    state.worker_count = worker_count;
    memset(&instance->profile, 0, sizeof(instance->profile));
    if (profile != NULL) {
        profile->layout_candidate_nodes += program->nhwc_effective_nodes +
                                           program->nchw_effective_nodes;
        profile->layout_selected_nodes += program->nhwc_effective_nodes;
        profile->layout_fallback_nodes += program->nchw_effective_nodes;
        profile->layout_analysis_runs += 1u;
    }
    for (i = 0u; i < program->op_count; ++i) {
        const lw_x64_det_op* op = &program->ops[i];
        uint64_t started = profile != NULL && profile->clock != NULL
            ? profile->clock(profile->clock_context) : 0u;
        status = execute_op(instance, op, &state, error);
        if (status != LW_STATUS_OK) {
            if (error != NULL && error->message[0] == '\0') {
                char message[96];
                (void)snprintf(message, sizeof(message), "DET physical op %u kind %u failed",
                               (unsigned)i, (unsigned)op->kind);
                lw_set_error(error, status, message);
            }
            return status;
        }
        if (profile != NULL && profile->clock != NULL) {
            uint64_t finished = profile->clock(profile->clock_context);
            uint64_t elapsed = finished >= started ? finished - started : 0u;
            uint32_t op_workers = det_op_workers(&state, op);
            total_ns += elapsed;
            switch (op->kind) {
            case LW_X64_DET_OP_POINTWISE: instance->profile.pointwise_ns += elapsed; break;
            case LW_X64_DET_OP_DENSE: instance->profile.dense_ns += elapsed; break;
            case LW_X64_DET_OP_DEPTHWISE: instance->profile.depthwise_ns += elapsed; break;
            case LW_X64_DET_OP_CONV_NCHW: instance->profile.conv_nchw_ns += elapsed; break;
            case LW_X64_DET_OP_ADD:
            case LW_X64_DET_OP_MUL:
            case LW_X64_DET_OP_DIV: instance->profile.binary_ns += elapsed; break;
            case LW_X64_DET_OP_RELU:
            case LW_X64_DET_OP_ERF:
            case LW_X64_DET_OP_GELU:
            case LW_X64_DET_OP_HARD_SIGMOID: instance->profile.unary_ns += elapsed; break;
            case LW_X64_DET_OP_REDUCE_MEAN: instance->profile.reduce_ns += elapsed; break;
            case LW_X64_DET_OP_AVG_POOL:
            case LW_X64_DET_OP_MAX_POOL: instance->profile.pool_ns += elapsed; break;
            case LW_X64_DET_OP_LAYOUT_CONVERT:
                instance->profile.convert_ns += elapsed;
                profile->layout_transform_nanoseconds += elapsed;
                profile->layout_transform_invocations += 1u;
                profile->layout_transform_bytes +=
                    (uint64_t)op->data.convert.batch * op->data.convert.channels *
                    op->data.convert.height * op->data.convert.width * sizeof(float);
                break;
            case LW_X64_DET_OP_CONV_TRANSPOSE_NCHW:
            case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC:
            case LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1:
                instance->profile.conv_transpose_ns += elapsed;
                break;
            case LW_X64_DET_OP_RESIZE_NCHW:
            case LW_X64_DET_OP_RESIZE_NHWC:
                instance->profile.resize_ns += elapsed;
                break;
            case LW_X64_DET_OP_SIGMOID_NCHW: instance->profile.sigmoid_ns += elapsed; break;
            case LW_X64_DET_OP_CONCAT_NCHW:
            case LW_X64_DET_OP_CONCAT_NHWC:
                instance->profile.concat_ns += elapsed;
                break;
            default: break;
            }
            if (op_workers >= 1u && op_workers < LW_EXECUTION_PROFILE_THREAD_HISTOGRAM_CAPACITY) {
                if (op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NHWC ||
                    op->kind == LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1) {
                    profile->conv_transpose_thread_histogram[op_workers] += 1u;
                } else if (op->kind == LW_X64_DET_OP_POINTWISE ||
                           op->kind == LW_X64_DET_OP_DENSE ||
                           op->kind == LW_X64_DET_OP_DEPTHWISE ||
                           op->kind == LW_X64_DET_OP_CONV_NCHW) {
                    profile->conv_thread_histogram[op_workers] += 1u;
                }
            }
            profile_physical_op(program->model, profile, op, elapsed);
        }
    }
    instance->profile.total_ns = total_ns;
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_COMPILED_DET)
    if (profile != NULL && profile->clock != NULL &&
        getenv("LW_WASM_OCR_PROFILE") != NULL) {
        const lw_x64_det_profile* det_profile = &instance->profile;
        const double to_ms = 1.0 / 1000000.0;
        (void)fprintf(stderr,
            "LW_WASM_DET_PROFILE total=%.3f pointwise=%.3f dense=%.3f "
            "depthwise=%.3f convtranspose=%.3f binary=%.3f pool=%.3f "
            "concat=%.3f resize=%.3f other=%.3f\n",
            det_profile->total_ns * to_ms,
            det_profile->pointwise_ns * to_ms,
            det_profile->dense_ns * to_ms,
            det_profile->depthwise_ns * to_ms,
            det_profile->conv_transpose_ns * to_ms,
            det_profile->binary_ns * to_ms,
            det_profile->pool_ns * to_ms,
            det_profile->concat_ns * to_ms,
            det_profile->resize_ns * to_ms,
            (det_profile->total_ns - det_profile->pointwise_ns -
             det_profile->dense_ns - det_profile->depthwise_ns -
             det_profile->conv_transpose_ns - det_profile->binary_ns -
             det_profile->pool_ns - det_profile->concat_ns -
             det_profile->resize_ns) * to_ms);
    }
#endif
    if (output_value->layout == LW_X64_DET_LAYOUT_NCHW || output_value->alt_producer < 0) {
        memcpy(output, offset_ptr(instance, output_value->offset),
               (size_t)(output_value->bytes));
    } else {
        memcpy(output, offset_ptr(instance, output_value->alt_offset),
               (size_t)(output_value->bytes));
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
