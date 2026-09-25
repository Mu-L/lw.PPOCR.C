#include "../kernels/nhwc_internal.h"

#include <stddef.h>
#include <wasm_simd128.h>

static int valid_shape(const lw_nhwc_convtranspose_desc* desc) {
    return desc != NULL && desc->batch != 0u && desc->input_channels != 0u &&
        desc->input_height != 0u && desc->input_width != 0u &&
        desc->output_channels != 0u &&
        desc->input_height <= UINT32_MAX / 2u &&
        desc->input_width <= UINT32_MAX / 2u &&
        desc->output_height == desc->input_height * 2u &&
        desc->output_width == desc->input_width * 2u;
}

static v128_t relu_vector(v128_t value) {
    v128_t zero = wasm_f32x4_splat(0.0f);
    return wasm_v128_bitselect(value, zero, wasm_f32x4_gt(value, zero));
}

lw_status lw_wasm128_nhwc_convtranspose2x2_s2_f32(
    const float* input, const float* packed_weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc) {
    uint32_t blocks;
    if (input == NULL || packed_weights == NULL || output == NULL ||
        !valid_shape(desc) || (desc->output_channels & 7u) != 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    blocks = (desc->output_channels + LW_NHWC_OC_BLOCK - 1u) / LW_NHWC_OC_BLOCK;
    for (uint32_t n = 0u; n < desc->batch; ++n) {
        for (uint32_t iy = 0u; iy < desc->input_height; ++iy) {
            for (uint32_t ix = 0u; ix < desc->input_width; ++ix) {
                const float* source = input +
                    (((size_t)n * desc->input_height + iy) * desc->input_width + ix) *
                    desc->input_channels;
                for (uint32_t tap = 0u; tap < 4u; ++tap) {
                    uint32_t oy = iy * 2u + (tap >> 1u);
                    uint32_t ox = ix * 2u + (tap & 1u);
                    float* destination = output +
                        (((size_t)n * desc->output_height + oy) * desc->output_width + ox) *
                        desc->output_channels;
                    for (uint32_t block = 0u; block < blocks; ++block) {
                        uint32_t base = block * LW_NHWC_OC_BLOCK;
                        uint32_t valid = desc->output_channels - base;
                        v128_t acc[4];
                        float temporary[LW_NHWC_OC_BLOCK];
                        if (valid > LW_NHWC_OC_BLOCK) valid = LW_NHWC_OC_BLOCK;
                        for (uint32_t lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane)
                            temporary[lane] = epilogue != NULL && epilogue->bias != NULL &&
                                lane < valid ? epilogue->bias[base + lane] : 0.0f;
                        for (uint32_t vector = 0u; vector < 4u; ++vector)
                            acc[vector] = wasm_v128_load(temporary + vector * 4u);
                        for (uint32_t ic = 0u; ic < desc->input_channels; ++ic) {
                            const float* weight = packed_weights +
                                (((size_t)tap * blocks + block) * desc->input_channels + ic) *
                                LW_NHWC_OC_BLOCK;
                            v128_t sample = wasm_f32x4_splat(source[ic]);
                            for (uint32_t vector = 0u; vector < 4u; ++vector) {
                                acc[vector] = wasm_f32x4_add(acc[vector],
                                    wasm_f32x4_mul(sample,
                                        wasm_v128_load(weight + vector * 4u)));
                            }
                        }
                        for (uint32_t vector = 0u; vector < 4u; ++vector) {
                            if (epilogue != NULL &&
                                epilogue->activation == LW_NHWC_ACT_RELU)
                                acc[vector] = relu_vector(acc[vector]);
                        }
                        if (valid == LW_NHWC_OC_BLOCK) {
                            for (uint32_t vector = 0u; vector < 4u; ++vector)
                                wasm_v128_store(destination + base + vector * 4u, acc[vector]);
                        } else {
                            for (uint32_t vector = 0u; vector < 4u; ++vector)
                                wasm_v128_store(temporary + vector * 4u, acc[vector]);
                            for (uint32_t lane = 0u; lane < valid; ++lane)
                                destination[base + lane] = temporary[lane];
                        }
                    }
                }
            }
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_wasm128_nhwc_convtranspose2x2_s2_c1_f32(
    const float* input, const float* weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc) {
    if (input == NULL || weights == NULL || output == NULL ||
        !valid_shape(desc) || desc->output_channels != 1u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t n = 0u; n < desc->batch; ++n) {
        for (uint32_t iy = 0u; iy < desc->input_height; ++iy) {
            for (uint32_t ix = 0u; ix < desc->input_width; ++ix) {
                const float* source = input +
                    (((size_t)n * desc->input_height + iy) * desc->input_width + ix) *
                    desc->input_channels;
                v128_t acc = wasm_f32x4_splat(
                    epilogue != NULL && epilogue->bias != NULL ? epilogue->bias[0] : 0.0f);
                float taps[4];
                for (uint32_t ic = 0u; ic < desc->input_channels; ++ic) {
                    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(
                        wasm_f32x4_splat(source[ic]), wasm_v128_load(weights + (size_t)ic * 4u)));
                }
                if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_RELU)
                    acc = relu_vector(acc);
                wasm_v128_store(taps, acc);
                for (uint32_t tap = 0u; tap < 4u; ++tap) {
                    uint32_t oy = iy * 2u + (tap >> 1u);
                    uint32_t ox = ix * 2u + (tap & 1u);
                    output[((size_t)n * desc->output_height + oy) *
                        desc->output_width + ox] = taps[tap];
                }
            }
        }
    }
    return LW_STATUS_OK;
}
