#include "nhwc_internal.h"
#include "simd_kernels.h"
#include "wasm128_epilogue_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

static float apply_activation(float value, uint16_t activation) {
    if (activation == LW_NHWC_ACT_RELU) return value > 0.0f ? value : 0.0f;
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        float gate = value + 3.0f;
        if (gate < 0.0f) gate = 0.0f;
        else if (gate > 6.0f) gate = 6.0f;
        return value * gate / 6.0f;
    }
    if (activation == LW_NHWC_ACT_GELU) {
        return lw_nhwc_gelu_scalar_exact_f32(value);
    }
    return value;
}

/* Keep the shared OC16 pack format. One weight load serves up to four
 * independent pixels; standard SIMD128 mul/add preserves non-FMA semantics. */
void lw_wasm128_nhwc_pointwise_4x16_f32(const float* input,
                                        const float* packed_weights,
                                        const lw_nhwc_epilogue* epilogue,
                                        float* output, uint32_t pixels,
                                        uint32_t input_channels,
                                        uint32_t output_channels) {
    for (uint32_t pixel = 0u; pixel < pixels; pixel += 4u) {
        uint32_t rows = pixels - pixel;
        if (rows > 4u) rows = 4u;
        for (uint32_t oc = 0u; oc < output_channels; oc += LW_NHWC_OC_BLOCK) {
            uint32_t lanes = output_channels - oc;
            float bias_values[LW_NHWC_OC_BLOCK] = {0.0f};
            v128_t initial[4];
            v128_t accumulators[4][4];
            const float* weights = packed_weights +
                (size_t)(oc / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            if (lanes > LW_NHWC_OC_BLOCK) lanes = LW_NHWC_OC_BLOCK;
            if (epilogue != NULL && epilogue->bias != NULL) {
                for (uint32_t lane = 0u; lane < lanes; ++lane) {
                    bias_values[lane] = epilogue->bias[oc + lane];
                }
            }
            for (uint32_t group = 0u; group < 4u; ++group) {
                initial[group] = wasm_v128_load(bias_values + group * 4u);
            }
            for (uint32_t row = 0u; row < rows; ++row) {
                for (uint32_t group = 0u; group < 4u; ++group) {
                    accumulators[row][group] = initial[group];
                }
            }
            for (uint32_t ic = 0u; ic < input_channels; ++ic) {
                const float* packed = weights + (size_t)ic * LW_NHWC_OC_BLOCK;
                v128_t weight[4];
                for (uint32_t group = 0u; group < 4u; ++group) {
                    weight[group] = wasm_v128_load(packed + group * 4u);
                }
                for (uint32_t row = 0u; row < rows; ++row) {
                    v128_t value = wasm_f32x4_splat(
                        input[(size_t)(pixel + row) * input_channels + ic]);
                    for (uint32_t group = 0u; group < 4u; ++group) {
                        accumulators[row][group] = wasm_f32x4_add(
                            accumulators[row][group],
                            wasm_f32x4_mul(value, weight[group]));
                    }
                }
            }
            for (uint32_t row = 0u; row < rows; ++row) {
                size_t output_base = (size_t)(pixel + row) * output_channels + oc;
                if (lanes == LW_NHWC_OC_BLOCK) {
                    v128_t values[4] = {
                        accumulators[row][0], accumulators[row][1],
                        accumulators[row][2], accumulators[row][3]
                    };
                    const float* residual = epilogue == NULL || epilogue->residual == NULL
                        ? NULL : epilogue->residual + output_base;
                    lw_wasm128_apply_epilogue_oc16(values, epilogue, oc, residual, 1);
                    for (uint32_t group = 0u; group < 4u; ++group) {
                        wasm_v128_store(output + output_base + group * 4u, values[group]);
                    }
                } else {
                    float sums[LW_NHWC_OC_BLOCK];
                    for (uint32_t group = 0u; group < 4u; ++group) {
                        wasm_v128_store(sums + group * 4u, accumulators[row][group]);
                    }
                    if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_GELU) {
                        for (uint32_t lane = 0u; lane < lanes; ++lane) {
                            if (epilogue->post_bias != NULL)
                                sums[lane] += epilogue->post_bias[oc + lane];
                            if (epilogue->residual != NULL)
                                sums[lane] += epilogue->residual[output_base + lane];
                        }
                        lw_wasm128_gelu_f32(sums, sums, LW_NHWC_OC_BLOCK);
                        for (uint32_t lane = 0u; lane < lanes; ++lane)
                            output[output_base + lane] = sums[lane];
                        continue;
                    }
                    for (uint32_t lane = 0u; lane < lanes; ++lane) {
                        float value = sums[lane];
                        if (epilogue != NULL) {
                            if (epilogue->post_bias != NULL) {
                                value += epilogue->post_bias[oc + lane];
                            }
                            if (epilogue->residual != NULL) {
                                value += epilogue->residual[output_base + lane];
                            }
                            value = apply_activation(value, epilogue->activation);
                        }
                        output[output_base + lane] = value;
                    }
                }
            }
        }
    }
}

static inline void wasm_pointwise_store_row(
    v128_t value0, v128_t value1, v128_t value2, v128_t value3,
    const lw_nhwc_epilogue* epilogue, float* output, size_t output_base,
    uint32_t oc, uint32_t lanes) {
    if (lanes == LW_NHWC_OC_BLOCK) {
        v128_t values[4] = {value0, value1, value2, value3};
        const float* residual = epilogue == NULL || epilogue->residual == NULL
            ? NULL : epilogue->residual + output_base;
        lw_wasm128_apply_epilogue_oc16(values, epilogue, oc, residual, 1);
        wasm_v128_store(output + output_base, values[0]);
        wasm_v128_store(output + output_base + 4u, values[1]);
        wasm_v128_store(output + output_base + 8u, values[2]);
        wasm_v128_store(output + output_base + 12u, values[3]);
        return;
    }
    {
        float sums[LW_NHWC_OC_BLOCK];
        wasm_v128_store(sums, value0);
        wasm_v128_store(sums + 4u, value1);
        wasm_v128_store(sums + 8u, value2);
        wasm_v128_store(sums + 12u, value3);
        if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_GELU) {
            for (uint32_t lane = 0u; lane < lanes; ++lane) {
                if (epilogue->post_bias != NULL)
                    sums[lane] += epilogue->post_bias[oc + lane];
                if (epilogue->residual != NULL)
                    sums[lane] += epilogue->residual[output_base + lane];
            }
            lw_wasm128_gelu_f32(sums, sums, LW_NHWC_OC_BLOCK);
            for (uint32_t lane = 0u; lane < lanes; ++lane)
                output[output_base + lane] = sums[lane];
            return;
        }
        for (uint32_t lane = 0u; lane < lanes; ++lane) {
            float value = sums[lane];
            if (epilogue != NULL) {
                if (epilogue->post_bias != NULL)
                    value += epilogue->post_bias[oc + lane];
                if (epilogue->residual != NULL)
                    value += epilogue->residual[output_base + lane];
                value = apply_activation(value, epilogue->activation);
            }
            output[output_base + lane] = value;
        }
    }
}

/* Two pixels share each OC16 weight block while keeping only eight SIMD
 * accumulators live. The IC accumulation order and non-FMA mul/add sequence
 * are identical to the existing 4x16 kernel. */
void lw_wasm128_nhwc_pointwise_2x16_f32(const float* input,
                                        const float* packed_weights,
                                        const lw_nhwc_epilogue* epilogue,
                                        float* output, uint32_t pixels,
                                        uint32_t input_channels,
                                        uint32_t output_channels) {
    uint32_t pixel = 0u;
    for (; pixel + 1u < pixels; pixel += 2u) {
        const float* input0 = input + (size_t)pixel * input_channels;
        const float* input1 = input + (size_t)(pixel + 1u) * input_channels;
        for (uint32_t oc = 0u; oc < output_channels; oc += LW_NHWC_OC_BLOCK) {
            uint32_t lanes = output_channels - oc;
            v128_t bias0, bias1, bias2, bias3;
            const float* weights = packed_weights +
                (size_t)(oc / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            if (lanes > LW_NHWC_OC_BLOCK) lanes = LW_NHWC_OC_BLOCK;
            if (lanes == LW_NHWC_OC_BLOCK && epilogue != NULL &&
                epilogue->bias != NULL) {
                bias0 = wasm_v128_load(epilogue->bias + oc);
                bias1 = wasm_v128_load(epilogue->bias + oc + 4u);
                bias2 = wasm_v128_load(epilogue->bias + oc + 8u);
                bias3 = wasm_v128_load(epilogue->bias + oc + 12u);
            } else if (epilogue != NULL && epilogue->bias != NULL) {
                float tail_bias[LW_NHWC_OC_BLOCK] = {0.0f};
                for (uint32_t lane = 0u; lane < lanes; ++lane)
                    tail_bias[lane] = epilogue->bias[oc + lane];
                bias0 = wasm_v128_load(tail_bias);
                bias1 = wasm_v128_load(tail_bias + 4u);
                bias2 = wasm_v128_load(tail_bias + 8u);
                bias3 = wasm_v128_load(tail_bias + 12u);
            } else {
                const v128_t zero = wasm_f32x4_splat(0.0f);
                bias0 = zero;
                bias1 = zero;
                bias2 = zero;
                bias3 = zero;
            }

            v128_t a00 = bias0, a01 = bias1, a02 = bias2, a03 = bias3;
            v128_t a10 = bias0, a11 = bias1, a12 = bias2, a13 = bias3;
            for (uint32_t ic = 0u; ic < input_channels; ++ic) {
                const float* packed = weights + (size_t)ic * LW_NHWC_OC_BLOCK;
                const v128_t x0 = wasm_f32x4_splat(input0[ic]);
                const v128_t x1 = wasm_f32x4_splat(input1[ic]);
                {
                    const v128_t w = wasm_v128_load(packed);
                    a00 = wasm_f32x4_add(a00, wasm_f32x4_mul(x0, w));
                    a10 = wasm_f32x4_add(a10, wasm_f32x4_mul(x1, w));
                }
                {
                    const v128_t w = wasm_v128_load(packed + 4u);
                    a01 = wasm_f32x4_add(a01, wasm_f32x4_mul(x0, w));
                    a11 = wasm_f32x4_add(a11, wasm_f32x4_mul(x1, w));
                }
                {
                    const v128_t w = wasm_v128_load(packed + 8u);
                    a02 = wasm_f32x4_add(a02, wasm_f32x4_mul(x0, w));
                    a12 = wasm_f32x4_add(a12, wasm_f32x4_mul(x1, w));
                }
                {
                    const v128_t w = wasm_v128_load(packed + 12u);
                    a03 = wasm_f32x4_add(a03, wasm_f32x4_mul(x0, w));
                    a13 = wasm_f32x4_add(a13, wasm_f32x4_mul(x1, w));
                }
            }
            wasm_pointwise_store_row(a00, a01, a02, a03, epilogue, output,
                                     (size_t)pixel * output_channels + oc, oc, lanes);
            wasm_pointwise_store_row(a10, a11, a12, a13, epilogue, output,
                                     (size_t)(pixel + 1u) * output_channels + oc,
                                     oc, lanes);
        }
    }
    if (pixel < pixels) {
        lw_nhwc_epilogue tail_epilogue;
        const lw_nhwc_epilogue* tail_epilogue_ptr = NULL;
        if (epilogue != NULL) {
            tail_epilogue = *epilogue;
            if (tail_epilogue.residual != NULL)
                tail_epilogue.residual += (size_t)pixel * output_channels;
            tail_epilogue_ptr = &tail_epilogue;
        }
        lw_wasm128_nhwc_pointwise_4x16_f32(
            input + (size_t)pixel * input_channels, packed_weights,
            tail_epilogue_ptr, output + (size_t)pixel * output_channels,
            1u, input_channels, output_channels);
    }
}
