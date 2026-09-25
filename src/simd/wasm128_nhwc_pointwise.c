#include "nhwc_internal.h"

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
                float sums[LW_NHWC_OC_BLOCK];
                size_t output_base = (size_t)(pixel + row) * output_channels + oc;
                for (uint32_t group = 0u; group < 4u; ++group) {
                    wasm_v128_store(sums + group * 4u, accumulators[row][group]);
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
