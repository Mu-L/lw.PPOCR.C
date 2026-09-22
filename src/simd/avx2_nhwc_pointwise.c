#include "nhwc_internal.h"

#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LW_NHWC_X86 1
#include <immintrin.h>
#endif

static float lw_nhwc_apply_activation(float value, uint16_t activation,
                                      float alpha, float beta) {
    if (activation == LW_NHWC_ACT_RELU) {
        return value > 0.0f ? value : 0.0f;
    }
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        float gate = value + 3.0f;
        if (gate < 0.0f) {
            gate = 0.0f;
        } else if (gate > 6.0f) {
            gate = 6.0f;
        }
        return value * gate / 6.0f;
    }
    if (activation == LW_NHWC_ACT_GELU) {
        /* The fast-path benchmark does not use GELU; keep a finite fallback. */
        float scaled = alpha == 0.0f ? value : alpha * value;
        return 0.5f * scaled * (1.0f + beta);
    }
    return value;
}

static void lw_nhwc_pointwise_scalar(const float* input,
                                     const float* packed_weights,
                                     const lw_nhwc_epilogue* epilogue,
                                     float* output,
                                     uint32_t pixels,
                                     uint32_t input_channels,
                                     uint32_t output_channels) {
    const float* bias = epilogue == NULL ? NULL : epilogue->bias;
    const float* residual = epilogue == NULL ? NULL : epilogue->residual;
    uint16_t activation = epilogue == NULL ? LW_NHWC_ACT_NONE : epilogue->activation;
    float alpha = epilogue == NULL ? 0.0f : epilogue->alpha;
    float beta = epilogue == NULL ? 0.0f : epilogue->beta;
    uint32_t pixel;

    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels;
             ++output_channel) {
            uint32_t block = output_channel / LW_NHWC_OC_BLOCK;
            uint32_t lane = output_channel % LW_NHWC_OC_BLOCK;
            const float* packed = packed_weights +
                                  (size_t)block * input_channels * LW_NHWC_OC_BLOCK;
            float sum = bias == NULL ? 0.0f : bias[output_channel];
            uint32_t input_channel;
            for (input_channel = 0u; input_channel < input_channels;
                 ++input_channel) {
                sum += input[(size_t)pixel * input_channels + input_channel] *
                       packed[(size_t)input_channel * LW_NHWC_OC_BLOCK + lane];
            }
            if (residual != NULL) {
                sum += residual[(size_t)pixel * output_channels + output_channel];
            }
            output[(size_t)pixel * output_channels + output_channel] =
                lw_nhwc_apply_activation(sum, activation, alpha, beta);
        }
    }
}

#if defined(LW_NHWC_X86)
#if defined(__GNUC__) || defined(__clang__)
#define LW_NHWC_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define LW_NHWC_AVX2_FMA
#endif

LW_NHWC_AVX2_FMA
static void lw_nhwc_store_row16(__m256 lo, __m256 hi,
                                float* output,
                                const float* residual,
                                uint16_t activation,
                                uint32_t channels) {
    if (channels < 16u) {
        float values[16];
        _mm256_storeu_ps(values, lo);
        _mm256_storeu_ps(values + 8u, hi);
        for (uint32_t i = 0u; i < channels; ++i) {
            float value = values[i] + (residual == NULL ? 0.0f : residual[i]);
            if (activation == LW_NHWC_ACT_RELU) {
                value = value < 0.0f ? 0.0f : value;
            } else if (activation == LW_NHWC_ACT_HARDSWISH) {
                float gate = value + 3.0f;
                if (gate < 0.0f) gate = 0.0f;
                if (gate > 6.0f) gate = 6.0f;
                value = value * gate * (1.0f / 6.0f);
            }
            output[i] = value;
        }
        return;
    }
    if (residual != NULL) {
        lo = _mm256_add_ps(lo, _mm256_loadu_ps(residual));
        hi = _mm256_add_ps(hi, _mm256_loadu_ps(residual + 8u));
    }
    if (activation == LW_NHWC_ACT_RELU) {
        __m256 zero = _mm256_setzero_ps();
        lo = _mm256_max_ps(lo, zero);
        hi = _mm256_max_ps(hi, zero);
    } else if (activation == LW_NHWC_ACT_HARDSWISH) {
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 six = _mm256_set1_ps(6.0f);
        __m256 inv_six = _mm256_set1_ps(1.0f / 6.0f);
        __m256 zero = _mm256_setzero_ps();
        __m256 gate_lo = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(lo, three), zero), six);
        __m256 gate_hi = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(hi, three), zero), six);
        lo = _mm256_mul_ps(_mm256_mul_ps(lo, gate_lo), inv_six);
        hi = _mm256_mul_ps(_mm256_mul_ps(hi, gate_hi), inv_six);
    }
    _mm256_storeu_ps(output, lo);
    _mm256_storeu_ps(output + 8u, hi);
}

LW_NHWC_AVX2_FMA
static void lw_nhwc_tile_rows16(const float* input,
                                const float* packed_weights,
                                const float* bias,
                                const float* residual,
                                float* output,
                                uint32_t rows,
                                uint32_t input_channels,
                                uint32_t output_stride,
                                uint16_t activation,
                                uint32_t output_channels) {
    __m256 lo[6];
    __m256 hi[6];
    uint32_t row;
    uint32_t input_channel;
    __m256 zero = _mm256_setzero_ps();

    for (row = 0u; row < 6u; ++row) {
        if (bias != NULL && output_channels < LW_NHWC_OC_BLOCK) {
            float bias_values[LW_NHWC_OC_BLOCK] = {0.0f};
            for (uint32_t lane = 0u; lane < output_channels; ++lane) {
                bias_values[lane] = bias[lane];
            }
            lo[row] = _mm256_loadu_ps(bias_values);
            hi[row] = _mm256_loadu_ps(bias_values + 8u);
        } else if (bias != NULL) {
            lo[row] = _mm256_loadu_ps(bias);
            hi[row] = _mm256_loadu_ps(bias + 8u);
        } else {
            lo[row] = zero;
            hi[row] = zero;
        }
    }
    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights = packed_weights + (size_t)input_channel * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        for (row = 0u; row < 6u; ++row) {
            const float* row_input = input + (size_t)(row < rows ? row : 0u) * input_channels;
            __m256 value = _mm256_set1_ps(row_input[input_channel]);
            lo[row] = _mm256_fmadd_ps(value, w0, lo[row]);
            hi[row] = _mm256_fmadd_ps(value, w1, hi[row]);
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL :
            residual + (size_t)row * output_stride;
        lw_nhwc_store_row16(lo[row], hi[row], row_output, row_residual,
                            activation, output_channels);
    }
}
#endif

void lw_avx2_fma_nhwc_pointwise_grouped_f32(const float* input,
                                             const float* packed_weights,
                                             const lw_nhwc_epilogue* epilogue,
                                             float* output,
                                             uint32_t pixels,
                                             uint32_t input_channels,
                                             uint32_t output_channels,
                                             uint32_t group_tiles) {
    const float* bias = epilogue == NULL ? NULL : epilogue->bias;
    const float* residual = epilogue == NULL ? NULL : epilogue->residual;
    uint16_t activation = epilogue == NULL ? LW_NHWC_ACT_NONE : epilogue->activation;

    if (input == NULL || packed_weights == NULL || output == NULL ||
        input_channels == 0u || output_channels == 0u || pixels == 0u) {
        return;
    }
    if (activation != LW_NHWC_ACT_NONE && activation != LW_NHWC_ACT_RELU &&
        activation != LW_NHWC_ACT_HARDSWISH) {
        lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                                 input_channels, output_channels);
        return;
    }

#if defined(LW_NHWC_X86)
    {
        uint32_t tile_count = (pixels + LW_NHWC_PIXEL_TILE - 1u) /
                              LW_NHWC_PIXEL_TILE;
        uint32_t tile_group = group_tiles == 0u ? tile_count : group_tiles;
        uint32_t group_begin;
        if (tile_group == 0u) {
            tile_group = 1u;
        }
        for (group_begin = 0u; group_begin < tile_count;
             group_begin += tile_group) {
            uint32_t group_end = group_begin + tile_group;
            uint32_t output_channel;
            if (group_end > tile_count) {
                group_end = tile_count;
            }
            for (output_channel = 0u; output_channel < output_channels;
                 output_channel += LW_NHWC_OC_BLOCK) {
                const float* packed = packed_weights +
                    (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels *
                    LW_NHWC_OC_BLOCK;
                for (uint32_t tile = group_begin; tile < group_end; ++tile) {
                    uint32_t pixel = tile * LW_NHWC_PIXEL_TILE;
                    uint32_t rows = pixels - pixel;
                    const float* tile_input = input + (size_t)pixel * input_channels;
                    float* tile_output = output + (size_t)pixel * output_channels +
                                         output_channel;
                    const float* tile_residual = residual == NULL ? NULL :
                        residual + (size_t)pixel * output_channels + output_channel;
                    if (rows > LW_NHWC_PIXEL_TILE) {
                        rows = LW_NHWC_PIXEL_TILE;
                    }
                    uint32_t block_channels = output_channels - output_channel;
                    if (block_channels > LW_NHWC_OC_BLOCK) {
                        block_channels = LW_NHWC_OC_BLOCK;
                    }
                    lw_nhwc_tile_rows16(tile_input, packed,
                                        bias == NULL ? NULL : bias + output_channel,
                                        tile_residual, tile_output, rows,
                                        input_channels, output_channels, activation,
                                        block_channels);
                }
            }
        }
    }
#else
    (void)group_tiles;
    lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                             input_channels, output_channels);
#endif
}


#if defined(LW_NHWC_X86)
LW_NHWC_AVX2_FMA
static void lw_nhwc_tile_rows16_4(const float* input, const float* packed_weights,
                                   const float* bias, const float* residual, float* output,
                                   uint32_t rows, uint32_t input_channels,
                                   uint32_t output_stride, uint16_t activation) {
    __m256 lo[4], hi[4];
    __m256 zero = _mm256_setzero_ps();
    uint32_t row;
    for (row = 0u; row < 4u; ++row) {
        if (bias != NULL) {
            lo[row] = _mm256_loadu_ps(bias);
            hi[row] = _mm256_loadu_ps(bias + 8u);
        } else {
            lo[row] = zero;
            hi[row] = zero;
        }
    }
    for (uint32_t input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights = packed_weights + (size_t)input_channel * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        for (row = 0u; row < rows; ++row) {
            __m256 value = _mm256_set1_ps(input[(size_t)row * input_channels + input_channel]);
            lo[row] = _mm256_fmadd_ps(value, w0, lo[row]);
            hi[row] = _mm256_fmadd_ps(value, w1, hi[row]);
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL : residual + (size_t)row * output_stride;
        lw_nhwc_store_row16(lo[row], hi[row], row_output, row_residual,
                            activation, LW_NHWC_OC_BLOCK);
    }
}

static __m256 lw_nhwc_apply_vec(__m256 value, uint16_t activation) {
    if (activation == LW_NHWC_ACT_RELU) {
        return _mm256_max_ps(value, _mm256_setzero_ps());
    }
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        __m256 gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(value, _mm256_set1_ps(3.0f)),
                                                  _mm256_setzero_ps()),
                                    _mm256_set1_ps(6.0f));
        return _mm256_mul_ps(_mm256_mul_ps(value, gate), _mm256_set1_ps(1.0f / 6.0f));
    }
    return value;
}

static void lw_nhwc_store_row32(__m256* accumulators, float* output,
                                const float* residual, uint16_t activation) {
    for (uint32_t vector = 0u; vector < 4u; ++vector) {
        __m256 value = accumulators[vector];
        if (residual != NULL) value = _mm256_add_ps(value, _mm256_loadu_ps(residual + vector * 8u));
        value = lw_nhwc_apply_vec(value, activation);
        _mm256_storeu_ps(output + vector * 8u, value);
    }
}

static void lw_nhwc_tile_rows32(const float* input, const float* packed_weights,
                                const float* bias, const float* residual, float* output,
                                uint32_t rows, uint32_t input_channels,
                                uint32_t output_stride, uint16_t activation,
                                uint32_t row_tile) {
    __m256 accumulators[3][4];
    __m256 zero = _mm256_setzero_ps();
    uint32_t row;
    for (row = 0u; row < row_tile; ++row) {
        for (uint32_t vector = 0u; vector < 4u; ++vector) {
            uint32_t lane_offset = vector * 8u;
            accumulators[row][vector] = bias == NULL ? zero : _mm256_loadu_ps(bias + lane_offset);
        }
    }
    for (uint32_t input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights0 = packed_weights + (size_t)input_channel * 16u;
        const float* weights1 = weights0 + (size_t)input_channels * 16u;
        __m256 weights[4] = {
            _mm256_loadu_ps(weights0), _mm256_loadu_ps(weights0 + 8u),
            _mm256_loadu_ps(weights1), _mm256_loadu_ps(weights1 + 8u)
        };
        for (row = 0u; row < rows; ++row) {
            __m256 value = _mm256_set1_ps(input[(size_t)row * input_channels + input_channel]);
            for (uint32_t vector = 0u; vector < 4u; ++vector) {
                accumulators[row][vector] =
                    _mm256_fmadd_ps(value, weights[vector], accumulators[row][vector]);
            }
        }
    }
    for (row = 0u; row < rows; ++row) {
        float* row_output = output + (size_t)row * output_stride;
        const float* row_residual = residual == NULL ? NULL : residual + (size_t)row * output_stride;
        lw_nhwc_store_row32(accumulators[row], row_output, row_residual, activation);
    }
}
#endif

void lw_avx2_fma_nhwc_pointwise_4x16_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    if (output_channels % LW_NHWC_OC_BLOCK != 0u ||
        epilogue == NULL ||
        (epilogue->activation != LW_NHWC_ACT_NONE &&
         epilogue->activation != LW_NHWC_ACT_RELU &&
         epilogue->activation != LW_NHWC_ACT_HARDSWISH)) {
        lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels);
        return;
    }
#if defined(LW_NHWC_X86)
    for (uint32_t tile = 0u; tile < pixels; tile += 4u) {
        uint32_t rows = pixels - tile;
        if (rows > 4u) rows = 4u;
        for (uint32_t output_channel = 0u; output_channel < output_channels;
             output_channel += LW_NHWC_OC_BLOCK) {
            const float* packed = packed_weights +
                (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            lw_nhwc_tile_rows16_4(input + (size_t)tile * input_channels, packed,
                                   epilogue->bias == NULL ? NULL : epilogue->bias + output_channel,
                                   epilogue->residual == NULL ? NULL :
                                       epilogue->residual + (size_t)tile * output_channels + output_channel,
                                   output + (size_t)tile * output_channels + output_channel,
                                   rows, input_channels, output_channels, epilogue->activation);
        }
    }
#else
    lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                   pixels, input_channels, output_channels);
#endif
}

static void lw_avx2_fma_nhwc_pointwise_32x_f32(const float* input,
                                                const float* packed_weights,
                                                const lw_nhwc_epilogue* epilogue,
                                                float* output, uint32_t pixels,
                                                uint32_t input_channels,
                                                uint32_t output_channels,
                                                uint32_t row_tile) {
    if (output_channels % 32u != 0u ||
        epilogue == NULL ||
        (epilogue->activation != LW_NHWC_ACT_NONE &&
         epilogue->activation != LW_NHWC_ACT_RELU &&
         epilogue->activation != LW_NHWC_ACT_HARDSWISH)) {
        lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels);
        return;
    }
#if defined(LW_NHWC_X86)
    for (uint32_t tile = 0u; tile < pixels; tile += row_tile) {
        uint32_t rows = pixels - tile;
        if (rows > row_tile) rows = row_tile;
        for (uint32_t output_channel = 0u; output_channel < output_channels;
             output_channel += 32u) {
            const float* packed = packed_weights +
                (size_t)(output_channel / LW_NHWC_OC_BLOCK) * input_channels * LW_NHWC_OC_BLOCK;
            lw_nhwc_tile_rows32(input + (size_t)tile * input_channels, packed,
                                epilogue->bias == NULL ? NULL : epilogue->bias + output_channel,
                                epilogue->residual == NULL ? NULL :
                                    epilogue->residual + (size_t)tile * output_channels + output_channel,
                                output + (size_t)tile * output_channels + output_channel,
                                rows, input_channels, output_channels, epilogue->activation, row_tile);
        }
    }
#else
    lw_avx2_fma_nhwc_pointwise_f32(input, packed_weights, epilogue, output,
                                   pixels, input_channels, output_channels);
#endif
}

void lw_avx2_fma_nhwc_pointwise_3x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    lw_avx2_fma_nhwc_pointwise_32x_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels, 3u);
}

void lw_avx2_fma_nhwc_pointwise_2x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels) {
    lw_avx2_fma_nhwc_pointwise_32x_f32(input, packed_weights, epilogue, output,
                                       pixels, input_channels, output_channels, 2u);
}

void lw_avx2_fma_nhwc_pointwise_f32(const float* input,
                                    const float* packed_weights,
                                    const lw_nhwc_epilogue* epilogue,
                                    float* output,
                                    uint32_t pixels,
                                    uint32_t input_channels,
                                    uint32_t output_channels) {
    lw_avx2_fma_nhwc_pointwise_grouped_f32(
        input, packed_weights, epilogue, output, pixels, input_channels,
        output_channels, LW_NHWC_POINTWISE_GROUP_TILES);
}
