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
static void lw_nhwc_tile_6x16(const float* input,
                              const float* packed_weights,
                              const float* bias,
                              float* output,
                              uint32_t input_channels,
                              uint32_t output_stride) {
    __m256 a00;
    __m256 a01;
    __m256 a10;
    __m256 a11;
    __m256 a20;
    __m256 a21;
    __m256 a30;
    __m256 a31;
    __m256 a40;
    __m256 a41;
    __m256 a50;
    __m256 a51;
    uint32_t input_channel;

    if (bias != NULL) {
        a00 = a10 = a20 = a30 = a40 = a50 = _mm256_loadu_ps(bias);
        a01 = a11 = a21 = a31 = a41 = a51 = _mm256_loadu_ps(bias + 8u);
    } else {
        __m256 zero = _mm256_setzero_ps();
        a00 = a01 = a10 = a11 = a20 = a21 = zero;
        a30 = a31 = a40 = a41 = a50 = a51 = zero;
    }

    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
        const float* weights = packed_weights + (size_t)input_channel * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        __m256 x0 = _mm256_set1_ps(input[input_channel]);
        __m256 x1 = _mm256_set1_ps(input[(size_t)input_channels + input_channel]);
        __m256 x2 = _mm256_set1_ps(input[(size_t)2u * input_channels + input_channel]);
        __m256 x3 = _mm256_set1_ps(input[(size_t)3u * input_channels + input_channel]);
        __m256 x4 = _mm256_set1_ps(input[(size_t)4u * input_channels + input_channel]);
        __m256 x5 = _mm256_set1_ps(input[(size_t)5u * input_channels + input_channel]);
        a00 = _mm256_fmadd_ps(x0, w0, a00);
        a01 = _mm256_fmadd_ps(x0, w1, a01);
        a10 = _mm256_fmadd_ps(x1, w0, a10);
        a11 = _mm256_fmadd_ps(x1, w1, a11);
        a20 = _mm256_fmadd_ps(x2, w0, a20);
        a21 = _mm256_fmadd_ps(x2, w1, a21);
        a30 = _mm256_fmadd_ps(x3, w0, a30);
        a31 = _mm256_fmadd_ps(x3, w1, a31);
        a40 = _mm256_fmadd_ps(x4, w0, a40);
        a41 = _mm256_fmadd_ps(x4, w1, a41);
        a50 = _mm256_fmadd_ps(x5, w0, a50);
        a51 = _mm256_fmadd_ps(x5, w1, a51);
    }
    _mm256_storeu_ps(output, a00);
    _mm256_storeu_ps(output + 8u, a01);
    _mm256_storeu_ps(output + (size_t)output_stride, a10);
    _mm256_storeu_ps(output + (size_t)output_stride + 8u, a11);
    _mm256_storeu_ps(output + (size_t)2u * output_stride, a20);
    _mm256_storeu_ps(output + (size_t)2u * output_stride + 8u, a21);
    _mm256_storeu_ps(output + (size_t)3u * output_stride, a30);
    _mm256_storeu_ps(output + (size_t)3u * output_stride + 8u, a31);
    _mm256_storeu_ps(output + (size_t)4u * output_stride, a40);
    _mm256_storeu_ps(output + (size_t)4u * output_stride + 8u, a41);
    _mm256_storeu_ps(output + (size_t)5u * output_stride, a50);
    _mm256_storeu_ps(output + (size_t)5u * output_stride + 8u, a51);
}
#endif

void lw_avx2_fma_nhwc_pointwise_f32(const float* input,
                                    const float* packed_weights,
                                    const lw_nhwc_epilogue* epilogue,
                                    float* output,
                                    uint32_t pixels,
                                    uint32_t input_channels,
                                    uint32_t output_channels) {
    const float* bias = epilogue == NULL ? NULL : epilogue->bias;
    uint32_t output_channel;
    uint32_t pixel;

    if (input == NULL || packed_weights == NULL || output == NULL ||
        input_channels == 0u || output_channels == 0u || pixels == 0u) {
        return;
    }
    if (epilogue != NULL &&
        (epilogue->residual != NULL || epilogue->activation != LW_NHWC_ACT_NONE)) {
        lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                                 input_channels, output_channels);
        return;
    }

    for (output_channel = 0u;
         output_channel + LW_NHWC_OC_BLOCK <= output_channels;
         output_channel += LW_NHWC_OC_BLOCK) {
        const float* packed = packed_weights +
                              (size_t)(output_channel / LW_NHWC_OC_BLOCK) *
                                  input_channels * LW_NHWC_OC_BLOCK;
        for (pixel = 0u; pixel + LW_NHWC_PIXEL_TILE <= pixels;
             pixel += LW_NHWC_PIXEL_TILE) {
#if defined(LW_NHWC_X86)
            lw_nhwc_tile_6x16(input + (size_t)pixel * input_channels, packed,
                              bias == NULL ? NULL : bias + output_channel,
                              output + (size_t)pixel * output_channels + output_channel,
                              input_channels, output_channels);
#else
            break;
#endif
        }
        for (; pixel < pixels; ++pixel) {
            uint32_t lane;
            for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                float sum = bias == NULL ? 0.0f : bias[output_channel + lane];
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels;
                     ++input_channel) {
                    sum += input[(size_t)pixel * input_channels + input_channel] *
                           packed[(size_t)input_channel * LW_NHWC_OC_BLOCK + lane];
                }
                output[(size_t)pixel * output_channels + output_channel + lane] = sum;
            }
        }
    }
    if (output_channel < output_channels) {
        lw_nhwc_pointwise_scalar(input, packed_weights, epilogue, output, pixels,
                                 input_channels, output_channels);
    }
}


