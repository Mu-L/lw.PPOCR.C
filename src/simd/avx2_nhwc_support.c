#include "nhwc_internal.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#define LW_NHWC_SUPPORT_X86 1
#else
#define LW_NHWC_SUPPORT_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LW_NHWC_SUPPORT_TARGET __attribute__((target("avx2,fma")))
#else
#define LW_NHWC_SUPPORT_TARGET
#endif

LW_NHWC_SUPPORT_TARGET
void lw_avx2_nhwc_affine_f32(const float* input, const float* mul, const float* add,
                             float* output, uint32_t pixels, uint32_t channels) {
    uint32_t pixel;
    if (input == NULL || mul == NULL || add == NULL || output == NULL) return;
    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t channel = 0u;
#if LW_NHWC_SUPPORT_X86
        for (; channel + 8u <= channels; channel += 8u) {
            __m256 x = _mm256_loadu_ps(input + (size_t)pixel * channels + channel);
            __m256 scale = _mm256_loadu_ps(mul + channel);
            __m256 bias = _mm256_loadu_ps(add + channel);
            _mm256_storeu_ps(output + (size_t)pixel * channels + channel,
                             _mm256_fmadd_ps(x, scale, bias));
        }
#endif
        for (; channel < channels; ++channel) {
            size_t index = (size_t)pixel * channels + channel;
            output[index] = input[index] * mul[channel] + add[channel];
        }
    }
}

LW_NHWC_SUPPORT_TARGET
void lw_avx2_nhwc_reduce_mean_hw_f32(const float* input, float* output,
                                     uint32_t batch, uint32_t height,
                                     uint32_t width, uint32_t channels) {
    uint32_t n;
    if (input == NULL || output == NULL || batch == 0u || height == 0u ||
        width == 0u || channels == 0u) return;
    for (n = 0u; n < batch; ++n) {
        uint32_t channel = 0u;
#if LW_NHWC_SUPPORT_X86
        for (; channel + 8u <= channels; channel += 8u) {
            __m256 sum = _mm256_setzero_ps();
            for (uint32_t y = 0u; y < height; ++y) {
                for (uint32_t x = 0u; x < width; ++x) {
                    const float* source = input + ((((size_t)n * height + y) * width + x) * channels + channel);
                    sum = _mm256_add_ps(sum, _mm256_loadu_ps(source));
                }
            }
            sum = _mm256_mul_ps(sum, _mm256_set1_ps(1.0f / ((float)height * (float)width)));
            _mm256_storeu_ps(output + (size_t)n * channels + channel, sum);
        }
#endif
        for (; channel < channels; ++channel) {
            float sum = 0.0f;
            for (uint32_t y = 0u; y < height; ++y) {
                for (uint32_t x = 0u; x < width; ++x) {
                    sum += input[(((size_t)n * height + y) * width + x) * channels + channel];
                }
            }
            output[(size_t)n * channels + channel] = sum / ((float)height * (float)width);
        }
    }
}

LW_NHWC_SUPPORT_TARGET
void lw_avx2_nhwc_pool_f32(const float* input, float* output,
                           uint32_t batch, uint32_t input_height,
                           uint32_t input_width, uint32_t output_height,
                           uint32_t output_width, uint32_t channels,
                           uint32_t kernel_h, uint32_t kernel_w,
                           uint32_t stride_h, uint32_t stride_w,
                           uint32_t pad_top, uint32_t pad_left,
                           uint8_t count_include_pad, uint8_t is_max) {
    if (input == NULL || output == NULL || batch == 0u || input_height == 0u ||
        input_width == 0u || output_height == 0u || output_width == 0u ||
        channels == 0u || kernel_h == 0u || kernel_w == 0u ||
        stride_h == 0u || stride_w == 0u) return;
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t oy = 0u; oy < output_height; ++oy) {
            for (uint32_t ox = 0u; ox < output_width; ++ox) {
                int32_t iy0 = (int32_t)((uint64_t)oy * stride_h) - (int32_t)pad_top;
                int32_t ix0 = (int32_t)((uint64_t)ox * stride_w) - (int32_t)pad_left;
                uint32_t count = 0u;
                uint32_t channel = 0u;
#if LW_NHWC_SUPPORT_X86
                for (; channel + 8u <= channels; channel += 8u) {
                    __m256 value = is_max ? _mm256_set1_ps(-FLT_MAX) : _mm256_setzero_ps();
                    for (uint32_t ky = 0u; ky < kernel_h; ++ky) {
                        int32_t iy = iy0 + (int32_t)ky;
                        if (iy < 0 || iy >= (int32_t)input_height) continue;
                        for (uint32_t kx = 0u; kx < kernel_w; ++kx) {
                            int32_t ix = ix0 + (int32_t)kx;
                            if (ix < 0 || ix >= (int32_t)input_width) continue;
                            const float* source = input + ((((size_t)n * input_height + (uint32_t)iy) *
                                input_width + (uint32_t)ix) * channels + channel);
                            __m256 sample = _mm256_loadu_ps(source);
                            value = is_max ? _mm256_max_ps(value, sample) : _mm256_add_ps(value, sample);
                            if (channel == 0u) ++count;
                        }
                    }
                    if (!is_max) {
                        uint32_t denominator = count_include_pad ? kernel_h * kernel_w : count;
                        value = _mm256_mul_ps(value, _mm256_set1_ps(1.0f / (float)denominator));
                    }
                    _mm256_storeu_ps(output + ((((size_t)n * output_height + oy) * output_width + ox) * channels + channel), value);
                }
#endif
                for (; channel < channels; ++channel) {
                    float value = is_max ? -FLT_MAX : 0.0f;
                    for (uint32_t ky = 0u; ky < kernel_h; ++ky) {
                        int32_t iy = iy0 + (int32_t)ky;
                        if (iy < 0 || iy >= (int32_t)input_height) continue;
                        for (uint32_t kx = 0u; kx < kernel_w; ++kx) {
                            int32_t ix = ix0 + (int32_t)kx;
                            float sample;
                            if (ix < 0 || ix >= (int32_t)input_width) continue;
                            sample = input[(((size_t)n * input_height + (uint32_t)iy) *
                                input_width + (uint32_t)ix) * channels + channel];
                            if (is_max) {
                                if (sample > value) value = sample;
                            } else {
                                value += sample;
                            }
                        }
                    }
                    if (!is_max) {
                        uint32_t denominator = count_include_pad ? kernel_h * kernel_w : count;
                        value /= (float)denominator;
                    }
                    output[((((size_t)n * output_height + oy) * output_width + ox) * channels + channel)] = value;
                }
            }
        }
    }
}