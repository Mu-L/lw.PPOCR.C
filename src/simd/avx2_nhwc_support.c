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

static void lw_nhwc_pool_valid_window(int32_t input_y0, int32_t input_x0,
                                      uint32_t input_height, uint32_t input_width,
                                      uint32_t kernel_h, uint32_t kernel_w,
                                      uint32_t* ky_begin, uint32_t* ky_end,
                                      uint32_t* kx_begin, uint32_t* kx_end) {
    int32_t y_end = (int32_t)kernel_h;
    int32_t x_end = (int32_t)kernel_w;
    int32_t y_begin = input_y0 < 0 ? -input_y0 : 0;
    int32_t x_begin = input_x0 < 0 ? -input_x0 : 0;
    if (input_y0 + y_end > (int32_t)input_height) y_end = (int32_t)input_height - input_y0;
    if (input_x0 + x_end > (int32_t)input_width) x_end = (int32_t)input_width - input_x0;
    if (y_begin < 0) y_begin = 0;
    if (x_begin < 0) x_begin = 0;
    if (y_end < y_begin) y_end = y_begin;
    if (x_end < x_begin) x_end = x_begin;
    if (y_begin > (int32_t)kernel_h) y_begin = (int32_t)kernel_h;
    if (x_begin > (int32_t)kernel_w) x_begin = (int32_t)kernel_w;
    if (y_end > (int32_t)kernel_h) y_end = (int32_t)kernel_h;
    if (x_end > (int32_t)kernel_w) x_end = (int32_t)kernel_w;
    *ky_begin = (uint32_t)y_begin;
    *ky_end = (uint32_t)y_end;
    *kx_begin = (uint32_t)x_begin;
    *kx_end = (uint32_t)x_end;
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
                uint32_t ky_begin;
                uint32_t ky_end;
                uint32_t kx_begin;
                uint32_t kx_end;
                uint32_t valid_count;
                uint32_t denominator;
                uint32_t channel = 0u;
                lw_nhwc_pool_valid_window(iy0, ix0, input_height, input_width,
                                          kernel_h, kernel_w, &ky_begin, &ky_end,
                                          &kx_begin, &kx_end);
                valid_count = (ky_end - ky_begin) * (kx_end - kx_begin);
                denominator = count_include_pad ? kernel_h * kernel_w : valid_count;
                if (denominator == 0u) denominator = 1u;
#if LW_NHWC_SUPPORT_X86
                for (; channel + 8u <= channels; channel += 8u) {
                    __m256 value = is_max ? _mm256_set1_ps(-FLT_MAX) : _mm256_setzero_ps();
                    for (uint32_t ky = ky_begin; ky < ky_end; ++ky) {
                        uint32_t iy = (uint32_t)(iy0 + (int32_t)ky);
                        for (uint32_t kx = kx_begin; kx < kx_end; ++kx) {
                            uint32_t ix = (uint32_t)(ix0 + (int32_t)kx);
                            const float* source = input + ((((size_t)n * input_height + iy) *
                                input_width + ix) * channels + channel);
                            __m256 sample = _mm256_loadu_ps(source);
                            value = is_max ? _mm256_max_ps(value, sample) : _mm256_add_ps(value, sample);
                        }
                    }
                    if (!is_max) {
                        value = _mm256_mul_ps(value, _mm256_set1_ps(1.0f / (float)denominator));
                    }
                    _mm256_storeu_ps(output + ((((size_t)n * output_height + oy) * output_width + ox) *
                        channels + channel), value);
                }
#endif
                for (; channel < channels; ++channel) {
                    float value = is_max ? -FLT_MAX : 0.0f;
                    for (uint32_t ky = ky_begin; ky < ky_end; ++ky) {
                        uint32_t iy = (uint32_t)(iy0 + (int32_t)ky);
                        for (uint32_t kx = kx_begin; kx < kx_end; ++kx) {
                            uint32_t ix = (uint32_t)(ix0 + (int32_t)kx);
                            float sample = input[(((size_t)n * input_height + iy) *
                                input_width + ix) * channels + channel];
                            if (is_max) {
                                if (sample > value) value = sample;
                            } else {
                                value += sample;
                            }
                        }
                    }
                    if (!is_max) value /= (float)denominator;
                    output[((((size_t)n * output_height + oy) * output_width + ox) *
                        channels + channel)] = value;
                }
            }
        }
    }
}