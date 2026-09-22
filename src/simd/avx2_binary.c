#include "simd_kernels.h"

#include <stddef.h>
#include <math.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_BINARY 1
#else
#  define LW_COMPILES_AVX2_BINARY 0
#endif

#if LW_COMPILES_AVX2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_binary_contiguous_f32(
    lw_scalar_binary_op operation,
    const float* left,
    const float* right,
    float* output,
    uint64_t element_count) {
    uint64_t index = 0u;
#if LW_COMPILES_AVX2_BINARY
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_add_ps(_mm256_loadu_ps(left + (size_t)index),
                                          _mm256_loadu_ps(right + (size_t)index));
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_mul_ps(_mm256_loadu_ps(left + (size_t)index),
                                          _mm256_loadu_ps(right + (size_t)index));
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_DIV) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_div_ps(_mm256_loadu_ps(left + (size_t)index),
                                          _mm256_loadu_ps(right + (size_t)index));
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    }
#endif
    lw_scalar_binary_contiguous_f32(operation, left + (size_t)index, right + (size_t)index,
                                    output + (size_t)index, element_count - index);
}

#if LW_COMPILES_AVX2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_binary_right_scalar_f32(
    lw_scalar_binary_op operation,
    const float* left,
    float right,
    float* output,
    uint64_t element_count) {
    uint64_t index = 0u;
#if LW_COMPILES_AVX2_BINARY
    __m256 right_values = _mm256_set1_ps(right);
    if (operation == LW_SCALAR_BINARY_ADD) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_add_ps(_mm256_loadu_ps(left + (size_t)index), right_values);
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_MUL) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_mul_ps(_mm256_loadu_ps(left + (size_t)index), right_values);
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    } else if (operation == LW_SCALAR_BINARY_DIV) {
        for (; index + 8u <= element_count; index += 8u) {
            __m256 result = _mm256_div_ps(_mm256_loadu_ps(left + (size_t)index), right_values);
            _mm256_storeu_ps(output + (size_t)index, result);
        }
    }
#endif
    lw_scalar_binary_right_scalar_f32(operation, left + (size_t)index, right,
                                      output + (size_t)index, element_count - index);
}

#if LW_COMPILES_AVX2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_relu_contiguous_f32(const float* input, float* output, uint64_t element_count) {
    uint64_t index = 0u;
#if LW_COMPILES_AVX2_BINARY
    __m256 zero = _mm256_setzero_ps();
    for (; index + 8u <= element_count; index += 8u) {
        __m256 value = _mm256_max_ps(_mm256_loadu_ps(input + (size_t)index), zero);
        _mm256_storeu_ps(output + (size_t)index, value);
    }
#endif
    for (; index < element_count; ++index) {
        float value = input[(size_t)index];
        output[(size_t)index] = value > 0.0f ? value : 0.0f;
    }
}

#if LW_COMPILES_AVX2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_binary_channel_f32(lw_scalar_binary_op operation, const float* full,
                                const float* channel, float* output, uint64_t pixels,
                                uint32_t channels, int broadcast_is_left) {
    uint64_t pixel;
    if (full == NULL || channel == NULL || output == NULL || channels == 0u) return;
    for (pixel = 0u; pixel < pixels; ++pixel) {
        const float* source = full + (size_t)pixel * channels;
        float* destination = output + (size_t)pixel * channels;
        uint32_t c = 0u;
#if LW_COMPILES_AVX2_BINARY
        for (; c + 8u <= channels; c += 8u) {
            __m256 value = _mm256_loadu_ps(source + c);
            __m256 scale = _mm256_loadu_ps(channel + c);
            __m256 result;
            if (operation == LW_SCALAR_BINARY_ADD) {
                result = _mm256_add_ps(broadcast_is_left ? scale : value,
                                       broadcast_is_left ? value : scale);
            } else if (operation == LW_SCALAR_BINARY_SUB) {
                result = _mm256_sub_ps(broadcast_is_left ? scale : value,
                                       broadcast_is_left ? value : scale);
            } else if (operation == LW_SCALAR_BINARY_MUL) {
                result = _mm256_mul_ps(broadcast_is_left ? scale : value,
                                       broadcast_is_left ? value : scale);
            } else if (operation == LW_SCALAR_BINARY_DIV) {
                result = _mm256_div_ps(broadcast_is_left ? scale : value,
                                       broadcast_is_left ? value : scale);
            } else {
                break;
            }
            _mm256_storeu_ps(destination + c, result);
        }
#endif
        for (; c < channels; ++c) {
            float left = broadcast_is_left ? channel[c] : source[c];
            float right = broadcast_is_left ? source[c] : channel[c];
            if (operation == LW_SCALAR_BINARY_ADD) destination[c] = left + right;
            else if (operation == LW_SCALAR_BINARY_SUB) destination[c] = left - right;
            else if (operation == LW_SCALAR_BINARY_MUL) destination[c] = left * right;
            else if (operation == LW_SCALAR_BINARY_DIV) destination[c] = left / right;
            else destination[c] = powf(left, right);
        }
    }
}


void lw_avx2_binary_channel_nchw_f32(lw_scalar_binary_op operation, const float* full,
                                     const float* channel, float* output, uint64_t spatial,
                                     uint32_t channels, int broadcast_is_left) {
    if (full == NULL || channel == NULL || output == NULL || channels == 0u) return;
    for (uint32_t c = 0u; c < channels; ++c) {
        const float* source = full + (size_t)c * spatial;
        float* destination = output + (size_t)c * spatial;
        uint64_t i = 0u;
#if LW_COMPILES_AVX2_BINARY
        const __m256 scalar = _mm256_set1_ps(channel[c]);
        for (; i + 8u <= spatial; i += 8u) {
            const __m256 value = _mm256_loadu_ps(source + i);
            __m256 result;
            if (operation == LW_SCALAR_BINARY_ADD) {
                result = _mm256_add_ps(broadcast_is_left ? scalar : value,
                                       broadcast_is_left ? value : scalar);
            } else if (operation == LW_SCALAR_BINARY_SUB) {
                result = _mm256_sub_ps(broadcast_is_left ? scalar : value,
                                       broadcast_is_left ? value : scalar);
            } else if (operation == LW_SCALAR_BINARY_MUL) {
                result = _mm256_mul_ps(broadcast_is_left ? scalar : value,
                                       broadcast_is_left ? value : scalar);
            } else if (operation == LW_SCALAR_BINARY_DIV) {
                result = _mm256_div_ps(broadcast_is_left ? scalar : value,
                                       broadcast_is_left ? value : scalar);
            } else {
                break;
            }
            _mm256_storeu_ps(destination + i, result);
        }
#endif
        for (; i < spatial; ++i) {
            const float left = broadcast_is_left ? channel[c] : source[i];
            const float right = broadcast_is_left ? source[i] : channel[c];
            if (operation == LW_SCALAR_BINARY_ADD) destination[i] = left + right;
            else if (operation == LW_SCALAR_BINARY_SUB) destination[i] = left - right;
            else if (operation == LW_SCALAR_BINARY_MUL) destination[i] = left * right;
            else if (operation == LW_SCALAR_BINARY_DIV) destination[i] = left / right;
            else destination[i] = powf(left, right);
        }
    }
}

#if LW_COMPILES_AVX2_BINARY && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_hard_sigmoid_contiguous_f32(const float* input, float* output,
                                         uint64_t element_count, float alpha, float beta) {
    uint64_t index = 0u;
#if LW_COMPILES_AVX2_BINARY
    const __m256 alpha_v = _mm256_set1_ps(alpha);
    const __m256 beta_v = _mm256_set1_ps(beta);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; index + 8u <= element_count; index += 8u) {
        __m256 value = _mm256_fmadd_ps(_mm256_loadu_ps(input + (size_t)index), alpha_v, beta_v);
        value = _mm256_max_ps(_mm256_min_ps(value, one), zero);
        _mm256_storeu_ps(output + (size_t)index, value);
    }
#endif
    for (; index < element_count; ++index) {
        float value = input[(size_t)index] * alpha + beta;
        if (value < 0.0f) value = 0.0f;
        else if (value > 1.0f) value = 1.0f;
        output[(size_t)index] = value;
    }
}
