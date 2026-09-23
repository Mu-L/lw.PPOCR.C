#include "x64_rec_fast_internal.h"

#include <stddef.h>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#define LW_CONVERT_X86 1
#else
#define LW_CONVERT_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LW_CONVERT_TARGET __attribute__((target("avx2,fma")))
#else
#define LW_CONVERT_TARGET
#endif

#if LW_CONVERT_X86
/* Transpose an 8x8 block of floats: v0..v7 hold 8-pixel rows of 8 channels
 * each; t0..t7 receive 8-channel rows of 8 pixels each. */
LW_CONVERT_TARGET
static void transpose8x8_ps(const __m256* v, __m256* t) {
    /* Classic AVX2 8x8 float transpose: unpack adjacent row pairs, shuffle
     * quads, then permute 128-bit lanes. */
    __m256 a[8];
    __m256 b[8];
    a[0] = _mm256_unpacklo_ps(v[0], v[1]);
    a[1] = _mm256_unpackhi_ps(v[0], v[1]);
    a[2] = _mm256_unpacklo_ps(v[2], v[3]);
    a[3] = _mm256_unpackhi_ps(v[2], v[3]);
    a[4] = _mm256_unpacklo_ps(v[4], v[5]);
    a[5] = _mm256_unpackhi_ps(v[4], v[5]);
    a[6] = _mm256_unpacklo_ps(v[6], v[7]);
    a[7] = _mm256_unpackhi_ps(v[6], v[7]);
    b[0] = _mm256_shuffle_ps(a[0], a[2], 0x44);
    b[1] = _mm256_shuffle_ps(a[0], a[2], 0xEE);
    b[2] = _mm256_shuffle_ps(a[1], a[3], 0x44);
    b[3] = _mm256_shuffle_ps(a[1], a[3], 0xEE);
    b[4] = _mm256_shuffle_ps(a[4], a[6], 0x44);
    b[5] = _mm256_shuffle_ps(a[4], a[6], 0xEE);
    b[6] = _mm256_shuffle_ps(a[5], a[7], 0x44);
    b[7] = _mm256_shuffle_ps(a[5], a[7], 0xEE);
    t[0] = _mm256_permute2f128_ps(b[0], b[4], 0x20);
    t[1] = _mm256_permute2f128_ps(b[1], b[5], 0x20);
    t[2] = _mm256_permute2f128_ps(b[2], b[6], 0x20);
    t[3] = _mm256_permute2f128_ps(b[3], b[7], 0x20);
    t[4] = _mm256_permute2f128_ps(b[0], b[4], 0x31);
    t[5] = _mm256_permute2f128_ps(b[1], b[5], 0x31);
    t[6] = _mm256_permute2f128_ps(b[2], b[6], 0x31);
    t[7] = _mm256_permute2f128_ps(b[3], b[7], 0x31);
}
#endif

LW_CONVERT_TARGET
void lw_x64_fast_nchw_to_nhwc(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width) {
#if LW_CONVERT_X86
    /* Vectorized 8x8 path for 8-multiple channels; the scalar fallback
     * covers the channel/width tails and odd channel counts. */
    if (channels % 8u == 0u) {
        uint32_t n;
        for (n = 0u; n < batch; ++n) {
            uint32_t y;
            for (y = 0u; y < height; ++y) {
                uint32_t x = 0u;
                while (x + 8u <= width) {
                    uint32_t c = 0u;
                    while (c + 8u <= channels) {
                        __m256 v[8];
                        __m256 t[8];
                        uint32_t k;
                        for (k = 0u; k < 8u; ++k) {
                            v[k] = _mm256_loadu_ps(
                                source + ((size_t)n * channels + c + k) * height * width +
                                (size_t)y * width + x);
                        }
                        transpose8x8_ps(v, t);
                        for (k = 0u; k < 8u; ++k) {
                            _mm256_storeu_ps(
                                destination + (((size_t)n * height + y) * width + x + k) *
                                    channels + c, t[k]);
                        }
                        c += 8u;
                    }
                    while (c < channels) {
                        uint32_t k;
                        for (k = 0u; k < 8u; ++k) {
                            destination[(((size_t)n * height + y) * width + x + k) *
                                            channels + c] =
                                source[((size_t)n * channels + c) * height * width +
                                       (size_t)y * width + x + k];
                        }
                        ++c;
                    }
                    x += 8u;
                }
                while (x < width) {
                    uint32_t c;
                    for (c = 0u; c < channels; ++c) {
                        destination[(((size_t)n * height + y) * width + x) * channels + c] =
                            source[((size_t)n * channels + c) * height * width +
                                   (size_t)y * width + x];
                    }
                    ++x;
                }
            }
        }
        return;
    }
#endif
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                float* dst = destination + (((size_t)n * height + y) * width + x) * channels;
                for (uint32_t c = 0u; c < channels; ++c) {
                    dst[c] = source[(((size_t)n * channels + c) * height + y) * width + x];
                }
            }
        }
    }
}

LW_CONVERT_TARGET
void lw_x64_fast_nhwc_to_nchw(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width) {
#if LW_CONVERT_X86
    if (channels % 8u == 0u) {
        uint32_t n;
        for (n = 0u; n < batch; ++n) {
            uint32_t y;
            for (y = 0u; y < height; ++y) {
                uint32_t x = 0u;
                while (x + 8u <= width) {
                    uint32_t c = 0u;
                    while (c + 8u <= channels) {
                        __m256 v[8];
                        __m256 t[8];
                        uint32_t k;
                        for (k = 0u; k < 8u; ++k) {
                            v[k] = _mm256_loadu_ps(
                                source + (((size_t)n * height + y) * width + x + k) *
                                    channels + c);
                        }
                        transpose8x8_ps(v, t);
                        for (k = 0u; k < 8u; ++k) {
                            _mm256_storeu_ps(
                                destination + ((size_t)n * channels + c + k) * height * width +
                                    (size_t)y * width + x, t[k]);
                        }
                        c += 8u;
                    }
                    while (c < channels) {
                        uint32_t k;
                        for (k = 0u; k < 8u; ++k) {
                            destination[((size_t)n * channels + c) * height * width +
                                        (size_t)y * width + x + k] =
                                source[(((size_t)n * height + y) * width + x + k) *
                                           channels + c];
                        }
                        ++c;
                    }
                    x += 8u;
                }
                while (x < width) {
                    uint32_t c;
                    for (c = 0u; c < channels; ++c) {
                        destination[((size_t)n * channels + c) * height * width +
                                     (size_t)y * width + x] =
                            source[(((size_t)n * height + y) * width + x) * channels + c];
                    }
                    ++x;
                }
            }
        }
        return;
    }
#endif
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t c = 0u; c < channels; ++c) {
            float* dst = destination + ((size_t)n * channels + c) * height * width;
            for (uint32_t y = 0u; y < height; ++y) {
                for (uint32_t x = 0u; x < width; ++x) {
                    dst[(size_t)y * width + x] =
                        source[(((size_t)n * height + y) * width + x) * channels + c];
                }
            }
        }
    }
}
