#include "simd_kernels.h"

/* AVX2 bitmap helpers for the DET postprocess: thresholding the probability
 * map and computing the 4-neighborhood "interior" bitmap that replaces the
 * per-pixel boundary probe during connected-component extraction. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_BITMAP 1
#else
#  define LW_COMPILES_AVX2_BITMAP 0
#endif

#if LW_COMPILES_AVX2_BITMAP && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
int lw_avx2_threshold_bitmap_f32(const float* prediction, uint8_t* bitmap,
                                 uint64_t pixel_count, float threshold) {
#if LW_COMPILES_AVX2_BITMAP
    /* One 64-bit word of output per 8 input floats: each movemask bit expands
     * to a full 0/1 byte through the 256-entry expansion table. */
    static const uint64_t mask_expand[256] = {
        0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000100ULL,
        0x0000000000000101ULL, 0x0000000000010000ULL, 0x0000000000010001ULL,
        0x0000000000010100ULL, 0x0000000000010101ULL, 0x0000000001000000ULL,
        0x0000000001000001ULL, 0x0000000001000100ULL, 0x0000000001000101ULL,
        0x0000000001010000ULL, 0x0000000001010001ULL, 0x0000000001010100ULL,
        0x0000000001010101ULL, 0x0000000100000000ULL, 0x0000000100000001ULL,
        0x0000000100000100ULL, 0x0000000100000101ULL, 0x0000000100010000ULL,
        0x0000000100010001ULL, 0x0000000100010100ULL, 0x0000000100010101ULL,
        0x0000000101000000ULL, 0x0000000101000001ULL, 0x0000000101000100ULL,
        0x0000000101000101ULL, 0x0000000101010000ULL, 0x0000000101010001ULL,
        0x0000000101010100ULL, 0x0000000101010101ULL, 0x0000010000000000ULL,
        0x0000010000000001ULL, 0x0000010000000100ULL, 0x0000010000000101ULL,
        0x0000010000010000ULL, 0x0000010000010001ULL, 0x0000010000010100ULL,
        0x0000010000010101ULL, 0x0000010001000000ULL, 0x0000010001000001ULL,
        0x0000010001000100ULL, 0x0000010001000101ULL, 0x0000010001010000ULL,
        0x0000010001010001ULL, 0x0000010001010100ULL, 0x0000010001010101ULL,
        0x0000010100000000ULL, 0x0000010100000001ULL, 0x0000010100000100ULL,
        0x0000010100000101ULL, 0x0000010100010000ULL, 0x0000010100010001ULL,
        0x0000010100010100ULL, 0x0000010100010101ULL, 0x0000010101000000ULL,
        0x0000010101000001ULL, 0x0000010101000100ULL, 0x0000010101000101ULL,
        0x0000010101010000ULL, 0x0000010101010001ULL, 0x0000010101010100ULL,
        0x0000010101010101ULL, 0x0001000000000000ULL, 0x0001000000000001ULL,
        0x0001000000000100ULL, 0x0001000000000101ULL, 0x0001000000010000ULL,
        0x0001000000010001ULL, 0x0001000000010100ULL, 0x0001000000010101ULL,
        0x0001000001000000ULL, 0x0001000001000001ULL, 0x0001000001000100ULL,
        0x0001000001000101ULL, 0x0001000001010000ULL, 0x0001000001010001ULL,
        0x0001000001010100ULL, 0x0001000001010101ULL, 0x0001000100000000ULL,
        0x0001000100000001ULL, 0x0001000100000100ULL, 0x0001000100000101ULL,
        0x0001000100010000ULL, 0x0001000100010001ULL, 0x0001000100010100ULL,
        0x0001000100010101ULL, 0x0001000101000000ULL, 0x0001000101000001ULL,
        0x0001000101000100ULL, 0x0001000101000101ULL, 0x0001000101010000ULL,
        0x0001000101010001ULL, 0x0001000101010100ULL, 0x0001000101010101ULL,
        0x0001010000000000ULL, 0x0001010000000001ULL, 0x0001010000000100ULL,
        0x0001010000000101ULL, 0x0001010000010000ULL, 0x0001010000010001ULL,
        0x0001010000010100ULL, 0x0001010000010101ULL, 0x0001010001000000ULL,
        0x0001010001000001ULL, 0x0001010001000100ULL, 0x0001010001000101ULL,
        0x0001010001010000ULL, 0x0001010001010001ULL, 0x0001010001010100ULL,
        0x0001010001010101ULL, 0x0001010100000000ULL, 0x0001010100000001ULL,
        0x0001010100000100ULL, 0x0001010100000101ULL, 0x0001010100010000ULL,
        0x0001010100010001ULL, 0x0001010100010100ULL, 0x0001010100010101ULL,
        0x0001010101000000ULL, 0x0001010101000001ULL, 0x0001010101000100ULL,
        0x0001010101000101ULL, 0x0001010101010000ULL, 0x0001010101010001ULL,
        0x0001010101010100ULL, 0x0001010101010101ULL, 0x0100000000000000ULL,
        0x0100000000000001ULL, 0x0100000000000100ULL, 0x0100000000000101ULL,
        0x0100000000010000ULL, 0x0100000000010001ULL, 0x0100000000010100ULL,
        0x0100000000010101ULL, 0x0100000001000000ULL, 0x0100000001000001ULL,
        0x0100000001000100ULL, 0x0100000001000101ULL, 0x0100000001010000ULL,
        0x0100000001010001ULL, 0x0100000001010100ULL, 0x0100000001010101ULL,
        0x0100000100000000ULL, 0x0100000100000001ULL, 0x0100000100000100ULL,
        0x0100000100000101ULL, 0x0100000100010000ULL, 0x0100000100010001ULL,
        0x0100000100010100ULL, 0x0100000100010101ULL, 0x0100000101000000ULL,
        0x0100000101000001ULL, 0x0100000101000100ULL, 0x0100000101000101ULL,
        0x0100000101010000ULL, 0x0100000101010001ULL, 0x0100000101010100ULL,
        0x0100000101010101ULL, 0x0100010000000000ULL, 0x0100010000000001ULL,
        0x0100010000000100ULL, 0x0100010000000101ULL, 0x0100010000010000ULL,
        0x0100010000010001ULL, 0x0100010000010100ULL, 0x0100010000010101ULL,
        0x0100010001000000ULL, 0x0100010001000001ULL, 0x0100010001000100ULL,
        0x0100010001000101ULL, 0x0100010001010000ULL, 0x0100010001010001ULL,
        0x0100010001010100ULL, 0x0100010001010101ULL, 0x0100010100000000ULL,
        0x0100010100000001ULL, 0x0100010100000100ULL, 0x0100010100000101ULL,
        0x0100010100010000ULL, 0x0100010100010001ULL, 0x0100010100010100ULL,
        0x0100010100010101ULL, 0x0100010101000000ULL, 0x0100010101000001ULL,
        0x0100010101000100ULL, 0x0100010101000101ULL, 0x0100010101010000ULL,
        0x0100010101010001ULL, 0x0100010101010100ULL, 0x0100010101010101ULL,
        0x0101000000000000ULL, 0x0101000000000001ULL, 0x0101000000000100ULL,
        0x0101000000000101ULL, 0x0101000000010000ULL, 0x0101000000010001ULL,
        0x0101000000010100ULL, 0x0101000000010101ULL, 0x0101000001000000ULL,
        0x0101000001000001ULL, 0x0101000001000100ULL, 0x0101000001000101ULL,
        0x0101000001010000ULL, 0x0101000001010001ULL, 0x0101000001010100ULL,
        0x0101000001010101ULL, 0x0101000100000000ULL, 0x0101000100000001ULL,
        0x0101000100000100ULL, 0x0101000100000101ULL, 0x0101000100010000ULL,
        0x0101000100010001ULL, 0x0101000100010100ULL, 0x0101000100010101ULL,
        0x0101000101000000ULL, 0x0101000101000001ULL, 0x0101000101000100ULL,
        0x0101000101000101ULL, 0x0101000101010000ULL, 0x0101000101010001ULL,
        0x0101000101010100ULL, 0x0101000101010101ULL, 0x0101010000000000ULL,
        0x0101010000000001ULL, 0x0101010000000100ULL, 0x0101010000000101ULL,
        0x0101010000010000ULL, 0x0101010000010001ULL, 0x0101010000010100ULL,
        0x0101010000010101ULL, 0x0101010001000000ULL, 0x0101010001000001ULL,
        0x0101010001000100ULL, 0x0101010001000101ULL, 0x0101010001010000ULL,
        0x0101010001010001ULL, 0x0101010001010100ULL, 0x0101010001010101ULL,
        0x0101010100000000ULL, 0x0101010100000001ULL, 0x0101010100000100ULL,
        0x0101010100000101ULL, 0x0101010100010000ULL, 0x0101010100010001ULL,
        0x0101010100010100ULL, 0x0101010100010101ULL, 0x0101010101000000ULL,
        0x0101010101000001ULL, 0x0101010101000100ULL, 0x0101010101000101ULL,
        0x0101010101010000ULL, 0x0101010101010001ULL, 0x0101010101010100ULL,
        0x0101010101010101ULL};
    const __m256 v_threshold = _mm256_set1_ps(threshold);
    const __m256 v_abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 v_infinity = _mm256_set1_ps(INFINITY);
    uint64_t index;
    for (index = 0u; index + 8u <= pixel_count; index += 8u) {
        __m256 values = _mm256_loadu_ps(prediction + (size_t)index);
        __m256 comparison =
            _mm256_cmp_ps(_mm256_and_ps(values, v_abs_mask), v_infinity, _CMP_NLT_UQ);
        if (_mm256_movemask_ps(comparison) != 0) {
            return -1; /* non-finite value (NaN or infinite) */
        }
        {
            int mask = _mm256_movemask_ps(
                _mm256_cmp_ps(values, v_threshold, _CMP_GT_OQ));
            uint64_t word = mask_expand[(uint32_t)mask];
            memcpy(bitmap + (size_t)index, &word, sizeof(word));
        }
    }
    for (; index < pixel_count; ++index) {
        float value = prediction[(size_t)index];
        if (!isfinite(value)) {
            return -1;
        }
        bitmap[(size_t)index] = value > threshold ? 1u : 0u;
    }
    return 0;
#else
    uint64_t index;
    (void)threshold;
    for (index = 0u; index < pixel_count; ++index) {
        float value = prediction[(size_t)index];
        if (!isfinite(value)) {
            return -1;
        }
        bitmap[(size_t)index] = value > threshold ? 1u : 0u;
    }
    return 0;
#endif
}

#if LW_COMPILES_AVX2_BITMAP && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_interior_bitmap_u8(const uint8_t* bitmap, uint8_t* interior,
                                uint32_t width, uint32_t height) {
    /* 4-neighborhood interior: interior[p] is non-zero only when all four
     * orthogonal neighbors are foreground. Border pixels stay zero, which
     * reproduces the per-pixel boundary probe exactly. */
    uint32_t y;
    memset(interior, 0, (size_t)width * height);
    if (width < 3u || height < 3u) return;
#if LW_COMPILES_AVX2_BITMAP
    for (y = 1u; y + 1u < height; ++y) {
        const uint8_t* previous = bitmap + (size_t)(y - 1u) * width;
        const uint8_t* current = bitmap + (size_t)y * width;
        const uint8_t* next = bitmap + (size_t)(y + 1u) * width;
        uint8_t* destination = interior + (size_t)y * width;
        uint32_t x = 1u;
        for (; x + 33u <= width; x += 32u) {
            __m256i vertical = _mm256_and_si256(
                _mm256_loadu_si256((const __m256i*)(const void*)(previous + x)),
                _mm256_loadu_si256((const __m256i*)(const void*)(next + x)));
            __m256i horizontal = _mm256_and_si256(
                _mm256_loadu_si256((const __m256i*)(const void*)(current + x - 1u)),
                _mm256_loadu_si256((const __m256i*)(const void*)(current + x + 1u)));
            _mm256_storeu_si256((__m256i*)(void*)(destination + x),
                                _mm256_and_si256(vertical, horizontal));
        }
        for (; x + 1u < width; ++x) {
            if (previous[x - 1u] != 0u && previous[x + 1u] != 0u &&
                current[x - 1u] != 0u && current[x + 1u] != 0u &&
                next[x - 1u] != 0u && next[x + 1u] != 0u) {
                destination[x] = 1u;
            }
        }
    }
#else
    for (y = 1u; y + 1u < height; ++y) {
        const uint8_t* previous = bitmap + (size_t)(y - 1u) * width;
        const uint8_t* current = bitmap + (size_t)y * width;
        const uint8_t* next = bitmap + (size_t)(y + 1u) * width;
        uint8_t* destination = interior + (size_t)y * width;
        uint32_t x;
        for (x = 1u; x + 1u < width; ++x) {
            if (previous[x - 1u] != 0u && previous[x + 1u] != 0u &&
                current[x - 1u] != 0u && current[x + 1u] != 0u &&
                next[x - 1u] != 0u && next[x + 1u] != 0u) {
                destination[x] = 1u;
            }
        }
    }
#endif
}
