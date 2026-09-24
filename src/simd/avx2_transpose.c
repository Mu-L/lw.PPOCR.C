#include "simd_kernels.h"

#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LW_TRANSPOSE_X86 1
#include <immintrin.h>
#else
#define LW_TRANSPOSE_X86 0
#endif

#if LW_TRANSPOSE_X86 && (defined(__GNUC__) || defined(__clang__))
#define LW_TRANSPOSE_AVX2 __attribute__((target("avx2")))
#else
#define LW_TRANSPOSE_AVX2
#endif

#if LW_TRANSPOSE_X86
/* In-register 8x8 float transpose. Pure data movement: bit-identical to any
 * scalar reordering of the same elements. */
LW_TRANSPOSE_AVX2
static void transpose8x8_f32(__m256 rows[8]) {
    __m256 t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    __m256 t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    __m256 t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    __m256 t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    __m256 t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    __m256 t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    __m256 t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    __m256 t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    rows[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    rows[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    rows[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    rows[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    rows[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    rows[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    rows[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    rows[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}
#endif

/* Transpose a rows x cols row-major matrix into a cols x rows row-major
 * matrix: output[col * rows + row] = input[row * cols + col]. */
LW_TRANSPOSE_AVX2
void lw_avx2_transpose_2d_f32(const float* input, float* output,
                              uint32_t rows, uint32_t cols) {
#if LW_TRANSPOSE_X86
    uint32_t row = 0u;
    uint32_t col;
    uint32_t r;
    uint32_t c;
    if (input == NULL || output == NULL) {
        return;
    }
    for (; row + 8u <= rows; row += 8u) {
        for (col = 0u; col + 8u <= cols; col += 8u) {
            __m256 block[8];
            uint32_t k;
            for (k = 0u; k < 8u; ++k) {
                block[k] = _mm256_loadu_ps(input + (size_t)(row + k) * cols + col);
            }
            transpose8x8_f32(block);
            for (k = 0u; k < 8u; ++k) {
                _mm256_storeu_ps(output + (size_t)(col + k) * rows + row, block[k]);
            }
        }
        /* Right edge: remaining columns for the blocked rows. */
        for (; col < cols; ++col) {
            for (r = 0u; r < 8u; ++r) {
                output[(size_t)col * rows + row + r] = input[(size_t)(row + r) * cols + col];
            }
        }
    }
    /* Bottom edge: remaining rows across all columns. */
    for (; row < rows; ++row) {
        for (c = 0u; c < cols; ++c) {
            output[(size_t)c * rows + row] = input[(size_t)row * cols + c];
        }
    }
#else
    uint32_t r;
    uint32_t c;
    if (input == NULL || output == NULL) {
        return;
    }
    for (r = 0u; r < rows; ++r) {
        for (c = 0u; c < cols; ++c) {
            output[(size_t)c * rows + r] = input[(size_t)r * cols + c];
        }
    }
#endif
}
