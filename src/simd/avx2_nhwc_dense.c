#include "nhwc_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LW_NHWC_X86 1
#include <immintrin.h>
#endif

static int dense_mul_u64(uint64_t left, uint64_t right, uint64_t* result) {
    if (result == NULL || (right != 0u && left > UINT64_MAX / right)) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int dense_add_u64(uint64_t left, uint64_t right, uint64_t* result) {
    if (result == NULL || left > UINT64_MAX - right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int dense_align64(uint64_t value, uint64_t* result) {
    if (value > UINT64_MAX - 63u) {
        return 0;
    }
    *result = (value + 63u) & ~UINT64_C(63);
    return 1;
}

static int dense_geometry(const lw_nhwc_dense_desc* desc, uint64_t* taps,
                          uint64_t* k_total, uint64_t* patch_width,
                          uint64_t* x_tiles) {
    uint64_t expected_height;
    uint64_t expected_width;
    if (desc == NULL || desc->batch == 0u || desc->input_channels == 0u ||
        desc->input_height == 0u || desc->input_width == 0u ||
        desc->output_channels == 0u || desc->output_height == 0u ||
        desc->output_width == 0u || desc->kernel_h == 0u || desc->kernel_w == 0u ||
        desc->stride_h == 0u || desc->stride_w == 0u ||
        desc->output_channels % LW_NHWC_OC_BLOCK != 0u) {
        return 0;
    }
    if (!dense_mul_u64(desc->kernel_h, desc->kernel_w, taps) ||
        !dense_mul_u64(*taps, desc->input_channels, k_total) ||
        !dense_mul_u64(LW_NHWC_PIXEL_TILE - 1u, desc->stride_w, patch_width) ||
        !dense_add_u64(*patch_width, desc->kernel_w, patch_width)) {
        return 0;
    }
    if (desc->input_height + (uint64_t)desc->pad_top + desc->pad_bottom < desc->kernel_h ||
        desc->input_width + (uint64_t)desc->pad_left + desc->pad_right < desc->kernel_w) {
        return 0;
    }
    expected_height = (desc->input_height + (uint64_t)desc->pad_top + desc->pad_bottom - desc->kernel_h) /
                      desc->stride_h + 1u;
    expected_width = (desc->input_width + (uint64_t)desc->pad_left + desc->pad_right - desc->kernel_w) /
                     desc->stride_w + 1u;
    if (expected_height != desc->output_height || expected_width != desc->output_width ||
        *patch_width > UINT32_MAX ||
        !dense_add_u64(desc->output_width, LW_NHWC_PIXEL_TILE - 1u, x_tiles)) {
        return 0;
    }
    *x_tiles /= LW_NHWC_PIXEL_TILE;
    return *k_total <= UINT32_MAX && *taps <= UINT32_MAX;
}

int lw_nhwc_dense_scratch_bytes(const lw_nhwc_dense_desc* desc, uint64_t* scratch_bytes) {
    uint64_t taps;
    uint64_t k_total;
    uint64_t patch_width;
    uint64_t x_tiles;
    uint64_t patch_floats;
    uint64_t partial_floats;
    uint64_t bytes;
    uint64_t aligned;
    if (scratch_bytes == NULL || !dense_geometry(desc, &taps, &k_total, &patch_width, &x_tiles)) {
        return 0;
    }
    (void)k_total;
    if (!dense_mul_u64(desc->kernel_h, patch_width, &patch_floats) ||
        !dense_mul_u64(patch_floats, desc->input_channels, &patch_floats) ||
        !dense_mul_u64(1u, LW_NHWC_PARTIAL_FLOATS, &partial_floats) ||
        !dense_mul_u64(taps, sizeof(int32_t) * 2u, &bytes) ||
        !dense_align64(bytes, &aligned) ||
        !dense_mul_u64(patch_floats, sizeof(float), &patch_floats) ||
        !dense_add_u64(aligned, patch_floats, &bytes) ||
        !dense_align64(bytes, &aligned) ||
        !dense_mul_u64(partial_floats, sizeof(float), &partial_floats) ||
        !dense_add_u64(aligned, partial_floats, &bytes) ||
        bytes > SIZE_MAX) {
        return 0;
    }
    *scratch_bytes = bytes;
    return 1;
}

static int dense_scratch_views(const lw_nhwc_dense_desc* desc, void* scratch,
                               uint64_t scratch_bytes, int32_t** tap_offsets,
                               int32_t** patch_offsets, float** patch, float** partial) {
    uint64_t required;
    uint64_t taps;
    uint64_t k_total;
    uint64_t patch_width;
    uint64_t x_tiles;
    uint64_t bytes;
    uint64_t aligned;
    uint64_t patch_floats;
    if (scratch == NULL || !lw_nhwc_dense_scratch_bytes(desc, &required) ||
        scratch_bytes < required || !dense_geometry(desc, &taps, &k_total, &patch_width, &x_tiles)) {
        return 0;
    }
    (void)k_total;
    (void)x_tiles;
    if (!dense_mul_u64(taps, sizeof(int32_t) * 2u, &bytes) ||
        !dense_align64(bytes, &aligned) ||
        !dense_mul_u64(desc->kernel_h, patch_width, &patch_floats) ||
        !dense_mul_u64(patch_floats, desc->input_channels, &patch_floats) ||
        !dense_mul_u64(patch_floats, sizeof(float), &patch_floats)) {
        return 0;
    }
    *tap_offsets = (int32_t*)scratch;
    *patch_offsets = (int32_t*)((uint8_t*)scratch + (size_t)taps * sizeof(int32_t));
    *patch = (float*)((uint8_t*)scratch + (size_t)aligned);
    if (!dense_align64(aligned + patch_floats, &bytes)) {
        return 0;
    }
    *partial = (float*)((uint8_t*)scratch + (size_t)bytes);
    return 1;
}

static int dense_build_offsets(const lw_nhwc_dense_desc* desc, int32_t* tap_offsets,
                               int32_t* patch_offsets, uint32_t patch_width) {
    uint32_t tap = 0u;
    uint32_t ky;
    for (ky = 0u; ky < desc->kernel_h; ++ky) {
        uint32_t kx;
        for (kx = 0u; kx < desc->kernel_w; ++kx) {
            uint64_t input_offset = ((uint64_t)ky * desc->input_width + kx) *
                                    desc->input_channels;
            uint64_t patch_offset = ((uint64_t)ky * patch_width + kx) *
                                    desc->input_channels;
            if (input_offset > INT32_MAX || patch_offset > INT32_MAX) {
                return 0;
            }
            tap_offsets[tap] = (int32_t)input_offset;
            patch_offsets[tap] = (int32_t)patch_offset;
            ++tap;
        }
    }
    return 1;
}

static void dense_gather_patch(const float* input, float* patch,
                               const lw_nhwc_dense_desc* desc, int32_t iy0,
                               int32_t ix0, uint32_t patch_width) {
    uint64_t patch_elements = (uint64_t)desc->kernel_h * patch_width * desc->input_channels;
    uint32_t py;
    memset(patch, 0, (size_t)patch_elements * sizeof(float));
    for (py = 0u; py < desc->kernel_h; ++py) {
        int32_t iy = iy0 + (int32_t)py;
        uint32_t px;
        if (iy < 0 || iy >= (int32_t)desc->input_height) {
            continue;
        }
        for (px = 0u; px < patch_width; ++px) {
            int32_t ix = ix0 + (int32_t)px;
            const float* source;
            float* destination;
            if (ix < 0 || ix >= (int32_t)desc->input_width) {
                continue;
            }
            source = input + (((size_t)iy * desc->input_width + (uint32_t)ix) *
                              desc->input_channels);
            destination = patch + (((size_t)py * patch_width + px) * desc->input_channels);
            memcpy(destination, source, (size_t)desc->input_channels * sizeof(float));
        }
    }
}

static float dense_apply_activation(float value, uint16_t activation) {
    if (activation == LW_NHWC_ACT_RELU) {
        return value > 0.0f ? value : 0.0f;
    }
    if (activation == LW_NHWC_ACT_HARDSWISH) {
        float gate = value + 3.0f;
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 6.0f) gate = 6.0f;
        return value * gate * (1.0f / 6.0f);
    }
    return value;
}

#if !defined(LW_NHWC_X86)
static void dense_tile_scalar(const float* const* row_ptrs, uint32_t rows,
                              const int32_t* offsets, uint32_t taps,
                              const float* packed, uint32_t k0, uint32_t k_end,
                              uint32_t k_total, const lw_nhwc_epilogue* epilogue,
                              float* output, uint32_t output_stride, float* partial) {
    float sums[LW_NHWC_PIXEL_TILE][LW_NHWC_OC_BLOCK];
    uint32_t row;
    uint32_t lane;
    if (k0 == 0u) {
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                sums[row][lane] = epilogue != NULL && epilogue->bias != NULL
                    ? epilogue->bias[lane] : 0.0f;
            }
        }
    } else {
        memcpy(sums, partial, sizeof(sums));
    }
    for (uint32_t k = k0; k < k_end; ++k) {
        uint32_t ic = k / taps;
        uint32_t tap = k - ic * taps;
        const float* weights = packed + (size_t)(k - k0) * LW_NHWC_OC_BLOCK;
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            float value = row_ptrs[row][offsets[tap] + (int32_t)ic];
            for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                sums[row][lane] += value * weights[lane];
            }
        }
    }
    if (k_end < k_total) {
        memcpy(partial, sums, sizeof(sums));
        return;
    }
    for (row = 0u; row < rows; ++row) {
        float* destination = output + (size_t)row * output_stride;
        const float* residual = epilogue == NULL || epilogue->residual == NULL
            ? NULL : epilogue->residual + (size_t)row * output_stride;
        for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
            float value = sums[row][lane] + (residual == NULL ? 0.0f : residual[lane]);
            destination[lane] = dense_apply_activation(value,
                epilogue == NULL ? LW_NHWC_ACT_NONE : epilogue->activation);
        }
    }
}

#endif

#if defined(LW_NHWC_X86)
#if defined(__GNUC__) || defined(__clang__)
#define LW_NHWC_DENSE_AVX2 __attribute__((target("avx2,fma")))
#else
#define LW_NHWC_DENSE_AVX2
#endif

LW_NHWC_DENSE_AVX2
static void dense_tile_avx2(const float* const* row_ptrs, uint32_t rows,
                            const int32_t* offsets, uint32_t taps,
                            const float* packed, uint32_t k0, uint32_t k_end,
                            uint32_t k_total, const lw_nhwc_epilogue* epilogue,
                            float* output, uint32_t output_stride, float* partial) {
    __m256 lo[LW_NHWC_PIXEL_TILE];
    __m256 hi[LW_NHWC_PIXEL_TILE];
    __m256 zero = _mm256_setzero_ps();
    uint32_t row;
    if (k0 == 0u) {
        __m256 bias_lo = epilogue != NULL && epilogue->bias != NULL
            ? _mm256_loadu_ps(epilogue->bias) : zero;
        __m256 bias_hi = epilogue != NULL && epilogue->bias != NULL
            ? _mm256_loadu_ps(epilogue->bias + 8u) : zero;
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            lo[row] = bias_lo;
            hi[row] = bias_hi;
        }
    } else {
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            lo[row] = _mm256_loadu_ps(partial + row * 16u);
            hi[row] = _mm256_loadu_ps(partial + row * 16u + 8u);
        }
    }
    for (uint32_t k = k0; k < k_end; ++k) {
        uint32_t ic = k / taps;
        uint32_t tap = k - ic * taps;
        const float* weights = packed + (size_t)(k - k0) * 16u;
        __m256 w0 = _mm256_loadu_ps(weights);
        __m256 w1 = _mm256_loadu_ps(weights + 8u);
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            __m256 value = _mm256_set1_ps(row_ptrs[row][offsets[tap] + (int32_t)ic]);
            lo[row] = _mm256_fmadd_ps(value, w0, lo[row]);
            hi[row] = _mm256_fmadd_ps(value, w1, hi[row]);
        }
    }
    if (k_end < k_total) {
        for (row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
            _mm256_storeu_ps(partial + row * 16u, lo[row]);
            _mm256_storeu_ps(partial + row * 16u + 8u, hi[row]);
        }
        return;
    }
    for (row = 0u; row < rows; ++row) {
        float* destination = output + (size_t)row * output_stride;
        if (epilogue != NULL && epilogue->residual != NULL) {
            __m256 residual_lo = _mm256_loadu_ps(epilogue->residual + row * output_stride);
            __m256 residual_hi = _mm256_loadu_ps(epilogue->residual + row * output_stride + 8u);
            lo[row] = _mm256_add_ps(lo[row], residual_lo);
            hi[row] = _mm256_add_ps(hi[row], residual_hi);
        }
        if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_RELU) {
            lo[row] = _mm256_max_ps(lo[row], zero);
            hi[row] = _mm256_max_ps(hi[row], zero);
        }
        _mm256_storeu_ps(destination, lo[row]);
        _mm256_storeu_ps(destination + 8u, hi[row]);
    }
}
#endif


lw_status lw_avx2_fma_nhwc_dense_f32(const float* input, const float* packed_weights,
                                     const lw_nhwc_epilogue* epilogue, float* output,
                                     const lw_nhwc_dense_desc* desc, void* scratch,
                                     uint64_t scratch_bytes) {
    int32_t* tap_offsets;
    int32_t* patch_offsets;
    float* patch;
    float* partial;
    uint64_t taps64;
    uint64_t k_total64;
    uint64_t patch_width64;
    uint64_t x_tiles64;
    uint32_t kc;
    if (input == NULL || packed_weights == NULL || output == NULL || desc == NULL ||
        !dense_geometry(desc, &taps64, &k_total64, &patch_width64, &x_tiles64) ||
        !dense_scratch_views(desc, scratch, scratch_bytes, &tap_offsets, &patch_offsets,
                             &patch, &partial) ||
        !dense_build_offsets(desc, tap_offsets, patch_offsets, (uint32_t)patch_width64)) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_GELU) {
        return LW_STATUS_UNSUPPORTED;
    }
    kc = desc->dense_kc == 0u ? (uint32_t)k_total64 : desc->dense_kc;
    if (kc == 0u) return LW_STATUS_INVALID_ARGUMENT;
    for (uint32_t batch = 0u; batch < desc->batch; ++batch) {
        const float* input_batch = input + (size_t)batch * desc->input_height *
                                   desc->input_width * desc->input_channels;
        float* output_batch = output + (size_t)batch * desc->output_height *
                              desc->output_width * desc->output_channels;
        for (uint32_t oy = 0u; oy < desc->output_height; ++oy) {
            int32_t iy0 = (int32_t)((uint64_t)oy * desc->stride_h) - (int32_t)desc->pad_top;
            for (uint32_t tile = 0u; tile < (uint32_t)x_tiles64; ++tile) {
                uint32_t ox = tile * LW_NHWC_PIXEL_TILE;
                uint32_t rows = desc->output_width - ox;
                int32_t ix0 = (int32_t)((uint64_t)ox * desc->stride_w) - (int32_t)desc->pad_left;
                int interior = iy0 >= 0 && iy0 + (int32_t)desc->kernel_h <= (int32_t)desc->input_height &&
                               ix0 >= 0 && ix0 + (int32_t)((LW_NHWC_PIXEL_TILE - 1u) * desc->stride_w + desc->kernel_w) <=
                               (int32_t)desc->input_width;
                const float* row_ptrs[LW_NHWC_PIXEL_TILE];
                if (rows > LW_NHWC_PIXEL_TILE) rows = LW_NHWC_PIXEL_TILE;
                if (interior) {
                    for (uint32_t row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
                        row_ptrs[row] = input_batch + (((size_t)iy0 * desc->input_width +
                            (uint32_t)(ix0 + (int32_t)row * desc->stride_w)) * desc->input_channels);
                    }
                } else {
                    dense_gather_patch(input_batch, patch, desc, iy0, ix0, (uint32_t)patch_width64);
                    for (uint32_t row = 0u; row < LW_NHWC_PIXEL_TILE; ++row) {
                        row_ptrs[row] = patch + (size_t)row * desc->stride_w * desc->input_channels;
                    }
                }
                for (uint32_t block = 0u; block < desc->output_channels / LW_NHWC_OC_BLOCK; ++block) {
                    const float* packed_block = packed_weights +
                        (size_t)block * k_total64 * LW_NHWC_OC_BLOCK;
                    float* destination = output_batch + ((size_t)oy * desc->output_width + ox) *
                                         desc->output_channels + block * LW_NHWC_OC_BLOCK;
                    lw_nhwc_epilogue local_epilogue = epilogue == NULL
                        ? (lw_nhwc_epilogue){0} : *epilogue;
                    if (local_epilogue.bias != NULL) local_epilogue.bias += block * LW_NHWC_OC_BLOCK;
                    if (local_epilogue.residual != NULL) {
                        local_epilogue.residual += ((size_t)batch * desc->output_height * desc->output_width +
                            (size_t)oy * desc->output_width + ox) * desc->output_channels +
                            block * LW_NHWC_OC_BLOCK;
                    }
                    for (uint32_t k0 = 0u; k0 < (uint32_t)k_total64; k0 += kc) {
                        uint32_t k_end = k0 + kc;
                        if (k_end > (uint32_t)k_total64) k_end = (uint32_t)k_total64;
#if defined(LW_NHWC_X86)
                        dense_tile_avx2(row_ptrs, rows,
                            interior ? tap_offsets : patch_offsets, (uint32_t)taps64,
                            packed_block + (size_t)k0 * LW_NHWC_OC_BLOCK, k0, k_end,
                            (uint32_t)k_total64, &local_epilogue, destination,
                            desc->output_channels, partial);
#else
                        dense_tile_scalar(row_ptrs, rows,
                            interior ? tap_offsets : patch_offsets, (uint32_t)taps64,
                            packed_block + (size_t)k0 * LW_NHWC_OC_BLOCK, k0, k_end,
                            (uint32_t)k_total64, &local_epilogue, destination,
                            desc->output_channels, partial);
#endif
                    }
                }
            }
        }
    }
    return LW_STATUS_OK;
}
