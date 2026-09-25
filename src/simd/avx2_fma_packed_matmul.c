#include "simd_kernels.h"

#include "packed_matmul_internal.h"

#include <math.h>
#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_FMA_MATMUL 1
#else
#  define LW_COMPILES_AVX2_FMA_MATMUL 0
#endif

#if LW_COMPILES_AVX2_FMA_MATMUL && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_fma_packed_matmul_bias_argmax_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    uint32_t* best_indices, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns) {
#if LW_COMPILES_AVX2_FMA_MATMUL
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    uint32_t batch;
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t row;
        for (row = 0u; row + 4u <= rows; row += 4u) {
            uint32_t column_panel;
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                __m256 accumulators[8] = {
                    _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
                uint32_t inner;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)column_panel * inner_dimension + inner) *
                                 LW_PACKED_MATMUL_COLUMN_TILE);
                    const uint64_t input_base =
                        ((uint64_t)batch * rows + row) * inner_dimension + inner;
                    const __m256 weight_low = _mm256_loadu_ps(packed);
                    const __m256 weight_high = _mm256_loadu_ps(packed + 8u);
                    __m256 input_value = _mm256_set1_ps(input[(size_t)input_base]);
                    accumulators[0] = _mm256_fmadd_ps(input_value, weight_low, accumulators[0]);
                    accumulators[1] = _mm256_fmadd_ps(input_value, weight_high, accumulators[1]);
                    input_value = _mm256_set1_ps(input[(size_t)(input_base + inner_dimension)]);
                    accumulators[2] = _mm256_fmadd_ps(input_value, weight_low, accumulators[2]);
                    accumulators[3] = _mm256_fmadd_ps(input_value, weight_high, accumulators[3]);
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 2u * inner_dimension)]);
                    accumulators[4] = _mm256_fmadd_ps(input_value, weight_low, accumulators[4]);
                    accumulators[5] = _mm256_fmadd_ps(input_value, weight_high, accumulators[5]);
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 3u * inner_dimension)]);
                    accumulators[6] = _mm256_fmadd_ps(input_value, weight_low, accumulators[6]);
                    accumulators[7] = _mm256_fmadd_ps(input_value, weight_high, accumulators[7]);
                }
                {
                    const uint32_t column_base = column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                    const uint32_t valid_columns =
                        columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                            ? columns - column_base
                            : LW_PACKED_MATMUL_COLUMN_TILE;
                    uint32_t current_row;
                    for (current_row = 0u; current_row < 4u; ++current_row) {
                        const uint64_t result_row = (uint64_t)batch * rows + row + current_row;
                        float* destination =
                            output + (size_t)(result_row * columns + column_base);
                        float values[LW_PACKED_MATMUL_COLUMN_TILE];
                        uint32_t best_index = column_base == 0u
                                                  ? 0u
                                                  : best_indices[(size_t)result_row];
                        float best_value = column_base == 0u
                                               ? 0.0f
                                               : output[(size_t)(result_row * columns +
                                                                 best_index)];
                        uint32_t lane;
                        _mm256_storeu_ps(values, accumulators[current_row * 2u]);
                        _mm256_storeu_ps(values + 8u, accumulators[current_row * 2u + 1u]);
                        for (lane = 0u; lane < valid_columns; ++lane) {
                            const float value = values[lane] + bias[column_base + lane];
                            destination[lane] = value;
                            if ((column_base != 0u || lane != 0u) && value > best_value) {
                                best_value = value;
                                best_index = column_base + lane;
                            } else if (column_base == 0u && lane == 0u) {
                                best_value = value;
                            }
                        }
                        best_indices[(size_t)result_row] = best_index;
                    }
                }
            }
        }
    }
#else
    (void)input;
    (void)packed_weights;
    (void)bias;
    (void)output;
    (void)best_indices;
    (void)batch_count;
    (void)rows;
    (void)inner_dimension;
    (void)columns;
#endif
}

#if LW_COMPILES_AVX2_FMA_MATMUL
/* Strict-greater vector running max: replaces the value and its index only
 * when the new value is strictly greater, which keeps the lowest column index
 * on ties (same semantics as the stored-logits kernel). */
static void update_max256(__m256* maxima, __m256* maxima_indices, __m256 values,
                          uint32_t column_base, __m256 lane_offsets) {
    const __m256 candidates =
        _mm256_add_ps(_mm256_set1_ps((float)column_base), lane_offsets);
    const __m256 mask = _mm256_cmp_ps(values, *maxima, _CMP_GT_OQ);
    *maxima = _mm256_max_ps(*maxima, values);
    *maxima_indices = _mm256_blendv_ps(*maxima_indices, candidates, mask);
}

/* Scalar horizontal reduce of one 8-lane max/index pair; strict-greater
 * keeps the lowest lane index on ties. */
static void reduce_max8(__m256 values, __m256 indices, float* best_value,
                        uint32_t* best_index) {
    float lane_values[8];
    float lane_indices[8];
    uint32_t lane;
    _mm256_storeu_ps(lane_values, values);
    _mm256_storeu_ps(lane_indices, indices);
    for (lane = 0u; lane < 8u; ++lane) {
        if (lane_values[lane] > *best_value) {
            *best_value = lane_values[lane];
            *best_index = (uint32_t)lane_indices[lane];
        }
    }
}
#endif

#if LW_COMPILES_AVX2_FMA_MATMUL && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_fma_packed_matmul_argmax_scores_f32(
    const float* input, const float* packed_weights, const float* bias,
    uint32_t* best_indices, float* scores, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns) {
#if LW_COMPILES_AVX2_FMA_MATMUL
    /* Panel-outer loop order with row chunks: the packed weights for one
     * column panel stay cache-resident while all rows of the chunk are
     * processed, so the weight matrix streams from memory once per chunk
     * instead of once per 4-row block.  Column panels are disjoint output
     * column blocks and every output element accumulates over the inner
     * dimension in the same order as the row-outer form, and each row's
     * running maximum still visits panels in ascending order, so results
     * are bit-identical. */
#define LW_CTC_HEAD_ROW_CHUNK 128u
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    const uint32_t aligned_rows = rows & ~3u;
    const int has_partial_panel = (columns % LW_PACKED_MATMUL_COLUMN_TILE) != 0u;
    const __m256 lane_offsets = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f,
                                              3.0f, 2.0f, 1.0f, 0.0f);
    uint32_t batch;
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t chunk_begin;
        for (chunk_begin = 0u; chunk_begin < aligned_rows;
             chunk_begin += LW_CTC_HEAD_ROW_CHUNK) {
            uint32_t chunk_rows = aligned_rows - chunk_begin;
            __m256 maxima[LW_CTC_HEAD_ROW_CHUNK * 2u];
            __m256 maxima_indices[LW_CTC_HEAD_ROW_CHUNK * 2u];
            float partial_values[LW_CTC_HEAD_ROW_CHUNK][LW_PACKED_MATMUL_COLUMN_TILE];
            uint32_t chunk_row;
            uint32_t column_panel;
            if (chunk_rows > LW_CTC_HEAD_ROW_CHUNK) {
                chunk_rows = LW_CTC_HEAD_ROW_CHUNK;
            }
            for (chunk_row = 0u; chunk_row < chunk_rows; ++chunk_row) {
                const __m256 negative_infinity = _mm256_set1_ps(-(float)INFINITY);
                maxima[chunk_row * 2u] = negative_infinity;
                maxima[chunk_row * 2u + 1u] = negative_infinity;
                maxima_indices[chunk_row * 2u] = _mm256_setzero_ps();
                maxima_indices[chunk_row * 2u + 1u] = _mm256_setzero_ps();
            }
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                const uint32_t column_base =
                    column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                const uint32_t valid_columns =
                    columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                        ? columns - column_base
                        : LW_PACKED_MATMUL_COLUMN_TILE;
                const int full_panel = valid_columns == LW_PACKED_MATMUL_COLUMN_TILE;
                for (chunk_row = 0u; chunk_row < chunk_rows; chunk_row += 4u) {
                    const uint32_t row = chunk_begin + chunk_row;
                    __m256 accumulators[8] = {
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps()};
                    uint32_t inner;
                    uint32_t current_row;
                    for (inner = 0u; inner < inner_dimension; ++inner) {
                        const float* packed =
                            packed_weights +
                            (size_t)(((uint64_t)column_panel * inner_dimension + inner) *
                                     LW_PACKED_MATMUL_COLUMN_TILE);
                        const uint64_t input_base =
                            ((uint64_t)batch * rows + row) * inner_dimension + inner;
                        const __m256 weight_low = _mm256_loadu_ps(packed);
                        const __m256 weight_high = _mm256_loadu_ps(packed + 8u);
                        __m256 input_value = _mm256_set1_ps(input[(size_t)input_base]);
                        accumulators[0] = _mm256_fmadd_ps(input_value, weight_low, accumulators[0]);
                        accumulators[1] = _mm256_fmadd_ps(input_value, weight_high, accumulators[1]);
                        input_value = _mm256_set1_ps(input[(size_t)(input_base + inner_dimension)]);
                        accumulators[2] = _mm256_fmadd_ps(input_value, weight_low, accumulators[2]);
                        accumulators[3] = _mm256_fmadd_ps(input_value, weight_high, accumulators[3]);
                        input_value =
                            _mm256_set1_ps(input[(size_t)(input_base + 2u * inner_dimension)]);
                        accumulators[4] = _mm256_fmadd_ps(input_value, weight_low, accumulators[4]);
                        accumulators[5] = _mm256_fmadd_ps(input_value, weight_high, accumulators[5]);
                        input_value =
                            _mm256_set1_ps(input[(size_t)(input_base + 3u * inner_dimension)]);
                        accumulators[6] = _mm256_fmadd_ps(input_value, weight_low, accumulators[6]);
                        accumulators[7] = _mm256_fmadd_ps(input_value, weight_high, accumulators[7]);
                    }
                    if (bias != NULL) {
                        __m256 bias_low;
                        __m256 bias_high;
                        if (full_panel) {
                            bias_low = _mm256_loadu_ps(bias + column_base);
                            bias_high = _mm256_loadu_ps(bias + column_base + 8u);
                        } else {
                            float bias_values[LW_PACKED_MATMUL_COLUMN_TILE] = {0.0f};
                            uint32_t lane;
                            for (lane = 0u; lane < valid_columns; ++lane) {
                                bias_values[lane] = bias[column_base + lane];
                            }
                            bias_low = _mm256_loadu_ps(bias_values);
                            bias_high = _mm256_loadu_ps(bias_values + 8u);
                        }
                        for (current_row = 0u; current_row < 4u; ++current_row) {
                            accumulators[current_row * 2u] =
                                _mm256_add_ps(accumulators[current_row * 2u], bias_low);
                            accumulators[current_row * 2u + 1u] =
                                _mm256_add_ps(accumulators[current_row * 2u + 1u], bias_high);
                        }
                    }
                    if (full_panel) {
                        for (current_row = 0u; current_row < 4u; ++current_row) {
                            uint32_t state_row = chunk_row + current_row;
                            update_max256(&maxima[state_row * 2u],
                                          &maxima_indices[state_row * 2u],
                                          accumulators[current_row * 2u], column_base,
                                          lane_offsets);
                            update_max256(&maxima[state_row * 2u + 1u],
                                          &maxima_indices[state_row * 2u + 1u],
                                          accumulators[current_row * 2u + 1u],
                                          column_base + 8u, lane_offsets);
                        }
                    } else {
                        /* Partial final panel: keep the raw values per row;
                         * the scalar merge runs after the vector reduction,
                         * exactly like the row-outer form. */
                        for (current_row = 0u; current_row < 4u; ++current_row) {
                            _mm256_storeu_ps(partial_values[chunk_row + current_row],
                                             accumulators[current_row * 2u]);
                            _mm256_storeu_ps(partial_values[chunk_row + current_row] + 8u,
                                             accumulators[current_row * 2u + 1u]);
                        }
                    }
                }
            }
            for (chunk_row = 0u; chunk_row < chunk_rows; ++chunk_row) {
                const uint64_t result_row =
                    (uint64_t)batch * rows + chunk_begin + chunk_row;
                float best_value = -(float)INFINITY;
                uint32_t best_index = 0u;
                reduce_max8(maxima[chunk_row * 2u], maxima_indices[chunk_row * 2u],
                            &best_value, &best_index);
                reduce_max8(maxima[chunk_row * 2u + 1u],
                            maxima_indices[chunk_row * 2u + 1u],
                            &best_value, &best_index);
                if (has_partial_panel) {
                    const uint32_t column_base =
                        (column_panels - 1u) * LW_PACKED_MATMUL_COLUMN_TILE;
                    const uint32_t valid_columns = columns - column_base;
                    uint32_t lane;
                    for (lane = 0u; lane < valid_columns; ++lane) {
                        const float value = partial_values[chunk_row][lane];
                        if ((column_base != 0u || lane != 0u) &&
                            value > best_value) {
                            best_value = value;
                            best_index = column_base + lane;
                        } else if (column_base == 0u && lane == 0u) {
                            best_value = value;
                        }
                    }
                }
                scores[(size_t)result_row] = best_value;
                best_indices[(size_t)result_row] = best_index;
            }
        }
    }
#else
    (void)input;
    (void)packed_weights;
    (void)bias;
    (void)best_indices;
    (void)scores;
    (void)batch_count;
    (void)rows;
    (void)inner_dimension;
    (void)columns;
#endif
}
