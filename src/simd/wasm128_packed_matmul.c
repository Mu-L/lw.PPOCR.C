#include "../kernels/packed_matmul_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

#define LW_CTC_PANEL LW_PACKED_MATMUL_COLUMN_TILE
#define LW_CTC_ROWS 4u

/* The packed layout is [panel][inner][16]. Keep panels outermost so their
 * weights stay cache-resident across row tiles, without a logits tensor. */
static void project_panel(const float* input, const float* packed_weights,
                          const float* bias, float values[LW_CTC_ROWS][LW_CTC_PANEL],
                          uint32_t rows, uint32_t inner_dimension,
                          uint32_t column_base, uint32_t valid_columns) {
    const float* panel = packed_weights +
        (size_t)(column_base / LW_CTC_PANEL) * inner_dimension * LW_CTC_PANEL;
#if defined(__wasm_simd128__)
    v128_t accumulators[LW_CTC_ROWS][4];
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t group = 0u; group < 4u; ++group) {
            accumulators[row][group] = wasm_f32x4_splat(0.0f);
        }
    }
    for (uint32_t inner = 0u; inner < inner_dimension; ++inner) {
        v128_t weights[4];
        const float* packed = panel + (size_t)inner * LW_CTC_PANEL;
        for (uint32_t group = 0u; group < 4u; ++group) {
            weights[group] = wasm_v128_load(packed + group * 4u);
        }
        for (uint32_t row = 0u; row < rows; ++row) {
            v128_t value = wasm_f32x4_splat(input[(size_t)row * inner_dimension + inner]);
            for (uint32_t group = 0u; group < 4u; ++group) {
                accumulators[row][group] = wasm_f32x4_add(
                    accumulators[row][group], wasm_f32x4_mul(value, weights[group]));
            }
        }
    }
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t group = 0u; group < 4u; ++group) {
            wasm_v128_store(values[row] + group * 4u, accumulators[row][group]);
        }
    }
#else
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
            float sum = 0.0f;
            for (uint32_t inner = 0u; inner < inner_dimension; ++inner) {
                sum += input[(size_t)row * inner_dimension + inner] *
                       panel[(size_t)inner * LW_CTC_PANEL + lane];
            }
            values[row][lane] = sum;
        }
    }
#endif
    for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
            values[row][lane] += bias != NULL ? bias[column_base + lane] : 0.0f;
        }
    }
}

void lw_wasm128_packed_matmul_argmax_scores_f32(
    const float* input, const float* packed_weights, const float* bias,
    uint32_t* best_indices, float* scores, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns) {
    if (columns == 0u) return;
    for (uint32_t row = 0u; row < rows; ++row) {
        scores[row] = -INFINITY;
        best_indices[row] = 0u;
    }
    for (uint32_t column_base = 0u; column_base < columns;
         column_base += LW_CTC_PANEL) {
        uint32_t valid_columns = columns - column_base;
        if (valid_columns > LW_CTC_PANEL) valid_columns = LW_CTC_PANEL;
        for (uint32_t row_base = 0u; row_base < rows; row_base += LW_CTC_ROWS) {
            float values[LW_CTC_ROWS][LW_CTC_PANEL];
            uint32_t tile_rows = rows - row_base;
            if (tile_rows > LW_CTC_ROWS) tile_rows = LW_CTC_ROWS;
            project_panel(input + (size_t)row_base * inner_dimension,
                          packed_weights, bias, values, tile_rows,
                          inner_dimension, column_base, valid_columns);
            for (uint32_t row = 0u; row < tile_rows; ++row) {
                for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
                    float value = values[row][lane];
                    if (value > scores[row_base + row]) {
                        scores[row_base + row] = value;
                        best_indices[row_base + row] = column_base + lane;
                    }
                }
            }
        }
    }
}

void lw_wasm128_ctc_row_probabilities_f32(
    const float* input, const float* packed_weights, const float* bias,
    const uint32_t* best_indices, const float* scores, float* probabilities,
    uint32_t rows, uint32_t inner_dimension, uint32_t columns) {
    for (uint32_t row = 0u; row < rows; ++row) {
        uint32_t index = best_indices[row];
        if (index != 0u && (row == 0u || index != best_indices[row - 1u])) {
            float sum = 0.0f;
            for (uint32_t column_base = 0u; column_base < columns;
                 column_base += LW_CTC_PANEL) {
                float values[LW_CTC_ROWS][LW_CTC_PANEL];
                uint32_t valid_columns = columns - column_base;
                if (valid_columns > LW_CTC_PANEL) valid_columns = LW_CTC_PANEL;
                project_panel(input + (size_t)row * inner_dimension,
                              packed_weights, bias, values, 1u,
                              inner_dimension, column_base, valid_columns);
                for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
                    sum += expf(values[0][lane] - scores[row]);
                }
            }
            probabilities[row] = 1.0f / sum;
        }
    }
}
