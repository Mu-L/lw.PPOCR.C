#include "simd_kernels.h"
#include "../kernels/packed_matmul_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

/* Shared packed MatMul [panel][inner][OC16], four rows at a time. Keep the
 * inner loop order unchanged so every output lane has the same accumulation
 * sequence as the scalar physical backend. */
void lw_wasm128_packed_matmul_shared_f32(const float* input,
    const float* packed_weights, float* output, uint32_t batch_count,
    uint32_t rows, uint32_t inner_dimension, uint32_t columns) {
    for (uint32_t batch = 0u; batch < batch_count; ++batch) {
        for (uint32_t column_base = 0u; column_base < columns;
             column_base += LW_PACKED_MATMUL_COLUMN_TILE) {
            const float* panel = packed_weights +
                (size_t)(column_base / LW_PACKED_MATMUL_COLUMN_TILE) *
                inner_dimension * LW_PACKED_MATMUL_COLUMN_TILE;
            uint32_t valid_columns = columns - column_base;
            if (valid_columns > LW_PACKED_MATMUL_COLUMN_TILE)
                valid_columns = LW_PACKED_MATMUL_COLUMN_TILE;
            for (uint32_t row_base = 0u; row_base < rows; row_base += 4u) {
                v128_t sums[4][4];
                uint32_t row_count = rows - row_base;
                if (row_count > 4u) row_count = 4u;
                for (uint32_t row = 0u; row < row_count; ++row)
                    for (uint32_t group = 0u; group < 4u; ++group)
                        sums[row][group] = wasm_f32x4_splat(0.0f);
                for (uint32_t inner = 0u; inner < inner_dimension; ++inner) {
                    const float* packed = panel +
                        (size_t)inner * LW_PACKED_MATMUL_COLUMN_TILE;
                    v128_t weights[4];
                    for (uint32_t group = 0u; group < 4u; ++group)
                        weights[group] = wasm_v128_load(packed + group * 4u);
                    for (uint32_t row = 0u; row < row_count; ++row) {
                        v128_t value = wasm_f32x4_splat(input[
                            ((size_t)batch * rows + row_base + row) * inner_dimension + inner]);
                        for (uint32_t group = 0u; group < 4u; ++group)
                            sums[row][group] = wasm_f32x4_add(sums[row][group],
                                wasm_f32x4_mul(value, weights[group]));
                    }
                }
                for (uint32_t row = 0u; row < row_count; ++row) {
                    float* destination = output +
                        ((size_t)batch * rows + row_base + row) * columns + column_base;
                    if (valid_columns == LW_PACKED_MATMUL_COLUMN_TILE) {
                        for (uint32_t group = 0u; group < 4u; ++group)
                            wasm_v128_store(destination + group * 4u, sums[row][group]);
                    } else {
                        float tail[LW_PACKED_MATMUL_COLUMN_TILE];
                        for (uint32_t group = 0u; group < 4u; ++group)
                            wasm_v128_store(tail + group * 4u, sums[row][group]);
                        for (uint32_t lane = 0u; lane < valid_columns; ++lane)
                            destination[lane] = tail[lane];
                    }
                }
            }
        }
    }
}
