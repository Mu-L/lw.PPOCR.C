#include "simd_kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

/* Preserve the AVX2/scalar DB contract: reject NaN/Inf and set each byte only
 * when probability is strictly greater than the threshold. */
int lw_wasm128_threshold_bitmap_f32(const float* prediction, uint8_t* bitmap,
                                    uint64_t pixel_count, float threshold) {
    const v128_t absolute_mask = wasm_i32x4_splat(0x7fffffff);
    const v128_t infinity_bits = wasm_i32x4_splat(0x7f800000);
    const v128_t limit = wasm_f32x4_splat(threshold);
    uint64_t index = 0u;
    if (prediction == NULL || bitmap == NULL) return -1;
    for (; index + 4u <= pixel_count; index += 4u) {
        v128_t values = wasm_v128_load(prediction + (size_t)index);
        v128_t absolute_bits = wasm_v128_and(values, absolute_mask);
        uint32_t mask;
        if (wasm_i32x4_bitmask(wasm_i32x4_ge(absolute_bits, infinity_bits)) != 0)
            return -1;
        mask = (uint32_t)wasm_i32x4_bitmask(wasm_f32x4_gt(values, limit));
        bitmap[(size_t)index] = (uint8_t)(mask & 1u);
        bitmap[(size_t)index + 1u] = (uint8_t)((mask >> 1u) & 1u);
        bitmap[(size_t)index + 2u] = (uint8_t)((mask >> 2u) & 1u);
        bitmap[(size_t)index + 3u] = (uint8_t)((mask >> 3u) & 1u);
    }
    for (; index < pixel_count; ++index) {
        float value = prediction[(size_t)index];
        if (!isfinite(value)) return -1;
        bitmap[(size_t)index] = value > threshold ? 1u : 0u;
    }
    return 0;
}
