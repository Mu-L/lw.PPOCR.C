#ifndef LW_WASM128_EPILOGUE_INTERNAL_H
#define LW_WASM128_EPILOGUE_INTERNAL_H

#include "nhwc_internal.h"
#include "simd_kernels.h"

#if defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)
#include <wasm_simd128.h>

/* Full OC16 blocks only: callers handle the channel tail without reading
 * past post_bias or residual. Bias is already in the accumulators. */
static inline void lw_wasm128_apply_epilogue_oc16(
    v128_t values[4], const lw_nhwc_epilogue* epilogue,
    uint32_t channel_base, const float* residual, int hardswish_divide) {
    uint32_t group;
    if (epilogue == NULL) return;
    if (epilogue->post_bias != NULL) {
        for (group = 0u; group < 4u; ++group) {
            values[group] = wasm_f32x4_add(values[group],
                wasm_v128_load(epilogue->post_bias + channel_base + group * 4u));
        }
    }
    if (residual != NULL) {
        for (group = 0u; group < 4u; ++group) {
            values[group] = wasm_f32x4_add(values[group],
                wasm_v128_load(residual + group * 4u));
        }
    }
    if (epilogue->activation == LW_NHWC_ACT_RELU) {
        const v128_t zero = wasm_f32x4_splat(0.0f);
        for (group = 0u; group < 4u; ++group) {
            values[group] = wasm_v128_bitselect(values[group], zero,
                wasm_f32x4_gt(values[group], zero));
        }
    } else if (epilogue->activation == LW_NHWC_ACT_HARDSWISH) {
        const v128_t zero = wasm_f32x4_splat(0.0f);
        const v128_t three = wasm_f32x4_splat(3.0f);
        const v128_t six = wasm_f32x4_splat(6.0f);
        const v128_t inverse_six = wasm_f32x4_splat(1.0f / 6.0f);
        for (group = 0u; group < 4u; ++group) {
            v128_t gate = wasm_f32x4_add(values[group], three);
            gate = wasm_v128_bitselect(zero, gate, wasm_f32x4_lt(gate, zero));
            gate = wasm_v128_bitselect(six, gate, wasm_f32x4_gt(gate, six));
            gate = wasm_f32x4_mul(values[group], gate);
            values[group] = hardswish_divide != 0
                ? wasm_f32x4_div(gate, six)
                : wasm_f32x4_mul(gate, inverse_six);
        }
    } else if (epilogue->activation == LW_NHWC_ACT_GELU) {
        float block[LW_NHWC_OC_BLOCK];
        for (group = 0u; group < 4u; ++group) {
            wasm_v128_store(block + group * 4u, values[group]);
        }
        lw_wasm128_gelu_f32(block, block, LW_NHWC_OC_BLOCK);
        for (group = 0u; group < 4u; ++group) {
            values[group] = wasm_v128_load(block + group * 4u);
        }
    }
}
#endif

#endif
