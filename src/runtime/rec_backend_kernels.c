#include "rec_backend_kernels_internal.h"
#include "../simd/simd_kernels.h"

#if defined(__EMSCRIPTEN__)
void lw_wasm128_nhwc_pointwise_2x16_f32(const float*, const float*,
                                        const lw_nhwc_epilogue*, float*,
                                        uint32_t, uint32_t, uint32_t);
void lw_wasm128_nhwc_pointwise_4x16_f32(const float*, const float*,
                                        const lw_nhwc_epilogue*, float*,
                                        uint32_t, uint32_t, uint32_t);
#ifndef LW_WASM_POINTWISE_ROWS
#define LW_WASM_POINTWISE_ROWS 2
#endif
#if LW_WASM_POINTWISE_ROWS == 2
#define LW_WASM_POINTWISE_KERNEL lw_wasm128_nhwc_pointwise_2x16_f32
#elif LW_WASM_POINTWISE_ROWS == 4
#define LW_WASM_POINTWISE_KERNEL lw_wasm128_nhwc_pointwise_4x16_f32
#else
#error Unsupported LW_WASM_POINTWISE_ROWS
#endif
void lw_wasm128_packed_matmul_argmax_scores_f32(
    const float*, const float*, const float*, uint32_t*, float*,
    uint32_t, uint32_t, uint32_t);
void lw_wasm128_ctc_row_probabilities_f32(
    const float*, const float*, const float*, const uint32_t*, const float*,
    float*, uint32_t, uint32_t, uint32_t);
#else
static void x64_ctc_argmax_scores(const float* input, const float* weights,
                                  const float* bias, uint32_t* indices,
                                  float* scores, uint32_t rows,
                                  uint32_t inner, uint32_t columns) {
    lw_avx2_fma_packed_matmul_argmax_scores_f32(
        input, weights, bias, indices, scores, 1u, rows, inner, columns);
}
#endif

const lw_rec_backend_kernels* lw_rec_backend_kernels_current(void) {
#if defined(__EMSCRIPTEN__)
    static const lw_rec_backend_kernels kernels = {
        "wasm128", 4u, LW_NHWC_OC_BLOCK, 0u,
        LW_WASM_POINTWISE_KERNEL,
        lw_avx2_fma_nhwc_dense_f32,
        lw_avx2_fma_nhwc_depthwise_f32,
        lw_wasm128_packed_matmul_argmax_scores_f32,
        lw_wasm128_ctc_row_probabilities_f32
    };
#else
    static const lw_rec_backend_kernels kernels = {
        "avx2-fma", 8u, LW_NHWC_OC_BLOCK, 1u,
        lw_avx2_fma_nhwc_pointwise_f32,
        lw_avx2_fma_nhwc_dense_f32,
        lw_avx2_fma_nhwc_depthwise_f32,
        x64_ctc_argmax_scores,
        lw_avx2_ctc_row_probabilities_f32
    };
#endif
    return &kernels;
}
