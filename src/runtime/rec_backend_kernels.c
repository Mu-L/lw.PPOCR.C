#include "rec_backend_kernels_internal.h"

#if defined(__EMSCRIPTEN__)
void lw_wasm128_nhwc_pointwise_4x16_f32(const float*, const float*,
                                        const lw_nhwc_epilogue*, float*,
                                        uint32_t, uint32_t, uint32_t);
#endif

const lw_rec_backend_kernels* lw_rec_backend_kernels_current(void) {
#if defined(__EMSCRIPTEN__)
    static const lw_rec_backend_kernels kernels = {
        "wasm128", 4u, LW_NHWC_OC_BLOCK, 0u,
        lw_wasm128_nhwc_pointwise_4x16_f32,
        lw_avx2_fma_nhwc_dense_f32,
        lw_avx2_fma_nhwc_depthwise_f32
    };
#else
    static const lw_rec_backend_kernels kernels = {
        "avx2-fma", 8u, LW_NHWC_OC_BLOCK, 1u,
        lw_avx2_fma_nhwc_pointwise_f32,
        lw_avx2_fma_nhwc_dense_f32,
        lw_avx2_fma_nhwc_depthwise_f32
    };
#endif
    return &kernels;
}
