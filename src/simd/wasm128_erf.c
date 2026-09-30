#include "simd_kernels.h"

/* WASM SIMD128 Erf approximation shared by browser, Node, and compiled REC. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#  define LW_COMPILES_WASM128_ERF 1
#else
#  define LW_COMPILES_WASM128_ERF 0
#endif

#if LW_COMPILES_WASM128_ERF
#include "wasm128_erf_internal.h"
#endif

void lw_wasm128_erf_f32(const float* input, float* output, uint64_t element_count) {
#if LW_COMPILES_WASM128_ERF
    uint64_t index = 0u;

    for (; index + 4u <= element_count; index += 4u) {
        wasm_v128_store(output + (size_t)index,
            erf_approximation_f32(wasm_v128_load(input + (size_t)index)));
    }
    for (; index < element_count; ++index) {
        output[(size_t)index] = erff(input[(size_t)index]);
    }
#else
    uint64_t index;
    for (index = 0u; index < element_count; ++index) {
        output[(size_t)index] = erff(input[(size_t)index]);
    }
#endif
}

/* Reuse the canonical WASM Erf polynomial in fused REC GELU epilogues.
 * Input and output may alias. */
void lw_wasm128_gelu_f32(const float* input, float* output, uint64_t element_count) {
#if LW_COMPILES_WASM128_ERF
    uint64_t index = 0u;
    for (; index + 4u <= element_count; index += 4u) {
        v128_t value = wasm_v128_load(input + (size_t)index);
        wasm_v128_store(output + (size_t)index, lw_wasm128_gelu_vector_f32(value));
    }
    for (; index < element_count; ++index) {
        float value = input[(size_t)index];
        float activated = erff(value / 1.4142135381698608f) + 1.0f;
        activated = value * activated;
        output[(size_t)index] = activated * 0.5f;
    }
#else
    uint64_t index;
    for (index = 0u; index < element_count; ++index) {
        float value = input[(size_t)index];
        float activated = erff(value / 1.4142135381698608f) + 1.0f;
        activated = value * activated;
        output[(size_t)index] = activated * 0.5f;
    }
#endif
}
