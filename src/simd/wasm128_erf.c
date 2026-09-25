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
static v128_t erf_small_magnitude_f32(v128_t absolute, v128_t squared) {
    v128_t polynomial = wasm_f32x4_splat(1.0590875083315439e-6f);
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-1.3906452410711274e-5f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.1955437252428243e-4f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-8.5424759600797662e-4f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(5.2237718994271529e-3f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-2.6866128888288668e-2f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.1283791226625459e-1f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-3.7612638883043881e-1f),
                                wasm_f32x4_mul(squared, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.1283791670929921f),
                                wasm_f32x4_mul(squared, polynomial));
    return wasm_f32x4_mul(absolute, polynomial);
}

static v128_t erf_middle_magnitude_f32(v128_t absolute) {
    v128_t shifted = wasm_f32x4_sub(absolute, wasm_f32x4_splat(1.5f));
    v128_t polynomial = wasm_f32x4_splat(-2.4006675278365739e-3f);
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-3.8855162788028288e-3f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.9332860298601401e-2f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-1.5013607234871428e-2f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-4.4599369472787913e-2f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.3876136033191724e-1f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-1.7839541988688759e-1f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.1893013063163335e-1f),
                                wasm_f32x4_mul(shifted, polynomial));
    return wasm_f32x4_add(wasm_f32x4_splat(9.6610514641406819e-1f),
                           wasm_f32x4_mul(shifted, polynomial));
}

static v128_t erf_large_magnitude_f32(v128_t absolute) {
    v128_t shifted = wasm_f32x4_sub(absolute, wasm_f32x4_splat(3.0f));
    v128_t polynomial = wasm_f32x4_splat(-8.8750765320565021e-5f);
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(3.8805285630072217e-4f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-7.7812010711428433e-4f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.0255980996254569e-3f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-1.0307062838910627e-3f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(7.8586080129729565e-4f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(-4.1936053855221229e-4f),
                                wasm_f32x4_mul(shifted, polynomial));
    polynomial = wasm_f32x4_add(wasm_f32x4_splat(1.3951109720745927e-4f),
                                wasm_f32x4_mul(shifted, polynomial));
    return wasm_f32x4_add(wasm_f32x4_splat(9.9997793886838715e-1f),
                           wasm_f32x4_mul(shifted, polynomial));
}

static v128_t erf_approximation_f32(v128_t value) {
    const v128_t absolute_mask = wasm_i32x4_splat(INT32_C(0x7fffffff));
    const v128_t sign_mask = wasm_i32x4_splat(INT32_MIN);
    const v128_t one = wasm_f32x4_splat(1.0f);
    v128_t absolute = wasm_v128_and(value, absolute_mask);
    v128_t squared = wasm_f32x4_mul(absolute, absolute);
    v128_t result = wasm_v128_bitselect(erf_middle_magnitude_f32(absolute),
        erf_large_magnitude_f32(absolute),
        wasm_f32x4_lt(absolute, wasm_f32x4_splat(2.0f)));
    result = wasm_v128_bitselect(erf_small_magnitude_f32(absolute, squared),
        result, wasm_f32x4_lt(absolute, one));
    result = wasm_v128_bitselect(one, result,
        wasm_f32x4_ge(absolute, wasm_f32x4_splat(4.0f)));
    result = wasm_v128_xor(result, wasm_v128_and(value, sign_mask));
    return wasm_v128_bitselect(value, result, wasm_f32x4_ne(value, value));
}
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
    const v128_t square_root_two = wasm_f32x4_splat(1.4142135381698608f);
    const v128_t one = wasm_f32x4_splat(1.0f);
    const v128_t half = wasm_f32x4_splat(0.5f);
    uint64_t index = 0u;
    for (; index + 4u <= element_count; index += 4u) {
        v128_t value = wasm_v128_load(input + (size_t)index);
        v128_t scaled = wasm_f32x4_div(value, square_root_two);
        v128_t activated = wasm_f32x4_add(erf_approximation_f32(scaled), one);
        activated = wasm_f32x4_mul(value, activated);
        wasm_v128_store(output + (size_t)index, wasm_f32x4_mul(activated, half));
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
