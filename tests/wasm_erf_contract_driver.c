/* Include the implementation to compare its private vector primitive with
 * the pre-optimization, unconditional three-polynomial reference. */
#include "../src/simd/wasm128_erf.c"
#include <stdio.h>
#include <string.h>

static v128_t reference_erf(v128_t value) {
    const v128_t one = wasm_f32x4_splat(1.0f);
    const v128_t sign = wasm_i32x4_splat(INT32_MIN);
    v128_t absolute = wasm_v128_and(value, wasm_i32x4_splat(INT32_C(0x7fffffff)));
    v128_t squared = wasm_f32x4_mul(absolute, absolute);
    v128_t result = wasm_v128_bitselect(erf_middle_magnitude_f32(absolute),
        erf_large_magnitude_f32(absolute),
        wasm_f32x4_lt(absolute, wasm_f32x4_splat(2.0f)));
    result = wasm_v128_bitselect(erf_small_magnitude_f32(absolute, squared),
        result, wasm_f32x4_lt(absolute, one));
    result = wasm_v128_bitselect(one, result,
        wasm_f32x4_ge(absolute, wasm_f32x4_splat(4.0f)));
    result = wasm_v128_xor(result, wasm_v128_and(value, sign));
    return wasm_v128_bitselect(value, result, wasm_f32x4_ne(value, value));
}

static int compare_block(const float* input) {
    float expected[4], actual[4], alias[4], direct[4];
    v128_t value = wasm_v128_load(input);
    wasm_v128_store(expected, reference_erf(value));
    lw_wasm128_erf_f32(input, actual, 4u);
    memcpy(alias, input, sizeof(alias));
    lw_wasm128_erf_f32(alias, alias, 4u);
    if (memcmp(expected, actual, sizeof(actual)) != 0 ||
        memcmp(expected, alias, sizeof(alias)) != 0) return 1;
    v128_t scaled = wasm_f32x4_div(value, wasm_f32x4_splat(1.4142135381698608f));
    v128_t activated = wasm_f32x4_add(reference_erf(scaled), wasm_f32x4_splat(1.0f));
    activated = wasm_f32x4_mul(value, activated);
    wasm_v128_store(expected, wasm_f32x4_mul(activated, wasm_f32x4_splat(0.5f)));
    wasm_v128_store(direct, lw_wasm128_gelu_vector_f32(value));
    lw_wasm128_gelu_f32(input, actual, 4u);
    memcpy(alias, input, sizeof(alias));
    lw_wasm128_gelu_f32(alias, alias, 4u);
    for (uint32_t lane = 0u; lane < 4u; ++lane) {
        /* Arithmetic NaN payloads are not specified by WebAssembly. */
        if (isnan(expected[lane]) && isnan(actual[lane]) &&
            isnan(alias[lane]) && isnan(direct[lane])) continue;
        if (memcmp(expected + lane, actual + lane, sizeof(float)) != 0 ||
            memcmp(expected + lane, alias + lane, sizeof(float)) != 0 ||
            memcmp(expected + lane, direct + lane, sizeof(float)) != 0) return 1;
    }
    return 0;
}

int main(void) {
    static const uint32_t edges[] = {
        0u, UINT32_C(0x80000000), 1u, UINT32_C(0x80000001),
        UINT32_C(0x3f7fffff), UINT32_C(0x3f800000), UINT32_C(0x3f800001),
        UINT32_C(0x3fffffff), UINT32_C(0x40000000), UINT32_C(0x40000001),
        UINT32_C(0x407fffff), UINT32_C(0x40800000), UINT32_C(0x40800001),
        UINT32_C(0x7f800000), UINT32_C(0xff800000), UINT32_C(0x7fc12345)
    };
    uint32_t state = UINT32_C(0x12345678);
    float block[4];
    for (uint32_t index = 0u; index < sizeof(edges) / sizeof(edges[0]); ++index) {
        for (uint32_t sign = 0u; sign < 2u; ++sign) {
            uint32_t bits = edges[index] ^ (sign != 0u ? UINT32_C(0x80000000) : 0u);
            for (uint32_t lane = 0u; lane < 4u; ++lane)
                memcpy(block + lane, &bits, sizeof(bits));
            if (compare_block(block)) return 1;
        }
    }
    for (uint32_t index = 0u; index < 65536u; ++index) {
        for (uint32_t lane = 0u; lane < 4u; ++lane) {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            if (index < 256u) {
                uint32_t bits = edges[(index + lane) % (sizeof(edges) / sizeof(edges[0]))];
                memcpy(block + lane, &bits, sizeof(bits));
            } else if ((index & 1u) == 0u) {
                block[lane] = (float)(int32_t)(state % 20001u) / 10001.0f - 1.0f;
            } else {
                memcpy(block + lane, &state, sizeof(state));
            }
        }
        if (compare_block(block)) {
            fprintf(stderr, "Erf/GELU bitwise mismatch at block %u\n", index);
            return 1;
        }
    }
    for (uint32_t count = 0u; count <= 19u; ++count) {
        float input[20], actual[20], alias[20];
        for (uint32_t i = 0u; i < 20u; ++i) input[i] = (float)i * 0.25f - 2.0f;
        memcpy(alias, input, sizeof(input));
        lw_wasm128_gelu_f32(input, actual, count);
        lw_wasm128_gelu_f32(alias, alias, count);
        if (memcmp(actual, alias, count * sizeof(float)) != 0) return 1;
        if (memcmp(alias + count, input + count, (20u - count) * sizeof(float)) != 0)
            return 1;
        for (uint32_t i = count - count % 4u; i < count; ++i) {
            float expected = input[i] * (erff(input[i] / 1.4142135381698608f) + 1.0f);
            expected *= 0.5f;
            if (memcmp(actual + i, &expected, sizeof(float)) != 0) return 1;
        }
    }
    puts("WASM Erf/GELU contract: 65536 vector blocks and alias/tail cases passed");
    return 0;
}
