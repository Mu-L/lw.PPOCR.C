/* Exercise the actual SIMD implementation, including its unchanged 4x16
 * reference. Explicit checks remain active under Release/NDEBUG. */
#include "../src/simd/wasm128_nhwc_pointwise.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t seed = 193u;
static float sample(void) {
    seed = seed * 1664525u + 1013904223u;
    return ((float)((seed >> 16u) & 255u) - 128.0f) / 512.0f;
}

static int check(uint32_t pixels, uint32_t ic, uint32_t oc,
                 uint16_t activation, unsigned mask, int alias) {
    size_t inputs = (size_t)pixels * ic;
    size_t outputs = (size_t)pixels * oc;
    size_t packed = (size_t)((oc + 15u) / 16u) * ic * 16u;
    float* input = malloc((inputs + 1u) * sizeof(float));
    float* weights = malloc((packed + 1u) * sizeof(float));
    float* bias = malloc(((size_t)oc + 1u) * sizeof(float));
    float* post_bias = malloc(((size_t)oc + 1u) * sizeof(float));
    float* residual = malloc((outputs + 1u) * sizeof(float));
    float* expected = malloc((outputs + 1u) * sizeof(float));
    float* actual = malloc((outputs + 1u) * sizeof(float));
    lw_nhwc_epilogue epilogue = {0};
    lw_nhwc_epilogue reference;
    int failed = 0;
    if (!input || !weights || !bias || !post_bias || !residual ||
        !expected || !actual) {
        failed = 1;
        goto done;
    }
    for (size_t i = 0u; i < inputs; ++i) input[i] = sample();
    for (size_t i = 0u; i < packed; ++i) weights[i] = sample();
    for (size_t i = 0u; i < oc; ++i) {
        bias[i] = sample();
        post_bias[i] = sample();
    }
    for (size_t i = 0u; i <= outputs; ++i)
        actual[i] = expected[i] = residual[i] = sample();
    epilogue.activation = activation;
    epilogue.bias = (mask & 1u) ? bias : NULL;
    epilogue.post_bias = (mask & 2u) ? post_bias : NULL;
    epilogue.residual = (mask & 4u) ? (alias ? actual : residual) : NULL;
    reference = epilogue;
    if (alias && reference.residual != NULL) reference.residual = expected;
    lw_wasm128_nhwc_pointwise_4x16_f32(input, weights,
        mask == 8u ? NULL : &reference, expected, pixels, ic, oc);
    lw_wasm128_nhwc_pointwise_2x16_f32(input, weights,
        mask == 8u ? NULL : &epilogue, actual, pixels, ic, oc);
    /* Include the end sentinel to detect overwriting the OC tail. */
    if (memcmp(expected, actual, (outputs + 1u) * sizeof(float)) != 0) {
        fprintf(stderr, "Pointwise mismatch pixels=%u IC=%u OC=%u act=%u mask=%u alias=%d\n",
                pixels, ic, oc, (unsigned)activation, mask, alias);
        failed = 1;
    }
done:
    free(input); free(weights); free(bias); free(post_bias);
    free(residual); free(expected); free(actual);
    return failed;
}

int main(void) {
    static const uint32_t shapes[][3] = {
        {0u, 0u, 0u}, {0u, 3u, 17u}, {3u, 5u, 0u},
        {1u, 0u, 3u}, {1u, 3u, 1u}, {2u, 5u, 17u},
        {3u, 17u, 31u}, {7u, 31u, 33u}, {8u, 64u, 64u},
        {9u, 128u, 128u}, {2u, 512u, 512u}
    };
    unsigned cases = 0u;
    for (size_t s = 0u; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
        for (uint16_t activation = 0u; activation <= LW_NHWC_ACT_GELU; ++activation)
            for (unsigned mask = 0u; mask <= 8u; ++mask)
                for (int alias = 0; alias <= 1; ++alias) {
                    if (check(shapes[s][0], shapes[s][1], shapes[s][2],
                              activation, mask, alias)) return 1;
                    ++cases;
                }
    printf("WASM Pointwise bitwise contract: %u cases passed\n", cases);
    return 0;
}
