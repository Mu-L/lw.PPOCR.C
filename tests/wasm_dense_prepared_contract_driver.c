/* Address-only specialization: arithmetic and all epilogues must be bitwise
 * identical. Explicit checks survive Release/NDEBUG. */
#include "nhwc_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t seed = 219u;
static float sample(void) {
    seed = seed * 1664525u + 1013904223u;
    return ((float)((seed >> 16u) & 255u) - 128.0f) / 512.0f;
}

static int check(lw_nhwc_dense_desc desc, uint16_t activation, unsigned mask, int alias) {
    size_t inputs = (size_t)desc.batch * desc.input_height * desc.input_width * desc.input_channels;
    size_t outputs = (size_t)desc.batch * desc.output_height * desc.output_width * desc.output_channels;
    size_t k_total = (size_t)desc.input_channels * desc.kernel_h * desc.kernel_w;
    size_t weights_count = k_total * desc.output_channels;
    uint64_t packed_count = 0u, normal_bytes = 0u, prepared_bytes = 0u, scratch_bytes;
    float *input = NULL, *weights = NULL, *packed = NULL, *expected = NULL, *actual = NULL;
    float *bias = NULL, *post_bias = NULL, *residual = NULL;
    int32_t* offsets = NULL;
    void* scratch = NULL;
    lw_nhwc_epilogue ep = {0}, reference;
    int failed = 1;
    if (!lw_nhwc_dense_packed_weight_count(desc.input_channels, desc.output_channels,
            desc.kernel_h, desc.kernel_w, &packed_count) ||
        !lw_nhwc_dense_scratch_bytes(&desc, &normal_bytes) ||
        !lw_nhwc_dense_prepared_scratch_bytes(&desc, &prepared_bytes)) goto done;
    scratch_bytes = normal_bytes > prepared_bytes ? normal_bytes : prepared_bytes;
    input = malloc((inputs + 1u) * sizeof(float));
    weights = malloc(weights_count * sizeof(float));
    packed = malloc((size_t)packed_count * sizeof(float));
    expected = malloc((outputs + 1u) * sizeof(float));
    actual = malloc((outputs + 1u) * sizeof(float));
    residual = malloc((outputs + 1u) * sizeof(float));
    bias = malloc((size_t)desc.output_channels * sizeof(float));
    post_bias = malloc((size_t)desc.output_channels * sizeof(float));
    offsets = malloc(k_total * 2u * sizeof(int32_t));
    scratch = malloc((size_t)scratch_bytes);
    if (!input || !weights || !packed || !expected || !actual || !residual ||
        !bias || !post_bias || !offsets || !scratch) goto done;
    for (size_t i = 0u; i < inputs; ++i) input[i] = sample();
    for (size_t i = 0u; i < weights_count; ++i) weights[i] = sample();
    for (size_t i = 0u; i <= outputs; ++i) expected[i] = actual[i] = residual[i] = sample();
    for (uint32_t i = 0u; i < desc.output_channels; ++i) {
        bias[i] = sample(); post_bias[i] = sample();
    }
    lw_pack_nhwc_dense_f32(weights, desc.input_channels, desc.output_channels,
                           desc.kernel_h, desc.kernel_w, packed);
    {
        uint32_t taps = desc.kernel_h * desc.kernel_w;
        uint32_t patch_width = (LW_NHWC_PIXEL_TILE - 1u) * desc.stride_w + desc.kernel_w;
        for (uint32_t c = 0u; c < desc.input_channels; ++c)
            for (uint32_t y = 0u; y < desc.kernel_h; ++y)
                for (uint32_t x = 0u; x < desc.kernel_w; ++x) {
                    size_t k = (size_t)c * taps + y * desc.kernel_w + x;
                    offsets[k] = (int32_t)(((y * desc.input_width + x) * desc.input_channels) + c);
                    offsets[k_total + k] = (int32_t)(((y * patch_width + x) * desc.input_channels) + c);
                }
    }
    ep.activation = activation;
    ep.bias = (mask & 1u) ? bias : NULL;
    ep.post_bias = (mask & 2u) ? post_bias : NULL;
    ep.residual = (mask & 4u) ? (alias ? actual : residual) : NULL;
    reference = ep;
    if (alias && reference.residual != NULL) reference.residual = expected;
    if (lw_avx2_fma_nhwc_dense_f32(input, packed, mask == 8u ? NULL : &reference,
            expected, &desc, scratch, scratch_bytes) != LW_STATUS_OK ||
        lw_avx2_fma_nhwc_dense_prepared_f32(input, packed, offsets, offsets + k_total,
            mask == 8u ? NULL : &ep, actual, &desc, scratch, scratch_bytes) != LW_STATUS_OK ||
        memcmp(expected, actual, (outputs + 1u) * sizeof(float)) != 0) goto done;
    /* Invalid metadata must reject before any output store; generic can still
     * run with the same max-sized scratch if a caller chooses to retry. */
    if (lw_avx2_fma_nhwc_dense_prepared_f32(input, packed, NULL, offsets + k_total,
            NULL, actual, &desc, scratch, scratch_bytes) != LW_STATUS_INVALID_ARGUMENT ||
        memcmp(expected, actual, (outputs + 1u) * sizeof(float)) != 0) goto done;
    failed = 0;
done:
    if (failed) fprintf(stderr, "Prepared Dense differs IC=%u OC=%u shape=%ux%u stride=%u KC=%u act=%u mask=%u alias=%d\n",
        desc.input_channels, desc.output_channels, desc.input_height, desc.input_width,
        desc.stride_w, desc.dense_kc, (unsigned)activation, mask, alias);
    free(input); free(weights); free(packed); free(expected); free(actual);
    free(residual); free(bias); free(post_bias); free(offsets); free(scratch);
    return failed;
}

int main(void) {
    /* Tiny/Small stem-like shapes plus tail, interior, border and K-blocking. */
    static const uint32_t shapes[][5] = {
        {3u, 16u, 7u, 17u, 1u}, {3u, 17u, 8u, 25u, 2u},
        {17u, 31u, 5u, 13u, 1u}, {80u, 33u, 3u, 7u, 1u},
        {5u, 3u, 1u, 1u, 1u}, {32u, 64u, 9u, 31u, 2u}
    };
    static const uint32_t blocks[] = {0u, 64u, 512u};
    unsigned cases = 0u;
    for (size_t s = 0u; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
        for (size_t b = 0u; b < sizeof(blocks) / sizeof(blocks[0]); ++b)
            for (uint16_t activation = 0u; activation <= LW_NHWC_ACT_GELU; ++activation)
                for (unsigned mask = 0u; mask <= 8u; ++mask)
                    for (int alias = 0; alias <= 1; ++alias) {
                        lw_nhwc_dense_desc d = {0};
                        d.batch = s == 0u ? 2u : 1u;
                        d.input_channels = shapes[s][0]; d.output_channels = shapes[s][1];
                        d.input_height = shapes[s][2]; d.input_width = shapes[s][3];
                        d.stride_h = d.stride_w = shapes[s][4];
                        d.kernel_h = d.kernel_w = 3u;
                        d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = 1u;
                        d.output_height = (d.input_height - 1u) / d.stride_h + 1u;
                        d.output_width = (d.input_width - 1u) / d.stride_w + 1u;
                        d.dense_kc = blocks[b];
                        if (check(d, activation, mask, alias)) return 1;
                        ++cases;
                    }
    printf("WASM_DENSE_PREPARED_CONTRACT cases=%u bitwise=pass\n", cases);
    return 0;
}
