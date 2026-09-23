/* Kernel-level contract for the NHWC ConvTranspose 2x2/s2 variants and the
 * NHWC integer-scale nearest resize, against the scalar references. */

#include "cpu_features.h"
#include "nhwc_internal.h"
#include "x64_det_backend_internal.h"
#include "../src/kernels/scalar_kernels.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_values(float* values, uint64_t count, uint32_t seed) {
    uint32_t state = seed;
    for (uint64_t index = 0u; index < count; ++index) {
        state = state * 1664525u + 1013904223u;
        values[(size_t)index] = (float)((int32_t)(state >> 8u) % 257) / 4096.0f;
    }
}

static float max_abs(const float* left, const float* right, uint64_t count) {
    float result = 0.0f;
    for (uint64_t i = 0u; i < count; ++i) {
        float difference = fabsf(left[(size_t)i] - right[(size_t)i]);
        if (!isfinite(difference) || difference > result) result = difference;
    }
    return result;
}

/* The scalar reference consumes NCHW input; the NHWC kernels consume the same
 * logical tensor in NHWC order. */
static float* to_nchw(const float* nhwc, uint32_t batch, uint32_t channels,
                      uint32_t height, uint32_t width) {
    float* nchw = (float*)malloc((size_t)batch * height * width * channels * sizeof(float));
    uint32_t n;
    if (nchw == NULL) return NULL;
    for (n = 0u; n < batch; ++n) {
        uint32_t y;
        for (y = 0u; y < height; ++y) {
            uint32_t x;
            for (x = 0u; x < width; ++x) {
                uint32_t c;
                for (c = 0u; c < channels; ++c) {
                    nchw[((size_t)n * channels + c) * height * width +
                         (size_t)y * width + x] =
                        nhwc[(((size_t)n * height + y) * width + x) * channels + c];
                }
            }
        }
    }
    return nchw;
}

static float* to_nhwc(const float* nchw, uint32_t batch, uint32_t channels,
                      uint32_t height, uint32_t width) {
    float* nhwc = (float*)malloc((size_t)batch * height * width * channels * sizeof(float));
    uint32_t n;
    if (nhwc == NULL) return NULL;
    for (n = 0u; n < batch; ++n) {
        uint32_t y;
        for (y = 0u; y < height; ++y) {
            uint32_t x;
            for (x = 0u; x < width; ++x) {
                uint32_t c;
                for (c = 0u; c < channels; ++c) {
                    nhwc[(((size_t)n * height + y) * width + x) * channels + c] =
                        nchw[((size_t)n * channels + c) * height * width +
                             (size_t)y * width + x];
                }
            }
        }
    }
    return nhwc;
}

static int test_convtranspose_16(uint32_t input_channels, uint32_t output_channels,
                                 uint32_t input_height, uint32_t input_width,
                                 uint32_t batch, uint16_t activation, int with_bias) {
    uint32_t output_height = input_height * 2u;
    uint32_t output_width = input_width * 2u;
    uint64_t input_count = (uint64_t)batch * input_height * input_width * input_channels;
    uint64_t output_count = (uint64_t)batch * output_height * output_width * output_channels;
    uint64_t weight_count = (uint64_t)input_channels * output_channels * 4u;
    uint64_t packed_count = 0u;
    int32_t in_dims[4] = { (int32_t)batch, (int32_t)input_channels,
                           (int32_t)input_height, (int32_t)input_width };
    int32_t out_dims[4] = { (int32_t)batch, (int32_t)output_channels,
                            (int32_t)output_height, (int32_t)output_width };
    float* input = (float*)malloc((size_t)input_count * sizeof(float));
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    float* bias = with_bias ? (float*)malloc((size_t)output_channels * sizeof(float)) : NULL;
    float* expected = (float*)malloc((size_t)output_count * sizeof(float));
    float* actual = (float*)malloc((size_t)output_count * sizeof(float));
    float* packed = NULL;
    lw_nhwc_convtranspose_desc desc;
    lw_nhwc_epilogue epilogue;
    float difference;
    int ok = 0;
    if (input == NULL || weights == NULL || expected == NULL || actual == NULL ||
        (with_bias && bias == NULL)) {
        goto cleanup;
    }
    if (!lw_nhwc_convtranspose_packed_weight_count(input_channels, output_channels,
                                                   &packed_count)) {
        fprintf(stderr, "packed weight count failed for %ux%u\n", input_channels,
                output_channels);
        goto cleanup;
    }
    packed = (float*)malloc((size_t)packed_count * sizeof(float));
    if (packed == NULL) goto cleanup;
    fill_values(input, input_count, 0x11u);
    fill_values(weights, weight_count, 0x37u);
    if (bias != NULL) fill_values(bias, output_channels, 0x91u);
    lw_pack_nhwc_convtranspose2x2_f32(weights, input_channels, output_channels, packed);
    {
        /* Verify the packed layout against the ONNX [ic][oc][4] source. */
        uint32_t tap;
        uint32_t blocks = (output_channels + 15u) / 16u;
        for (tap = 0u; tap < 4u; ++tap) {
            uint32_t block;
            for (block = 0u; block < blocks; ++block) {
                uint32_t ic_i;
                for (ic_i = 0u; ic_i < input_channels; ++ic_i) {
                    uint32_t lane;
                    for (lane = 0u; lane < 16u; ++lane) {
                        float expected_weight = 0.0f;
                        if (block * 16u + lane < output_channels) {
                            expected_weight =
                                weights[((size_t)ic_i * output_channels + block * 16u + lane) *
                                        4u + tap];
                        }
                        float packed_weight =
                            packed[(((size_t)tap * blocks + block) * input_channels + ic_i) *
                                       16u + lane];
                        if (packed_weight != expected_weight) {
                            fprintf(stderr, "pack mismatch tap=%u ic=%u lane=%u: %.9g != %.9g\n",
                                    tap, ic_i, lane, packed_weight, expected_weight);
                            goto cleanup;
                        }
                    }
                }
            }
        }
    }
    {
        float* nhwc_input = to_nhwc(input, batch, input_channels, input_height, input_width);
        if (nhwc_input == NULL) goto cleanup;
        lw_scalar_conv_transpose2x2_stride2_f32(input, weights, bias, expected, in_dims,
                                                out_dims);
        memset(actual, 0, (size_t)output_count * sizeof(float));
        epilogue.bias = bias;
        epilogue.residual = NULL;
        epilogue.activation = activation;
        epilogue.alpha = 0.0f;
        epilogue.beta = 0.0f;
        desc.batch = batch;
        desc.input_channels = input_channels;
        desc.input_height = input_height;
        desc.input_width = input_width;
        desc.output_channels = output_channels;
        desc.output_height = output_height;
        desc.output_width = output_width;
        if (lw_avx2_fma_nhwc_convtranspose2x2_s2_f32(nhwc_input, packed, &epilogue, actual,
                                                    &desc) != LW_STATUS_OK) {
            free(nhwc_input);
            fprintf(stderr, "convtranspose16 kernel failed\n");
            goto cleanup;
        }
        free(nhwc_input);
        /* The kernel writes NHWC; normalize to NCHW for the comparison. */
        {
            float* nchw_actual = to_nchw(actual, batch, output_channels, output_height,
                                         output_width);
            if (nchw_actual == NULL) goto cleanup;
            memcpy(actual, nchw_actual, (size_t)output_count * sizeof(float));
            free(nchw_actual);
        }
    }
    /* Apply the same activation to the reference for comparison. */
    if (activation == LW_NHWC_ACT_RELU) {
        uint64_t i;
        for (i = 0u; i < output_count; ++i) {
            if (expected[i] < 0.0f) expected[i] = 0.0f;
        }
    }
    difference = max_abs(expected, actual, output_count);
    if (difference > 2.0e-4f) {
        uint64_t i;
        uint64_t printed = 0u;
        for (i = 0u; i < output_count && printed < 4u; ++i) {
            if (fabsf(expected[i] - actual[i]) > 2.0e-4f) {
                fprintf(stderr, "mismatch %llu (y=%llu x=%llu c=%llu): ref %.9g got %.9g\n",
                        (unsigned long long)i,
                        (unsigned long long)((i / output_channels) / output_width),
                        (unsigned long long)((i / output_channels) % output_width),
                        (unsigned long long)(i % output_channels),
                        expected[i], actual[i]);
                ++printed;
            }
        }
        /* Hand-computed dots for the first mismatch element, both weight
         * orders and all taps. */
        {
            uint64_t element = 1u;
            uint32_t oc = (uint32_t)(element % output_channels);
            uint32_t tap;
            for (tap = 0u; tap < 4u; ++tap) {
                float dot_ic_major = 0.0f;
                float dot_oc_major = 0.0f;
                uint32_t ic_i;
                for (ic_i = 0u; ic_i < input_channels; ++ic_i) {
                    dot_ic_major += input[(size_t)ic_i * input_height * input_width] *
                                    weights[((size_t)ic_i * output_channels + oc) * 4u + tap];
                    dot_oc_major += input[(size_t)ic_i * input_height * input_width] *
                                    weights[((size_t)oc * input_channels + ic_i) * 4u + tap];
                }
                fprintf(stderr, "hand dot c=%u tap=%u: ic-major %.9g oc-major %.9g\n",
                        oc, tap, dot_ic_major, dot_oc_major);
            }
        }
    }
    printf("{\"case\":\"ct16-ic%u-oc%u-%ux%u-b%u%s%s\",\"max_abs\":%.9g}\n",
           input_channels, output_channels, input_height, input_width, batch,
           activation == LW_NHWC_ACT_RELU ? "-relu" : "",
           with_bias ? "-bias" : "", difference);
    ok = difference <= 2.0e-4f;
cleanup:
    free(packed);
    free(actual);
    free(expected);
    free(bias);
    free(weights);
    free(input);
    return ok;
}

static int test_convtranspose_c1(uint32_t input_channels, uint32_t input_height,
                                 uint32_t input_width, uint32_t batch, int with_relu) {
    uint32_t output_height = input_height * 2u;
    uint32_t output_width = input_width * 2u;
    uint64_t input_count = (uint64_t)batch * input_height * input_width * input_channels;
    uint64_t output_count = (uint64_t)batch * output_height * output_width;
    uint64_t weight_count = (uint64_t)input_channels * 4u;
    int32_t in_dims[4] = { (int32_t)batch, (int32_t)input_channels,
                           (int32_t)input_height, (int32_t)input_width };
    int32_t out_dims[4] = { (int32_t)batch, 1, (int32_t)output_height, (int32_t)output_width };
    float* input = (float*)malloc((size_t)input_count * sizeof(float));
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    float* expected = (float*)malloc((size_t)output_count * sizeof(float));
    float* actual = (float*)malloc((size_t)output_count * sizeof(float));
    lw_nhwc_convtranspose_desc desc;
    lw_nhwc_epilogue epilogue;
    float difference;
    int ok = 0;
    if (input == NULL || weights == NULL || expected == NULL || actual == NULL) goto cleanup;
    fill_values(input, input_count, 0x55u);
    fill_values(weights, weight_count, 0x73u);
    {
        float* nhwc_input = to_nhwc(input, batch, input_channels, input_height, input_width);
        if (nhwc_input == NULL) goto cleanup;
        lw_scalar_conv_transpose2x2_stride2_f32(input, weights, NULL, expected, in_dims,
                                                out_dims);
        desc.batch = batch;
        desc.input_channels = input_channels;
        desc.input_height = input_height;
        desc.input_width = input_width;
        desc.output_channels = 1u;
        desc.output_height = output_height;
        desc.output_width = output_width;
        epilogue.bias = NULL;
        epilogue.residual = NULL;
        epilogue.activation = with_relu ? LW_NHWC_ACT_RELU : LW_NHWC_ACT_NONE;
        epilogue.alpha = 0.0f;
        epilogue.beta = 0.0f;
        if (lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32(nhwc_input, weights, &epilogue, actual,
                                                        &desc) != LW_STATUS_OK) {
            free(nhwc_input);
            fprintf(stderr, "convtranspose c1 kernel failed\n");
            goto cleanup;
        }
        free(nhwc_input);
    }
    if (with_relu) {
        uint64_t i;
        for (i = 0u; i < output_count; ++i) {
            if (expected[i] < 0.0f) expected[i] = 0.0f;
        }
    }
    difference = max_abs(expected, actual, output_count);
    printf("{\"case\":\"ctc1-ic%u-%ux%u-b%u%s\",\"max_abs\":%.9g}\n",
           input_channels, input_height, input_width, batch, with_relu ? "-relu" : "",
           difference);
    ok = difference <= 2.0e-4f;
cleanup:
    free(actual);
    free(expected);
    free(weights);
    free(input);
    return ok;
}

static int test_resize(uint32_t channels, uint32_t input_height, uint32_t input_width,
                       uint32_t scale_h, uint32_t scale_w, uint32_t batch) {
    uint32_t output_height = input_height * scale_h;
    uint32_t output_width = input_width * scale_w;
    uint64_t input_count = (uint64_t)batch * input_height * input_width * channels;
    uint64_t output_count = (uint64_t)batch * output_height * output_width * channels;
    float* input = (float*)malloc((size_t)input_count * sizeof(float));
    float* expected = (float*)malloc((size_t)output_count * sizeof(float));
    float* actual = (float*)malloc((size_t)output_count * sizeof(float));
    uint32_t n;
    int ok = 0;
    if (input == NULL || expected == NULL || actual == NULL) goto cleanup;
    fill_values(input, input_count, 0x99u);
    for (n = 0u; n < batch; ++n) {
        uint32_t oy;
        for (oy = 0u; oy < output_height; ++oy) {
            uint32_t source_y = oy / scale_h;
            uint32_t ox;
            for (ox = 0u; ox < output_width; ++ox) {
                uint32_t source_x = ox / scale_w;
                const float* source_pixel = input +
                    (((size_t)n * input_height + source_y) * input_width + source_x) * channels;
                float* output_pixel = expected +
                    (((size_t)n * output_height + oy) * output_width + ox) * channels;
                memcpy(output_pixel, source_pixel, (size_t)channels * sizeof(float));
            }
        }
    }
    lw_avx2_nhwc_resize_nearest_f32(input, actual, batch, channels, input_height, input_width,
                                    output_height, output_width);
    /* Pure data movement: expect bit-identical output. */
    ok = memcmp(expected, actual, (size_t)output_count * sizeof(float)) == 0;
    printf("{\"case\":\"resize-c%u-%ux%u-s%ux%u-b%u\",\"bit_exact\":%d}\n",
           channels, input_height, input_width, scale_h, scale_w, batch, ok);
cleanup:
    free(actual);
    free(expected);
    free(input);
    return ok;
}

static int test_depthwise_rows(uint32_t channels, uint32_t input_height, uint32_t input_width,
                               uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h,
                               uint32_t stride_w, uint32_t pad_top, uint32_t pad_left) {
    uint32_t output_height = (input_height + pad_top * 2u - kernel_h) / stride_h + 1u;
    uint32_t output_width = (input_width + pad_left * 2u - kernel_w) / stride_w + 1u;
    uint64_t input_count = (uint64_t)input_height * input_width * channels;
    uint64_t output_count = (uint64_t)output_height * output_width * channels;
    uint64_t weight_count = (uint64_t)channels * kernel_h * kernel_w;
    uint64_t packed_count = 0u;
    float* input = (float*)malloc((size_t)input_count * sizeof(float));
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    float* packed = NULL;
    float* full = (float*)malloc((size_t)output_count * sizeof(float));
    float* sharded = (float*)malloc((size_t)output_count * sizeof(float));
    uint32_t split = output_height / 2u;
    int ok = 0;
    if (input == NULL || weights == NULL || full == NULL || sharded == NULL) goto cleanup;
    if (!lw_nhwc_depthwise_packed_weight_count(channels, kernel_h, kernel_w, &packed_count)) {
        fprintf(stderr, "depthwise packed count failed\n");
        goto cleanup;
    }
    packed = (float*)malloc((size_t)packed_count * sizeof(float));
    if (packed == NULL) goto cleanup;
    fill_values(input, input_count, 0x3du);
    fill_values(weights, weight_count, 0x7fu);
    lw_pack_nhwc_depthwise_f32(weights, channels, kernel_h, kernel_w, packed);
    {
        lw_nhwc_depthwise_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u;
        desc.channels = channels;
        desc.input_height = input_height;
        desc.input_width = input_width;
        desc.output_height = output_height;
        desc.output_width = output_width;
        desc.kernel_h = kernel_h;
        desc.kernel_w = kernel_w;
        desc.stride_h = stride_h;
        desc.stride_w = stride_w;
        desc.pad_top = pad_top;
        desc.pad_left = pad_left;
        desc.pad_bottom = pad_top;
        desc.pad_right = pad_left;
        if (lw_avx2_fma_nhwc_depthwise_f32(input, packed, NULL, full, &desc) != LW_STATUS_OK) {
            fprintf(stderr, "depthwise full kernel failed\n");
            goto cleanup;
        }
        /* Two sharded halves with row offsets. */
        {
            uint32_t part;
            for (part = 0u; part < 2u; ++part) {
                uint32_t begin = part * split;
                uint32_t end = part == 1u ? output_height : begin + split;
                desc.output_height = end - begin;
                desc.output_row_offset = begin;
                if (lw_avx2_fma_nhwc_depthwise_f32(
                        input, packed, NULL,
                        sharded + (size_t)begin * output_width * channels, &desc) !=
                    LW_STATUS_OK) {
                    fprintf(stderr, "depthwise shard %u kernel failed\n", part);
                    goto cleanup;
                }
            }
        }
    }
    ok = memcmp(full, sharded, (size_t)output_count * sizeof(float)) == 0;
    printf("{\"case\":\"depthwise-rows-c%u-%ux%u-k%ux%u-s%ux%u\",\"bit_exact\":%d}\n",
           channels, input_height, input_width, kernel_h, kernel_w, stride_h, stride_w, ok);
cleanup:
    free(sharded);
    free(full);
    free(packed);
    free(weights);
    free(input);
    return ok;
}


static int test_convert(uint32_t channels, uint32_t height, uint32_t width, uint32_t batch) {
    uint64_t count = (uint64_t)batch * height * width * channels;
    float* nchw = (float*)malloc((size_t)count * sizeof(float));
    float* nhwc = (float*)malloc((size_t)count * sizeof(float));
    float* roundtrip = (float*)malloc((size_t)count * sizeof(float));
    float* reference = (float*)malloc((size_t)count * sizeof(float));
    uint32_t n;
    int ok = 0;
    if (nchw == NULL || nhwc == NULL || roundtrip == NULL || reference == NULL) goto cleanup;
    fill_values(nchw, count, 0xABu);
    lw_x64_fast_nchw_to_nhwc(nchw, nhwc, batch, channels, height, width);
    lw_x64_fast_nhwc_to_nchw(nhwc, roundtrip, batch, channels, height, width);
    /* Scalar reference for the NHWC layout. */
    for (n = 0u; n < batch; ++n) {
        uint32_t y;
        for (y = 0u; y < height; ++y) {
            uint32_t x;
            for (x = 0u; x < width; ++x) {
                uint32_t c;
                for (c = 0u; c < channels; ++c) {
                    reference[(((size_t)n * height + y) * width + x) * channels + c] =
                        nchw[((size_t)n * channels + c) * height * width +
                             (size_t)y * width + x];
                }
            }
        }
    }
    ok = memcmp(nhwc, reference, (size_t)count * sizeof(float)) == 0 &&
         memcmp(roundtrip, nchw, (size_t)count * sizeof(float)) == 0;
    printf("{\"case\":\"convert-c%u-%ux%u-b%u\",\"bit_exact\":%d}\n",
           channels, height, width, batch, ok);
cleanup:
    free(reference);
    free(roundtrip);
    free(nhwc);
    free(nchw);
    return ok;
}

int main(void) {
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    if (!lw_simd_level_is_avx2(capabilities.simd) || !capabilities.has_avx2_fma) {
        printf("{\"status\":\"skipped\",\"reason\":\"requires_avx2_fma\"}\n");
        return 0;
    }
    if (!test_convtranspose_16(16u, 16u, 7u, 9u, 1u, LW_NHWC_ACT_NONE, 0)) return 1;
    if (!test_convtranspose_16(16u, 16u, 7u, 9u, 2u, LW_NHWC_ACT_NONE, 0)) return 1;
    if (!test_convtranspose_16(16u, 32u, 5u, 11u, 1u, LW_NHWC_ACT_RELU, 1)) return 1;
    if (!test_convtranspose_16(12u, 16u, 6u, 8u, 1u, LW_NHWC_ACT_NONE, 0)) return 1;
    if (!test_convtranspose_16(16u, 16u, 160u, 160u, 1u, LW_NHWC_ACT_NONE, 0)) return 1;
    if (!test_convtranspose_16(24u, 24u, 7u, 9u, 1u, LW_NHWC_ACT_NONE, 0)) return 1;
    if (!test_convtranspose_16(24u, 24u, 5u, 11u, 2u, LW_NHWC_ACT_RELU, 1)) return 1;
    if (!test_convtranspose_c1(16u, 7u, 9u, 1u, 0)) return 1;
    if (!test_convtranspose_c1(16u, 5u, 11u, 2u, 1)) return 1;
    if (!test_convtranspose_c1(24u, 6u, 8u, 1u, 0)) return 1;
    if (!test_resize(16u, 5u, 7u, 2u, 2u, 1u)) return 1;
    if (!test_resize(64u, 4u, 9u, 2u, 4u, 2u)) return 1;
    if (!test_resize(16u, 20u, 40u, 8u, 4u, 1u)) return 1;
    if (!test_depthwise_rows(32u, 16u, 32u, 3u, 3u, 1u, 1u, 1u, 1u)) return 1;
    if (!test_depthwise_rows(48u, 9u, 17u, 5u, 5u, 1u, 1u, 2u, 2u)) return 1;
    if (!test_depthwise_rows(64u, 8u, 16u, 3u, 3u, 2u, 2u, 1u, 1u)) return 1;
    if (!test_convert(16u, 5u, 9u, 1u)) return 1;
    if (!test_convert(24u, 8u, 13u, 2u)) return 1;
    if (!test_convert(64u, 3u, 7u, 1u)) return 1;
    if (!test_convert(3u, 32u, 64u, 1u)) return 1;
    if (!test_convert(12u, 16u, 33u, 2u)) return 1;
    return 0;
}
