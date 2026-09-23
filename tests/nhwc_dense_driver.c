#include "cpu_features.h"
#include "nhwc_internal.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct dense_case {
    const char* name;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
} dense_case;

static const dense_case k_cases[] = {
    {"border-3x3-s1", 37u, 32u, 7u, 13u, 3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u},
    {"border-5x5-s2", 64u, 48u, 11u, 17u, 5u, 5u, 2u, 2u, 2u, 2u, 2u, 2u},
    {"blocked-3x3-s1", 192u, 64u, 8u, 19u, 3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u},
    {"medium-rec-3x3-h6-w240", 96u, 192u, 6u, 240u, 3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u},
    /* DET graph shapes: asymmetric SAME_UPPER 2x2 pads (0,0,1,1) and the
     * stride-2 FPN stem. */
    {"det-2x2-s1-pads0011", 8u, 16u, 7u, 13u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 1u},
    {"det-3x3-s2-pad1", 32u, 16u, 9u, 17u, 3u, 3u, 2u, 2u, 1u, 1u, 1u, 1u},
    {"det-stem-3x3-s2-ic3", 3u, 16u, 9u, 17u, 3u, 3u, 2u, 2u, 1u, 1u, 1u, 1u},
    {"det-graph-stem-16x32", 3u, 16u, 16u, 32u, 3u, 3u, 2u, 2u, 1u, 1u, 1u, 1u},
    {"det-head-3x3-s1-64-16", 64u, 16u, 7u, 13u, 3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u},
    {"border-3x3-s1-asym", 37u, 32u, 7u, 13u, 3u, 3u, 1u, 1u, 1u, 1u, 0u, 2u},
    /* Small-model 24-channel family: partial 8-lane tail blocks. */
    {"det-24ch-3x3-s2", 48u, 24u, 9u, 17u, 3u, 3u, 2u, 2u, 1u, 1u, 1u, 1u},
    {"det-8ch-2x2-s1-pads0011", 12u, 8u, 7u, 13u, 2u, 2u, 1u, 1u, 0u, 0u, 1u, 1u},
};

static void fill_values(float* values, uint64_t count, uint32_t seed) {
    uint32_t state = seed;
    for (uint64_t index = 0u; index < count; ++index) {
        state = state * 1664525u + 1013904223u;
        values[(size_t)index] = (float)((int32_t)(state >> 8u) % 257) / 4096.0f;
    }
}

static uint64_t elements(uint32_t batch, uint32_t height, uint32_t width,
                         uint32_t channels) {
    return (uint64_t)batch * height * width * channels;
}

static uint32_t output_extent(uint32_t input, uint32_t kernel, uint32_t stride,
                              uint32_t pad_before, uint32_t pad_after) {
    return (input + pad_before + pad_after - kernel) / stride + 1u;
}

static void reference(const float* input, const float* weights, const float* bias,
                      float* output, const dense_case* test, uint32_t batch,
                      int apply_relu) {
    uint32_t output_height = output_extent(test->input_height, test->kernel_h,
                                           test->stride_h, test->pad_top, test->pad_bottom);
    uint32_t output_width = output_extent(test->input_width, test->kernel_w,
                                          test->stride_w, test->pad_left, test->pad_right);
    for (uint32_t b = 0u; b < batch; ++b) {
        for (uint32_t oy = 0u; oy < output_height; ++oy) {
            int32_t iy0 = (int32_t)(oy * test->stride_h) - (int32_t)test->pad_top;
            for (uint32_t ox = 0u; ox < output_width; ++ox) {
                int32_t ix0 = (int32_t)(ox * test->stride_w) - (int32_t)test->pad_left;
                for (uint32_t oc = 0u; oc < test->output_channels; ++oc) {
                    float sum = bias[oc];
                    for (uint32_t ky = 0u; ky < test->kernel_h; ++ky) {
                        int32_t iy = iy0 + (int32_t)ky;
                        if (iy < 0 || iy >= (int32_t)test->input_height) continue;
                        for (uint32_t kx = 0u; kx < test->kernel_w; ++kx) {
                            int32_t ix = ix0 + (int32_t)kx;
                            if (ix < 0 || ix >= (int32_t)test->input_width) continue;
                            for (uint32_t ic = 0u; ic < test->input_channels; ++ic) {
                                uint64_t in_index =
                                    (((uint64_t)b * test->input_height + (uint32_t)iy) *
                                     test->input_width + (uint32_t)ix) * test->input_channels + ic;
                                uint64_t weight_index =
                                    ((((uint64_t)oc * test->input_channels + ic) *
                                      test->kernel_h + ky) * test->kernel_w) + kx;
                                sum += input[(size_t)in_index] * weights[(size_t)weight_index];
                            }
                        }
                    }
                    if (apply_relu && sum < 0.0f) sum = 0.0f;
                    output[(((size_t)b * output_height + oy) * output_width + ox) *
                           test->output_channels + oc] = sum;
                }
            }
        }
    }
}

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static double median(double* values, size_t count) {
    for (size_t i = 0u; i < count; ++i) {
        for (size_t j = i + 1u; j < count; ++j) {
            if (values[j] < values[i]) {
                double temporary = values[i];
                values[i] = values[j];
                values[j] = temporary;
            }
        }
    }
    return values[count / 2u];
}
static float max_abs(const float* left, const float* right, uint64_t count) {
    float result = 0.0f;
    for (uint64_t i = 0u; i < count; ++i) {
        float difference = fabsf(left[(size_t)i] - right[(size_t)i]);
        if (!isfinite(difference) || difference > result) result = difference;
    }
    return result;
}

static int check_pack(const dense_case* test, const float* weights,
                      const float* packed) {
    uint32_t taps = test->kernel_h * test->kernel_w;
    for (uint32_t ic = 0u; ic < test->input_channels; ++ic) {
        for (uint32_t tap = 0u; tap < taps; ++tap) {
            uint32_t ky = tap / test->kernel_w;
            uint32_t kx = tap - ky * test->kernel_w;
            for (uint32_t lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                uint64_t packed_index = ((uint64_t)ic * taps + tap) * LW_NHWC_OC_BLOCK + lane;
                float expected = 0.0f;
                if (lane < test->output_channels) {
                    uint64_t weight_index = (((uint64_t)lane * test->input_channels + ic) *
                                             test->kernel_h + ky) * test->kernel_w + kx;
                    expected = weights[(size_t)weight_index];
                }
                if (packed[(size_t)packed_index] != expected) return 0;
            }
        }
    }
    return 1;
}

int main(void) {
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    if (!lw_simd_level_is_avx2(capabilities.simd) || !capabilities.has_avx2_fma) {
        printf("{\"status\":\"skipped\",\"reason\":\"requires_avx2_fma\"}\n");
        return 0;
    }
    const uint32_t dense_kc_values[] = {128u, 256u, 512u, 1024u, 0u};
    const uint32_t batch = 2u;
    for (size_t case_index = 0u; case_index < sizeof(k_cases) / sizeof(k_cases[0]); ++case_index) {
        const dense_case* test = &k_cases[case_index];
        uint32_t output_height = output_extent(test->input_height, test->kernel_h,
                                               test->stride_h, test->pad_top, test->pad_bottom);
        uint32_t output_width = output_extent(test->input_width, test->kernel_w,
                                              test->stride_w, test->pad_left, test->pad_right);
        uint64_t input_count = elements(batch, test->input_height, test->input_width,
                                        test->input_channels);
        uint64_t output_count = elements(batch, output_height, output_width,
                                         test->output_channels);
        uint64_t weight_count = (uint64_t)test->output_channels * test->input_channels *
                                test->kernel_h * test->kernel_w;
        uint64_t packed_count = 0u;
        if (!lw_nhwc_dense_packed_weight_count(test->input_channels, test->output_channels,
                                               test->kernel_h, test->kernel_w, &packed_count)) {
            fprintf(stderr, "weight count failed for %s\n", test->name);
            return 1;
        }
        float* input = (float*)malloc((size_t)input_count * sizeof(float));
        float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
        float* packed = (float*)calloc((size_t)packed_count, sizeof(float));
        float* bias = (float*)malloc((size_t)test->output_channels * sizeof(float));
        float* expected = (float*)malloc((size_t)output_count * sizeof(float));
        float* actual = (float*)malloc((size_t)output_count * sizeof(float));
        if (input == NULL || weights == NULL || packed == NULL || bias == NULL ||
            expected == NULL || actual == NULL) return 1;
        fill_values(input, input_count, (uint32_t)(91u + case_index));
        fill_values(weights, weight_count, (uint32_t)(173u + case_index));
        fill_values(bias, test->output_channels, (uint32_t)(251u + case_index));
        for (uint32_t bias_index = 0u; bias_index < test->output_channels; ++bias_index) {
            bias[bias_index] += 2.0f;
        }
        lw_pack_nhwc_dense_f32(weights, test->input_channels, test->output_channels,
                               test->kernel_h, test->kernel_w, packed);
        if (!check_pack(test, weights, packed)) {
            fprintf(stderr, "pack layout failed for %s\n", test->name);
            return 1;
        }
        reference(input, weights, bias, expected, test, batch, 1);
        for (size_t kc_index = 0u; kc_index < sizeof(dense_kc_values) / sizeof(dense_kc_values[0]); ++kc_index) {
            lw_nhwc_dense_desc desc = {batch, test->input_channels, test->input_height,
                test->input_width, test->output_channels, output_height, output_width,
                test->kernel_h, test->kernel_w, test->stride_h, test->stride_w,
                test->pad_top, test->pad_left, test->pad_bottom, test->pad_right, dense_kc_values[kc_index]};
            lw_nhwc_epilogue epilogue = {bias, NULL, LW_NHWC_ACT_RELU, 0u, 0.0f, 0.0f};
            uint64_t scratch_bytes = 0u;
            if (!lw_nhwc_dense_scratch_bytes(&desc, &scratch_bytes)) return 1;
            void* scratch = malloc((size_t)scratch_bytes);
            if (scratch == NULL) return 1;
            memset(actual, 0, (size_t)output_count * sizeof(float));
            if (lw_avx2_fma_nhwc_dense_f32(input, packed, &epilogue, actual, &desc,
                                           scratch, scratch_bytes) != LW_STATUS_OK) {
                fprintf(stderr, "kernel failed for %s kc=%u\n", test->name,
                        dense_kc_values[kc_index]);
                free(scratch);
                return 1;
            }
            float difference = max_abs(expected, actual, output_count);
            printf("{\"case\":\"%s\",\"kc\":%u,\"max_abs\":%.9g,\"scratch_bytes\":%" PRIu64 "}\n",
                   test->name, dense_kc_values[kc_index], difference, scratch_bytes);
            free(scratch);
            if (difference > 1.0e-4f) return 1;
        }
        /* HARDSWISH store-path parity for the full-16-block path. */
        {
            lw_nhwc_dense_desc desc = {batch, test->input_channels, test->input_height,
                test->input_width, test->output_channels, output_height, output_width,
                test->kernel_h, test->kernel_w, test->stride_h, test->stride_w,
                test->pad_top, test->pad_left, test->pad_bottom, test->pad_right, 512u};
            lw_nhwc_epilogue epilogue = {bias, NULL, LW_NHWC_ACT_HARDSWISH, 0u, 0.0f, 0.0f};
            uint64_t scratch_bytes = 0u;
            void* scratch;
            uint64_t i;
            float hardswish_difference;
            reference(input, weights, bias, expected, test, batch, 0);
            for (i = 0u; i < output_count; ++i) {
                float value = expected[i];
                float gate = value + 3.0f;
                if (gate < 0.0f) gate = 0.0f;
                if (gate > 6.0f) gate = 6.0f;
                expected[i] = value * gate * (1.0f / 6.0f);
            }
            if (!lw_nhwc_dense_scratch_bytes(&desc, &scratch_bytes)) return 1;
            scratch = malloc((size_t)scratch_bytes);
            if (scratch == NULL) return 1;
            memset(actual, 0, (size_t)output_count * sizeof(float));
            if (lw_avx2_fma_nhwc_dense_f32(input, packed, &epilogue, actual, &desc,
                                           scratch, scratch_bytes) != LW_STATUS_OK) {
                fprintf(stderr, "hardswish kernel failed for %s\n", test->name);
                free(scratch);
                return 1;
            }
            hardswish_difference = max_abs(expected, actual, output_count);
            printf("{\"case\":\"%s-hardswish\",\"max_abs\":%.9g}\n", test->name,
                   hardswish_difference);
            free(scratch);
            if (hardswish_difference > 1.0e-4f) return 1;
            /* Restore the ReLU reference for the perf section below. */
            reference(input, weights, bias, expected, test, batch, 1);
        }
        {
            lw_nhwc_dense_desc desc = {batch, test->input_channels, test->input_height,
                test->input_width, test->output_channels, output_height, output_width,
                test->kernel_h, test->kernel_w, test->stride_h, test->stride_w,
                test->pad_top, test->pad_left, test->pad_bottom, test->pad_right, 512u};
            lw_nhwc_epilogue epilogue = {bias, NULL, LW_NHWC_ACT_RELU, 0u, 0.0f, 0.0f};
            uint64_t scratch_bytes = 0u;
            double scalar_samples[5];
            double dense_samples[5];
            void* scratch;
            if (!lw_nhwc_dense_scratch_bytes(&desc, &scratch_bytes)) return 1;
            scratch = malloc((size_t)scratch_bytes);
            if (scratch == NULL) return 1;
            reference(input, weights, bias, expected, test, batch, 1);
            (void)lw_avx2_fma_nhwc_dense_f32(input, packed, &epilogue, actual, &desc,
                                             scratch, scratch_bytes);
            for (size_t round = 0u; round < 5u; ++round) {
                double start = monotonic_seconds();
                reference(input, weights, bias, expected, test, batch, 1);
                scalar_samples[round] = (monotonic_seconds() - start) * 1000.0;
                start = monotonic_seconds();
                if (lw_avx2_fma_nhwc_dense_f32(input, packed, &epilogue, actual, &desc,
                                               scratch, scratch_bytes) != LW_STATUS_OK) {
                    free(scratch);
                    return 1;
                }
                dense_samples[round] = (monotonic_seconds() - start) * 1000.0;
            }
            {
                double scalar_ms = median(scalar_samples, 5u);
                double dense_ms = median(dense_samples, 5u);
                printf("{\"perf_case\":\"%s\",\"kc\":512,\"scalar_ms\":%.6f,\"dense_ms\":%.6f,\"speedup\":%.6f,\"scratch_bytes\":%" PRIu64 "}\n",
                       test->name, scalar_ms, dense_ms, scalar_ms / dense_ms, scratch_bytes);
            }
            free(scratch);
        }
        free(input);
        free(weights);
        free(packed);
        free(bias);
        free(expected);
        free(actual);
    }
    return 0;
}
