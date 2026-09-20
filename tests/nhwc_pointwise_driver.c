#include "cpu_features.h"
#include "nhwc_internal.h"
#include "packed_conv_internal.h"

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

typedef struct nhwc_case {
    const char* name;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t height;
    uint32_t width;
} nhwc_case;

static const nhwc_case k_cases[] = {
    {"medium-rec-96x192-h6-w240", 96u, 192u, 6u, 240u},
    {"medium-rec-192x96-h6-w240", 192u, 96u, 6u, 240u},
    {"medium-rec-512x1024-h6-w240", 512u, 1024u, 6u, 240u},
    {"medium-rec-1024x512-h6-w240", 1024u, 512u, 6u, 240u},
    {"medium-rec-1536x768-h3-w240", 1536u, 768u, 3u, 240u},
};

enum { k_measure_rounds = 9, k_warmup_rounds = 2 };
typedef struct nhwc_chain_case {
    const char* name;
    uint32_t input_channels;
    uint32_t middle_channels;
    uint32_t output_channels;
    uint32_t height;
    uint32_t width;
} nhwc_chain_case;

static const nhwc_chain_case k_chain_cases[] = {
    {"medium-rec-chain-512x1024x512-h6-w240", 512u, 1024u, 512u, 6u, 240u},
    {"medium-rec-chain-1024x512x768-h6-w240", 1024u, 512u, 768u, 6u, 240u},
};

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart == 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static void fill_values(float* values, uint64_t count, uint32_t seed) {
    uint64_t index;
    uint32_t state = seed;
    for (index = 0u; index < count; ++index) {
        state = state * 1664525u + 1013904223u;
        values[(size_t)index] = (float)((int32_t)(state >> 9u) % 1021) / 511.0f;
    }
}

static void scale_values(float* values, uint64_t count, float scale) {
    uint64_t index;
    for (index = 0u; index < count; ++index) {
        values[(size_t)index] *= scale;
    }
}

static uint64_t checksum_bytes(const void* data, size_t bytes) {
    const unsigned char* values = (const unsigned char*)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static float max_abs_difference(const float* left, const float* right, uint64_t count) {
    uint64_t index;
    float maximum = 0.0f;
    for (index = 0u; index < count; ++index) {
        float difference = fabsf(left[(size_t)index] - right[(size_t)index]);
        if (!isfinite(difference)) {
            return INFINITY;
        }
        if (difference > maximum) {
            maximum = difference;
        }
    }
    return maximum;
}

static float max_rel_difference(const float* left, const float* right, uint64_t count) {
    uint64_t index;
    float maximum = 0.0f;
    for (index = 0u; index < count; ++index) {
        float expected = left[(size_t)index];
        float difference = fabsf(expected - right[(size_t)index]);
        float relative = difference / fmaxf(fabsf(expected), 1.0e-12f);
        if (!isfinite(relative)) {
            return INFINITY;
        }
        if (relative > maximum) {
            maximum = relative;
        }
    }
    return maximum;
}

static uint64_t count_mismatches(const float* left, const float* right, uint64_t count) {
    uint64_t index;
    uint64_t mismatches = 0u;
    for (index = 0u; index < count; ++index) {
        if (fabsf(left[(size_t)index] - right[(size_t)index]) > 1.0e-4f) {
            ++mismatches;
        }
    }
    return mismatches;
}

static void nhwc_to_nchw(const float* input, float* output, uint32_t pixels,
                         uint32_t channels) {
    uint32_t channel;
    for (channel = 0u; channel < channels; ++channel) {
        uint32_t pixel;
        for (pixel = 0u; pixel < pixels; ++pixel) {
            output[(size_t)channel * pixels + pixel] =
                input[(size_t)pixel * channels + channel];
        }
    }
}

static void nchw_to_nhwc(const float* input, float* output, uint32_t pixels,
                         uint32_t channels) {
    uint32_t channel;
    for (channel = 0u; channel < channels; ++channel) {
        uint32_t pixel;
        for (pixel = 0u; pixel < pixels; ++pixel) {
            output[(size_t)pixel * channels + channel] =
                input[(size_t)channel * pixels + pixel];
        }
    }
}

static uint64_t count_argmax_mismatches(const float* left, const float* right,
                                        uint32_t pixels, uint32_t channels) {
    uint32_t pixel;
    uint64_t mismatches = 0u;
    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t channel;
        uint32_t left_index = 0u;
        uint32_t right_index = 0u;
        for (channel = 1u; channel < channels; ++channel) {
            if (left[(size_t)pixel * channels + channel] >
                left[(size_t)pixel * channels + left_index]) {
                left_index = channel;
            }
            if (right[(size_t)pixel * channels + channel] >
                right[(size_t)pixel * channels + right_index]) {
                right_index = channel;
            }
        }
        if (left_index != right_index) {
            ++mismatches;
        }
    }
    return mismatches;
}
static double median(double* values, size_t count) {
    size_t i;
    size_t j;
    for (i = 0u; i < count; ++i) {
        for (j = i + 1u; j < count; ++j) {
            if (values[j] < values[i]) {
                double temporary = values[i];
                values[i] = values[j];
                values[j] = temporary;
            }
        }
    }
    return values[count / 2u];
}

static void run_nchw(const float* input, const float* weights, const float* bias,
                     float* output, const int32_t input_dimensions[4],
                     const int32_t output_dimensions[4]) {
    lw_packed_conv1x1_f32(input, weights, bias, output, input_dimensions,
                          output_dimensions);
}

static void run_nhwc(const float* input, const float* weights, const float* bias,
                     float* output, uint32_t pixels, uint32_t input_channels,
                     uint32_t output_channels) {
    lw_nhwc_epilogue epilogue;
    epilogue.bias = bias;
    epilogue.residual = NULL;
    epilogue.activation = LW_NHWC_ACT_NONE;
    epilogue.reserved = 0u;
    epilogue.alpha = 0.0f;
    epilogue.beta = 0.0f;
    lw_avx2_fma_nhwc_pointwise_f32(input, weights, &epilogue, output, pixels,
                                   input_channels, output_channels);
}

static int run_case(const nhwc_case* item, uint32_t case_index) {
    uint32_t pixels = item->height * item->width;
    uint64_t input_count = (uint64_t)pixels * item->input_channels;
    uint64_t output_count = (uint64_t)pixels * item->output_channels;
    uint64_t canonical_weight_count =
        (uint64_t)item->input_channels * item->output_channels;
    uint64_t nhwc_weight_count = 0u;
    uint64_t nchw_weight_count = 0u;
    float* input_nhwc = NULL;
    float* input_nchw = NULL;
    float* canonical_weights = NULL;
    float* nhwc_weights = NULL;
    float* nchw_weights = NULL;
    float* bias = NULL;
    float* output_nhwc = NULL;
    float* output_nchw = NULL;
    float* reference_nhwc = NULL;
    double nchw_samples[k_measure_rounds];
    double nhwc_samples[k_measure_rounds];
    int32_t input_dimensions[4] = {1, (int32_t)item->input_channels,
                                   (int32_t)item->height, (int32_t)item->width};
    int32_t output_dimensions[4] = {1, (int32_t)item->output_channels,
                                    (int32_t)item->height, (int32_t)item->width};
    uint32_t round;
    float max_abs;
    float max_rel;
    uint64_t mismatches;
    uint64_t argmax_mismatches;
    uint64_t nchw_checksum;
    uint64_t nhwc_checksum;
    int success = 0;

    if (!lw_nhwc_dense_packed_weight_count(item->input_channels, item->output_channels,
                                           1u, 1u, &nhwc_weight_count) ||
        !lw_packed_conv1x1_weight_count(item->input_channels, item->output_channels,
                                         &nchw_weight_count)) {
        return 0;
    }
    input_nhwc = (float*)malloc((size_t)input_count * sizeof(float));
    input_nchw = (float*)malloc((size_t)input_count * sizeof(float));
    canonical_weights = (float*)malloc((size_t)canonical_weight_count * sizeof(float));
    nhwc_weights = (float*)malloc((size_t)nhwc_weight_count * sizeof(float));
    nchw_weights = (float*)malloc((size_t)nchw_weight_count * sizeof(float));
    bias = (float*)malloc((size_t)item->output_channels * sizeof(float));
    output_nhwc = (float*)malloc((size_t)output_count * sizeof(float));
    output_nchw = (float*)malloc((size_t)output_count * sizeof(float));
    reference_nhwc = (float*)malloc((size_t)output_count * sizeof(float));
    if (input_nhwc == NULL || input_nchw == NULL || canonical_weights == NULL ||
        nhwc_weights == NULL || nchw_weights == NULL || bias == NULL ||
        output_nhwc == NULL || output_nchw == NULL || reference_nhwc == NULL) {
        goto cleanup;
    }
    fill_values(input_nhwc, input_count, 1009u + case_index * 17u);
    fill_values(canonical_weights, canonical_weight_count, 2003u + case_index * 31u);
    fill_values(bias, item->output_channels, 3001u + case_index * 43u);
    scale_values(input_nhwc, input_count, 0.25f);
    scale_values(canonical_weights, canonical_weight_count, 0.25f);
    scale_values(bias, item->output_channels, 0.25f);
    nhwc_to_nchw(input_nhwc, input_nchw, pixels, item->input_channels);
    lw_pack_nhwc_dense_f32(canonical_weights, item->input_channels,
                           item->output_channels, 1u, 1u, nhwc_weights);
    lw_pack_conv1x1_weights_f32(canonical_weights, item->input_channels,
                                item->output_channels, nchw_weights);

    run_nchw(input_nchw, nchw_weights, bias, output_nchw, input_dimensions,
             output_dimensions);
    nchw_to_nhwc(output_nchw, reference_nhwc, pixels, item->output_channels);
    run_nhwc(input_nhwc, nhwc_weights, bias, output_nhwc, pixels,
             item->input_channels, item->output_channels);
    max_abs = max_abs_difference(reference_nhwc, output_nhwc, output_count);
    max_rel = max_rel_difference(reference_nhwc, output_nhwc, output_count);
    mismatches = count_mismatches(reference_nhwc, output_nhwc, output_count);
    argmax_mismatches = count_argmax_mismatches(reference_nhwc, output_nhwc, pixels,
                                                  item->output_channels);
    if (max_abs > 1.0e-4f || max_rel > 1.0e-4f || mismatches != 0u ||
        argmax_mismatches != 0u) {
        fprintf(stderr, "NHWC parity failed for %s: max_abs=%.9g max_rel=%.9g output_mismatches=%" PRIu64 " argmax_mismatches=%" PRIu64 "\n",
                item->name, max_abs, max_rel, mismatches, argmax_mismatches);
        goto cleanup;
    }

    for (round = 0u; round < k_warmup_rounds; ++round) {
        run_nchw(input_nchw, nchw_weights, bias, output_nchw, input_dimensions,
                 output_dimensions);
        run_nhwc(input_nhwc, nhwc_weights, bias, output_nhwc, pixels,
                 item->input_channels, item->output_channels);
    }
    for (round = 0u; round < k_measure_rounds; ++round) {
        double start;
        double end;
        if ((round & 1u) == 0u) {
            start = monotonic_seconds();
            run_nchw(input_nchw, nchw_weights, bias, output_nchw, input_dimensions,
                     output_dimensions);
            end = monotonic_seconds();
            nchw_samples[round] = (end - start) * 1000.0;
            start = monotonic_seconds();
            run_nhwc(input_nhwc, nhwc_weights, bias, output_nhwc, pixels,
                     item->input_channels, item->output_channels);
            end = monotonic_seconds();
            nhwc_samples[round] = (end - start) * 1000.0;
        } else {
            start = monotonic_seconds();
            run_nhwc(input_nhwc, nhwc_weights, bias, output_nhwc, pixels,
                     item->input_channels, item->output_channels);
            end = monotonic_seconds();
            nhwc_samples[round] = (end - start) * 1000.0;
            start = monotonic_seconds();
            run_nchw(input_nchw, nchw_weights, bias, output_nchw, input_dimensions,
                     output_dimensions);
            end = monotonic_seconds();
            nchw_samples[round] = (end - start) * 1000.0;
        }
    }
    nchw_checksum = checksum_bytes(output_nchw, (size_t)output_count * sizeof(float));
    nhwc_checksum = checksum_bytes(output_nhwc, (size_t)output_count * sizeof(float));
    printf("    {\"name\":\"%s\",\"input_channels\":%u,\"output_channels\":%u,\"height\":%u,\"width\":%u,\"nchw_ms\":%.6f,\"nhwc_ms\":%.6f,\"speedup\":%.6f,\"max_abs\":%.9g,\"max_rel\":%.9g,\"output_mismatch_count\":%" PRIu64 ",\"argmax_mismatch_count\":%" PRIu64 ",\"nchw_checksum\":\"%016" PRIx64 "\",\"nhwc_checksum\":\"%016" PRIx64 "\"}%s\n",
           item->name, item->input_channels, item->output_channels, item->height,
           item->width, median(nchw_samples, k_measure_rounds),
           median(nhwc_samples, k_measure_rounds),
           median(nchw_samples, k_measure_rounds) / median(nhwc_samples, k_measure_rounds),
           max_abs, max_rel, mismatches, argmax_mismatches, nchw_checksum, nhwc_checksum,
           case_index + 1u < sizeof(k_cases) / sizeof(k_cases[0]) ? "," : "");
    success = 1;

cleanup:
    free(input_nhwc);
    free(input_nchw);
    free(canonical_weights);
    free(nhwc_weights);
    free(nchw_weights);
    free(bias);
    free(output_nhwc);
    free(output_nchw);
    free(reference_nhwc);
    return success;
}


static int run_chain_case(const nhwc_chain_case* item, uint32_t case_index) {
    uint32_t pixels = item->height * item->width;
    uint64_t input_count = (uint64_t)pixels * item->input_channels;
    uint64_t middle_count = (uint64_t)pixels * item->middle_channels;
    uint64_t output_count = (uint64_t)pixels * item->output_channels;
    uint64_t first_canonical_count =
        (uint64_t)item->input_channels * item->middle_channels;
    uint64_t second_canonical_count =
        (uint64_t)item->middle_channels * item->output_channels;
    uint64_t first_nhwc_count = 0u;
    uint64_t second_nhwc_count = 0u;
    uint64_t first_nchw_count = 0u;
    uint64_t second_nchw_count = 0u;
    float* input_nhwc = NULL;
    float* input_nchw = NULL;
    float* nhwc_middle = NULL;
    float* nhwc_output = NULL;
    float* nchw_middle = NULL;
    float* nchw_output = NULL;
    float* reference_nhwc = NULL;
    float* first_canonical = NULL;
    float* second_canonical = NULL;
    float* first_nhwc_weights = NULL;
    float* second_nhwc_weights = NULL;
    float* first_nchw_weights = NULL;
    float* second_nchw_weights = NULL;
    float* first_bias = NULL;
    float* second_bias = NULL;
    double nchw_samples[k_measure_rounds];
    double nhwc_samples[k_measure_rounds];
    int32_t input_dimensions[4] = {1, (int32_t)item->input_channels,
                                   (int32_t)item->height, (int32_t)item->width};
    int32_t middle_dimensions[4] = {1, (int32_t)item->middle_channels,
                                    (int32_t)item->height, (int32_t)item->width};
    int32_t output_dimensions[4] = {1, (int32_t)item->output_channels,
                                    (int32_t)item->height, (int32_t)item->width};
    uint32_t round;
    float max_abs;
    float max_rel;
    uint64_t output_mismatches;
    uint64_t argmax_mismatches;
    uint64_t nchw_checksum;
    uint64_t nhwc_checksum;
    int success = 0;

    if (!lw_nhwc_dense_packed_weight_count(item->input_channels, item->middle_channels,
                                           1u, 1u, &first_nhwc_count) ||
        !lw_nhwc_dense_packed_weight_count(item->middle_channels, item->output_channels,
                                           1u, 1u, &second_nhwc_count) ||
        !lw_packed_conv1x1_weight_count(item->input_channels, item->middle_channels,
                                         &first_nchw_count) ||
        !lw_packed_conv1x1_weight_count(item->middle_channels, item->output_channels,
                                         &second_nchw_count)) {
        return 0;
    }
    input_nhwc = (float*)malloc((size_t)input_count * sizeof(float));
    input_nchw = (float*)malloc((size_t)input_count * sizeof(float));
    nhwc_middle = (float*)malloc((size_t)middle_count * sizeof(float));
    nhwc_output = (float*)malloc((size_t)output_count * sizeof(float));
    nchw_middle = (float*)malloc((size_t)middle_count * sizeof(float));
    nchw_output = (float*)malloc((size_t)output_count * sizeof(float));
    reference_nhwc = (float*)malloc((size_t)output_count * sizeof(float));
    first_canonical = (float*)malloc((size_t)first_canonical_count * sizeof(float));
    second_canonical = (float*)malloc((size_t)second_canonical_count * sizeof(float));
    first_nhwc_weights = (float*)malloc((size_t)first_nhwc_count * sizeof(float));
    second_nhwc_weights = (float*)malloc((size_t)second_nhwc_count * sizeof(float));
    first_nchw_weights = (float*)malloc((size_t)first_nchw_count * sizeof(float));
    second_nchw_weights = (float*)malloc((size_t)second_nchw_count * sizeof(float));
    first_bias = (float*)malloc((size_t)item->middle_channels * sizeof(float));
    second_bias = (float*)malloc((size_t)item->output_channels * sizeof(float));
    if (input_nhwc == NULL || input_nchw == NULL || nhwc_middle == NULL ||
        nhwc_output == NULL || nchw_middle == NULL || nchw_output == NULL ||
        reference_nhwc == NULL || first_canonical == NULL || second_canonical == NULL ||
        first_nhwc_weights == NULL || second_nhwc_weights == NULL ||
        first_nchw_weights == NULL || second_nchw_weights == NULL ||
        first_bias == NULL || second_bias == NULL) {
        goto cleanup;
    }
    fill_values(input_nhwc, input_count, 4001u + case_index * 13u);
    fill_values(first_canonical, first_canonical_count, 5003u + case_index * 17u);
    fill_values(second_canonical, second_canonical_count, 6007u + case_index * 19u);
    fill_values(first_bias, item->middle_channels, 7001u + case_index * 23u);
    fill_values(second_bias, item->output_channels, 8009u + case_index * 29u);
    scale_values(input_nhwc, input_count, 0.25f);
    scale_values(first_canonical, first_canonical_count, 0.025f);
    scale_values(second_canonical, second_canonical_count, 0.025f);
    scale_values(first_bias, item->middle_channels, 0.025f);
    scale_values(second_bias, item->output_channels, 0.025f);
    nhwc_to_nchw(input_nhwc, input_nchw, pixels, item->input_channels);
    lw_pack_nhwc_dense_f32(first_canonical, item->input_channels, item->middle_channels,
                           1u, 1u, first_nhwc_weights);
    lw_pack_nhwc_dense_f32(second_canonical, item->middle_channels, item->output_channels,
                           1u, 1u, second_nhwc_weights);
    lw_pack_conv1x1_weights_f32(first_canonical, item->input_channels,
                                item->middle_channels, first_nchw_weights);
    lw_pack_conv1x1_weights_f32(second_canonical, item->middle_channels,
                                item->output_channels, second_nchw_weights);

    run_nchw(input_nchw, first_nchw_weights, first_bias, nchw_middle,
             input_dimensions, middle_dimensions);
    run_nchw(nchw_middle, second_nchw_weights, second_bias, nchw_output,
             middle_dimensions, output_dimensions);
    nchw_to_nhwc(nchw_output, reference_nhwc, pixels, item->output_channels);
    run_nhwc(input_nhwc, first_nhwc_weights, first_bias, nhwc_middle, pixels,
             item->input_channels, item->middle_channels);
    run_nhwc(nhwc_middle, second_nhwc_weights, second_bias, nhwc_output, pixels,
             item->middle_channels, item->output_channels);
    max_abs = max_abs_difference(reference_nhwc, nhwc_output, output_count);
    max_rel = max_rel_difference(reference_nhwc, nhwc_output, output_count);
    output_mismatches = count_mismatches(reference_nhwc, nhwc_output, output_count);
    argmax_mismatches = count_argmax_mismatches(reference_nhwc, nhwc_output, pixels,
                                                item->output_channels);
    if (max_abs > 1.0e-4f || max_rel > 1.0e-4f || output_mismatches != 0u ||
        argmax_mismatches != 0u) {
        fprintf(stderr, "NHWC chain parity failed for %s: max_abs=%.9g max_rel=%.9g output_mismatches=%" PRIu64 " argmax_mismatches=%" PRIu64 "\n",
                item->name, max_abs, max_rel, output_mismatches, argmax_mismatches);
        goto cleanup;
    }
    for (round = 0u; round < k_warmup_rounds; ++round) {
        run_nchw(input_nchw, first_nchw_weights, first_bias, nchw_middle,
                 input_dimensions, middle_dimensions);
        run_nchw(nchw_middle, second_nchw_weights, second_bias, nchw_output,
                 middle_dimensions, output_dimensions);
        run_nhwc(input_nhwc, first_nhwc_weights, first_bias, nhwc_middle, pixels,
                 item->input_channels, item->middle_channels);
        run_nhwc(nhwc_middle, second_nhwc_weights, second_bias, nhwc_output, pixels,
                 item->middle_channels, item->output_channels);
    }
    for (round = 0u; round < k_measure_rounds; ++round) {
        double start;
        double end;
        if ((round & 1u) == 0u) {
            start = monotonic_seconds();
            run_nchw(input_nchw, first_nchw_weights, first_bias, nchw_middle,
                     input_dimensions, middle_dimensions);
            run_nchw(nchw_middle, second_nchw_weights, second_bias, nchw_output,
                     middle_dimensions, output_dimensions);
            end = monotonic_seconds();
            nchw_samples[round] = (end - start) * 1000.0;
            start = monotonic_seconds();
            run_nhwc(input_nhwc, first_nhwc_weights, first_bias, nhwc_middle, pixels,
                     item->input_channels, item->middle_channels);
            run_nhwc(nhwc_middle, second_nhwc_weights, second_bias, nhwc_output, pixels,
                     item->middle_channels, item->output_channels);
            end = monotonic_seconds();
            nhwc_samples[round] = (end - start) * 1000.0;
        } else {
            start = monotonic_seconds();
            run_nhwc(input_nhwc, first_nhwc_weights, first_bias, nhwc_middle, pixels,
                     item->input_channels, item->middle_channels);
            run_nhwc(nhwc_middle, second_nhwc_weights, second_bias, nhwc_output, pixels,
                     item->middle_channels, item->output_channels);
            end = monotonic_seconds();
            nhwc_samples[round] = (end - start) * 1000.0;
            start = monotonic_seconds();
            run_nchw(input_nchw, first_nchw_weights, first_bias, nchw_middle,
                     input_dimensions, middle_dimensions);
            run_nchw(nchw_middle, second_nchw_weights, second_bias, nchw_output,
                     middle_dimensions, output_dimensions);
            end = monotonic_seconds();
            nchw_samples[round] = (end - start) * 1000.0;
        }
    }
    nchw_checksum = checksum_bytes(nchw_output, (size_t)output_count * sizeof(float));
    nhwc_checksum = checksum_bytes(nhwc_output, (size_t)output_count * sizeof(float));
    printf("    {\"name\":\"%s\",\"nchw_ms\":%.6f,\"nhwc_ms\":%.6f,\"speedup\":%.6f,\"max_abs\":%.9g,\"max_rel\":%.9g,\"output_mismatch_count\":%" PRIu64 ",\"argmax_mismatch_count\":%" PRIu64 ",\"nchw_checksum\":\"%016" PRIx64 "\",\"nhwc_checksum\":\"%016" PRIx64 "\"}%s\n",
           item->name, median(nchw_samples, k_measure_rounds),
           median(nhwc_samples, k_measure_rounds),
           median(nchw_samples, k_measure_rounds) / median(nhwc_samples, k_measure_rounds),
           max_abs, max_rel, output_mismatches, argmax_mismatches,
           nchw_checksum, nhwc_checksum,
           case_index + 1u < sizeof(k_chain_cases) / sizeof(k_chain_cases[0]) ? "," : "");
    success = 1;

cleanup:
    free(input_nhwc);
    free(input_nchw);
    free(nhwc_middle);
    free(nhwc_output);
    free(nchw_middle);
    free(nchw_output);
    free(reference_nhwc);
    free(first_canonical);
    free(second_canonical);
    free(first_nhwc_weights);
    free(second_nhwc_weights);
    free(first_nchw_weights);
    free(second_nchw_weights);
    free(first_bias);
    free(second_bias);
    return success;
}

int main(void) {
    const lw_cpu_capabilities capabilities = lw_get_cpu_capabilities();
    size_t index;

    if (!lw_simd_level_is_avx2(capabilities.simd) || !capabilities.has_avx2_fma) {
        printf("{\"schema_version\":1,\"status\":\"skipped\",\"reason\":\"requires_avx2_fma\"}\n");
        return 0;
    }
    printf("{\"schema_version\":1,\"status\":\"ok\",\"backend\":\"avx2+fma\",\"cases\":[\n");
    for (index = 0u; index < sizeof(k_cases) / sizeof(k_cases[0]); ++index) {
        if (!run_case(&k_cases[index], (uint32_t)index)) {
            return 1;
        }
    }
    printf("],\"chain_cases\":[\n");
    for (index = 0u; index < sizeof(k_chain_cases) / sizeof(k_chain_cases[0]); ++index) {
        if (!run_chain_case(&k_chain_cases[index], (uint32_t)index)) {
            return 1;
        }
    }
    printf("],\"promotion_gate\":\"each major hot shape must reach >=3x\"}\n");
    return 0;
}

