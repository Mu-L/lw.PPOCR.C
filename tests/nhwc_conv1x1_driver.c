#include "packed_conv_internal.h"
#include "cpu_features.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NHWC_OUTPUT_TILE 8u

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define NHWC_COMPILES_AVX2 1
#else
#  define NHWC_COMPILES_AVX2 0
#endif

static uint32_t round_up_output_channels(uint32_t output_channels) {
    return (output_channels + NHWC_OUTPUT_TILE - 1u) / NHWC_OUTPUT_TILE * NHWC_OUTPUT_TILE;
}

static void fill_tensor(float* values, uint64_t count, uint32_t seed) {
    uint64_t index;
    for (index = 0u; index < count; ++index) {
        uint32_t value = (uint32_t)((index * 2654435761u + seed) & 0xffffu);
        values[(size_t)index] = ((float)value / 32768.0f) - 1.0f;
    }
}

static void pack_nhwc_weights(const float* weights, uint32_t input_channels,
                              uint32_t output_channels, float* packed_weights) {
    uint32_t padded_outputs = round_up_output_channels(output_channels);
    uint32_t input_channel;
    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < padded_outputs; ++output_channel) {
            packed_weights[(size_t)input_channel * padded_outputs + output_channel] =
                output_channel < output_channels
                    ? weights[(size_t)output_channel * input_channels + input_channel]
                    : 0.0f;
        }
    }
}

static void nhwc_conv1x1_scalar(const float* input, const float* packed_weights,
                                const float* bias, float* output, uint32_t pixels,
                                uint32_t input_channels, uint32_t output_channels) {
    uint32_t padded_outputs = round_up_output_channels(output_channels);
    uint32_t pixel;
    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t output_channel;
        for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
            output[(size_t)pixel * output_channels + output_channel] =
                bias == NULL ? 0.0f : bias[output_channel];
        }
        for (uint32_t input_channel = 0u; input_channel < input_channels; ++input_channel) {
            float input_value = input[(size_t)pixel * input_channels + input_channel];
            const float* weights =
                packed_weights + (size_t)input_channel * padded_outputs;
            for (output_channel = 0u; output_channel < output_channels; ++output_channel) {
                output[(size_t)pixel * output_channels + output_channel] +=
                    input_value * weights[output_channel];
            }
        }
    }
}

#if NHWC_COMPILES_AVX2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
static void nhwc_conv1x1_avx2(const float* input, const float* packed_weights,
                               const float* bias, float* output, uint32_t pixels,
                               uint32_t input_channels, uint32_t output_channels) {
#if NHWC_COMPILES_AVX2
    uint32_t padded_outputs = round_up_output_channels(output_channels);
    uint32_t pixel;
    for (pixel = 0u; pixel < pixels; ++pixel) {
        uint32_t output_channel = 0u;
        for (; output_channel + NHWC_OUTPUT_TILE <= output_channels;
             output_channel += NHWC_OUTPUT_TILE) {
            __m256 accumulator =
                bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_channel);
            uint32_t input_channel;
            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                __m256 input_value =
                    _mm256_set1_ps(input[(size_t)pixel * input_channels + input_channel]);
                __m256 weights = _mm256_loadu_ps(
                    packed_weights + (size_t)input_channel * padded_outputs + output_channel);
                accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(input_value, weights));
            }
            _mm256_storeu_ps(output + (size_t)pixel * output_channels + output_channel,
                             accumulator);
        }
        for (; output_channel < output_channels; ++output_channel) {
            float accumulator = bias == NULL ? 0.0f : bias[output_channel];
            uint32_t input_channel;
            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                accumulator += input[(size_t)pixel * input_channels + input_channel] *
                               packed_weights[(size_t)input_channel * padded_outputs + output_channel];
            }
            output[(size_t)pixel * output_channels + output_channel] = accumulator;
        }
    }
#else
    nhwc_conv1x1_scalar(input, packed_weights, bias, output, pixels, input_channels,
                        output_channels);
#endif
}

static void nhwc_conv1x1_selected(const float* input, const float* packed_weights,
                                  const float* bias, float* output, uint32_t pixels,
                                  uint32_t input_channels, uint32_t output_channels) {
    if (lw_simd_level_is_avx2(lw_get_cpu_capabilities().simd)) {
        nhwc_conv1x1_avx2(input, packed_weights, bias, output, pixels, input_channels,
                          output_channels);
    } else {
        nhwc_conv1x1_scalar(input, packed_weights, bias, output, pixels, input_channels,
                            output_channels);
    }
}
static float max_nchw_nhwc_difference(const float* nchw, const float* nhwc,
                                      uint32_t pixels, uint32_t channels) {
    float maximum = 0.0f;
    uint32_t channel;
    for (channel = 0u; channel < channels; ++channel) {
        uint32_t pixel;
        for (pixel = 0u; pixel < pixels; ++pixel) {
            float difference = fabsf(nchw[(size_t)channel * pixels + pixel] -
                                    nhwc[(size_t)pixel * channels + channel]);
            if (difference > maximum) {
                maximum = difference;
            }
        }
    }
    return maximum;
}
static void nhwc_to_nchw(const float* nhwc, float* nchw, uint32_t pixels,
                         uint32_t input_channels) {
    uint32_t input_channel;
    for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
        uint32_t pixel;
        for (pixel = 0u; pixel < pixels; ++pixel) {
            nchw[(size_t)input_channel * pixels + pixel] =
                nhwc[(size_t)pixel * input_channels + input_channel];
        }
    }
}

static int run_parity_case(uint32_t input_channels, uint32_t output_channels,
                           uint32_t height, uint32_t width) {
    uint32_t pixels = height * width;
    uint32_t padded_outputs = round_up_output_channels(output_channels);
    uint64_t weight_count = (uint64_t)input_channels * output_channels;
    uint64_t nhwc_weight_count = (uint64_t)input_channels * padded_outputs;
    uint64_t value_count = (uint64_t)pixels * output_channels;
    uint64_t nchw_input_count = (uint64_t)pixels * input_channels;
    float* weights = (float*)malloc((size_t)weight_count * sizeof(float));
    float* packed_nhwc = (float*)malloc((size_t)nhwc_weight_count * sizeof(float));
    float* packed_nchw = NULL;
    float* bias = (float*)malloc((size_t)output_channels * sizeof(float));
    float* input_nhwc = (float*)malloc((size_t)nchw_input_count * sizeof(float));
    float* input_nchw = (float*)malloc((size_t)nchw_input_count * sizeof(float));
    float* expected = (float*)malloc((size_t)value_count * sizeof(float));
    float* actual = (float*)malloc((size_t)value_count * sizeof(float));
    int32_t input_dimensions[4] = {1, (int32_t)input_channels, (int32_t)height, (int32_t)width};
    int32_t output_dimensions[4] = {1, (int32_t)output_channels, (int32_t)height,
                                    (int32_t)width};
    uint64_t packed_nchw_count = 0u;
    float scalar_difference;
    float selected_difference;
    int success = 0;

    if (weights == NULL || packed_nhwc == NULL || bias == NULL || input_nhwc == NULL ||
        input_nchw == NULL || expected == NULL || actual == NULL ||
        !lw_packed_conv1x1_weight_count(input_channels, output_channels, &packed_nchw_count)) {
        goto cleanup;
    }
    packed_nchw = (float*)malloc((size_t)packed_nchw_count * sizeof(float));
    if (packed_nchw == NULL) {
        goto cleanup;
    }
    fill_tensor(weights, weight_count, 17u + input_channels);
    fill_tensor(input_nhwc, nchw_input_count, 31u + output_channels);
    fill_tensor(bias, output_channels, 47u + height);
    pack_nhwc_weights(weights, input_channels, output_channels, packed_nhwc);
    lw_pack_conv1x1_weights_f32(weights, input_channels, output_channels, packed_nchw);
    nhwc_to_nchw(input_nhwc, input_nchw, pixels, input_channels);

    lw_scalar_packed_conv1x1_f32(input_nchw, packed_nchw, bias, expected, input_dimensions,
                                 output_dimensions);
    nhwc_conv1x1_scalar(input_nhwc, packed_nhwc, bias, actual, pixels, input_channels,
                        output_channels);
    scalar_difference = max_nchw_nhwc_difference(expected, actual, pixels, output_channels);

    nhwc_conv1x1_selected(input_nhwc, packed_nhwc, bias, actual, pixels, input_channels,
                          output_channels);
    selected_difference = max_nchw_nhwc_difference(expected, actual, pixels, output_channels);
    printf("{\"case\":\"parity\",\"input_channels\":%u,\"output_channels\":%u,"
           "\"height\":%u,\"width\":%u,\"scalar_max_abs_difference\":%.9g,"
           "\"selected_max_abs_difference\":%.9g}\n",
           input_channels, output_channels, height, width, scalar_difference, selected_difference);
    success = scalar_difference <= 1.0e-6f && selected_difference <= 2.0e-4f;

cleanup:
    free(weights);
    free(packed_nhwc);
    free(packed_nchw);
    free(bias);
    free(input_nhwc);
    free(input_nchw);
    free(expected);
    free(actual);
    return success;
}

static double elapsed_milliseconds(clock_t begin, clock_t end) {
    return 1000.0 * (double)(end - begin) / (double)CLOCKS_PER_SEC;
}

static int run_benchmark(void) {
    const uint32_t input_channels = 96u;
    const uint32_t output_channels = 192u;
    const uint32_t pixels = 12u * 32u;
    const uint64_t input_count = (uint64_t)pixels * input_channels;
    const uint64_t output_count = (uint64_t)pixels * output_channels;
    const uint32_t padded_outputs = round_up_output_channels(output_channels);
    float* input = (float*)malloc((size_t)input_count * sizeof(float));
    float* packed_weights =
        (float*)malloc((size_t)input_channels * padded_outputs * sizeof(float));
    float* bias = (float*)malloc((size_t)output_channels * sizeof(float));
    float* output = (float*)malloc((size_t)output_count * sizeof(float));
    clock_t begin;
    clock_t end;
    double nhwc_milliseconds;
    double nchw_milliseconds;
    volatile float checksum = 0.0f;
    uint32_t iteration;
    int32_t input_dimensions[4] = {1, (int32_t)input_channels, 12, 32};
    int32_t output_dimensions[4] = {1, (int32_t)output_channels, 12, 32};
    uint64_t nchw_weight_count = 0u;
    float* nchw_input = NULL;
    float* nchw_weights = NULL;

    if (input == NULL || packed_weights == NULL || bias == NULL || output == NULL ||
        !lw_packed_conv1x1_weight_count(input_channels, output_channels, &nchw_weight_count)) {
        free(input);
        free(packed_weights);
        free(bias);
        free(output);
        return 0;
    }
    nchw_input = (float*)malloc((size_t)input_count * sizeof(float));
    nchw_weights = (float*)malloc((size_t)nchw_weight_count * sizeof(float));
    if (nchw_input == NULL || nchw_weights == NULL) {
        free(input);
        free(packed_weights);
        free(bias);
        free(output);
        free(nchw_input);
        free(nchw_weights);
        return 0;
    }
    fill_tensor(input, input_count, 71u);
    fill_tensor(bias, output_channels, 83u);
    {
        float* canonical_weights =
            (float*)malloc((size_t)input_channels * output_channels * sizeof(float));
        if (canonical_weights == NULL) {
            free(input);
            free(packed_weights);
            free(bias);
            free(output);
            free(nchw_input);
            free(nchw_weights);
            return 0;
        }
        fill_tensor(canonical_weights, (uint64_t)input_channels * output_channels, 97u);
        pack_nhwc_weights(canonical_weights, input_channels, output_channels, packed_weights);
        lw_pack_conv1x1_weights_f32(canonical_weights, input_channels, output_channels,
                                    nchw_weights);
        free(canonical_weights);
    }
    nhwc_to_nchw(input, nchw_input, pixels, input_channels);

    for (iteration = 0u; iteration < 10u; ++iteration) {
        nhwc_conv1x1_selected(input, packed_weights, bias, output, pixels, input_channels,
                          output_channels);
        lw_packed_conv1x1_f32(nchw_input, nchw_weights, bias, output, input_dimensions,
                              output_dimensions);
    }
    begin = clock();
    for (iteration = 0u; iteration < 40u; ++iteration) {
        nhwc_conv1x1_selected(input, packed_weights, bias, output, pixels, input_channels,
                          output_channels);
        checksum += output[(size_t)(iteration % pixels) * output_channels];
    }
    end = clock();
    nhwc_milliseconds = elapsed_milliseconds(begin, end);
    begin = clock();
    for (iteration = 0u; iteration < 40u; ++iteration) {
        lw_packed_conv1x1_f32(nchw_input, nchw_weights, bias, output, input_dimensions,
                              output_dimensions);
        checksum += output[(size_t)(iteration % pixels) * output_channels];
    }
    end = clock();
    nchw_milliseconds = elapsed_milliseconds(begin, end);
    printf("{\"case\":\"benchmark\",\"backend\":\"%s\",\"iterations\":%u,"
           "\"nhwc_ms\":%.6f,\"nchw_ms\":%.6f,\"nhwc_over_nchw\":%.6f,"
           "\"checksum\":%.9g}\n",
           lw_simd_level_is_avx2(lw_get_cpu_capabilities().simd) ? "avx2" : "scalar", iteration,
           nhwc_milliseconds, nchw_milliseconds,
           nchw_milliseconds == 0.0 ? 0.0 : nhwc_milliseconds / nchw_milliseconds, checksum);
    free(input);
    free(packed_weights);
    free(bias);
    free(output);
    free(nchw_input);
    free(nchw_weights);
    return 1;
}

int main(void) {
    if (!run_parity_case(37u, 29u, 5u, 13u)) {
        return 1;
    }
    return run_benchmark() ? 0 : 1;
}
