#include "cpu_features.h"
#include "scalar_kernels.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

typedef void (*kernel_fn)(const float*, const float*, const float*, float*,
                          const int32_t[4], const int32_t[4]);

static double now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart == 0) return 0.0;
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL || text[0] == '\0' || text[0] == '-') return 0;
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0ul || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static void fill_values(float* values, uint64_t count, uint32_t seed) {
    uint64_t i;
    uint32_t state = seed;
    for (i = 0u; i < count; ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        values[(size_t)i] = (float)((int32_t)(state >> 9u) % 1021) / 511.0f;
    }
}

static uint64_t checksum(const float* values, uint64_t count) {
    const unsigned char* bytes = (const unsigned char*)values;
    uint64_t hash = UINT64_C(1469598103934665603);
    uint64_t i;
    for (i = 0u; i < count * sizeof(float); ++i) {
        hash ^= bytes[(size_t)i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static float max_error(const float* left, const float* right, uint64_t count) {
    uint64_t i;
    float result = 0.0f;
    for (i = 0u; i < count; ++i) {
        float value = fabsf(left[(size_t)i] - right[(size_t)i]);
        if (!isfinite(value)) return INFINITY;
        if (value > result) result = value;
    }
    return result;
}

static int measure(kernel_fn kernel, const float* input, const float* weights,
                   const float* bias, float* output, const int32_t input_dims[4],
                   const int32_t output_dims[4], uint32_t iterations, double* result) {
    uint32_t i;
    double start;
    double finish;
    if (kernel == NULL || result == NULL || iterations == 0u) return 0;
    kernel(input, weights, bias, output, input_dims, output_dims);
    start = now_seconds();
    for (i = 0u; i < iterations; ++i)
        kernel(input, weights, bias, output, input_dims, output_dims);
    finish = now_seconds();
    if (start <= 0.0 || finish <= start) return 0;
    *result = (finish - start) * 1000.0 / (double)iterations;
    return isfinite(*result) && *result > 0.0;
}

static kernel_fn select_kernel(lw_simd_level level, const char** name) {
    if (lw_simd_level_is_avx2(level)) {
        *name = "avx2";
        return lw_avx2_conv_transpose2x2_stride2_f32;
    }
    if (lw_simd_level_is_sse2(level)) {
        *name = "sse2";
        return lw_sse2_conv_transpose2x2_stride2_f32;
    }
    if (lw_simd_level_is_neon(level)) {
        *name = "neon";
        return lw_neon_conv_transpose2x2_stride2_f32;
    }
    *name = "scalar";
    return lw_scalar_conv_transpose2x2_stride2_f32;
}

int main(int argc, char** argv) {
    uint32_t ic = 16u, oc = 16u, height = 128u, width = 128u, iterations = 10u;
    uint64_t input_count, weight_count, output_count;
    size_t input_bytes, weight_bytes, output_bytes;
    float* input = NULL;
    float* weights = NULL;
    float* bias = NULL;
    float* scalar_output = NULL;
    float* simd_output = NULL;
    int32_t input_dims[4], output_dims[4];
    const char* backend = NULL;
    kernel_fn simd_kernel;
    double scalar_ms, simd_ms;
    float error;
    int exit_code = 1;

    if (argc > 1 && !parse_u32(argv[1], &ic)) return 2;
    if (argc > 2 && !parse_u32(argv[2], &oc)) return 2;
    if (argc > 3 && !parse_u32(argv[3], &height)) return 2;
    if (argc > 4 && !parse_u32(argv[4], &width)) return 2;
    if (argc > 5 && !parse_u32(argv[5], &iterations)) return 2;
    if (argc > 6 || height > (uint32_t)INT32_MAX / 2u ||
        width > (uint32_t)INT32_MAX / 2u) {
        fprintf(stderr, "usage: convtranspose-benchmark-driver [ic] [oc] [height] [width] [iterations]\n");
        return 2;
    }

    input_count = (uint64_t)ic * height * width;
    weight_count = (uint64_t)ic * oc * 4u;
    output_count = (uint64_t)oc * height * 2u * width * 2u;
    if (input_count > SIZE_MAX / sizeof(float) ||
        weight_count > SIZE_MAX / sizeof(float) ||
        output_count > SIZE_MAX / sizeof(float)) return 2;
    input_bytes = (size_t)input_count * sizeof(float);
    weight_bytes = (size_t)weight_count * sizeof(float);
    output_bytes = (size_t)output_count * sizeof(float);
    input = (float*)malloc(input_bytes);
    weights = (float*)malloc(weight_bytes);
    bias = (float*)malloc((size_t)oc * sizeof(float));
    scalar_output = (float*)malloc(output_bytes);
    simd_output = (float*)malloc(output_bytes);
    if (input == NULL || weights == NULL || bias == NULL ||
        scalar_output == NULL || simd_output == NULL) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    fill_values(input, input_count, 17u);
    fill_values(weights, weight_count, 31u);
    fill_values(bias, oc, 47u);
    input_dims[0] = 1; input_dims[1] = (int32_t)ic;
    input_dims[2] = (int32_t)height; input_dims[3] = (int32_t)width;
    output_dims[0] = 1; output_dims[1] = (int32_t)oc;
    output_dims[2] = (int32_t)(height * 2u); output_dims[3] = (int32_t)(width * 2u);

    if (!measure(lw_scalar_conv_transpose2x2_stride2_f32, input, weights, bias,
                 scalar_output, input_dims, output_dims, iterations, &scalar_ms)) {
        fprintf(stderr, "scalar benchmark failed\n");
        goto cleanup;
    }
    simd_kernel = select_kernel(lw_detect_simd_level(), &backend);
    if (!measure(simd_kernel, input, weights, bias, simd_output, input_dims,
                 output_dims, iterations, &simd_ms)) {
        fprintf(stderr, "SIMD benchmark failed\n");
        goto cleanup;
    }
    error = max_error(scalar_output, simd_output, output_count);
    if (!isfinite(error) || error > 1.0e-3f) {
        fprintf(stderr, "SIMD parity failed: backend=%s max_abs_error=%g\n", backend, error);
        goto cleanup;
    }
    printf("{\"schema_version\":1,\"input_channels\":%u,\"output_channels\":%u,"
           "\"input_height\":%u,\"input_width\":%u,\"output_height\":%u,"
           "\"output_width\":%u,\"iterations\":%u,\"simd_backend\":\"%s\","
           "\"scalar_ms\":%.9f,\"simd_ms\":%.9f,\"speedup\":%.9f,"
           "\"max_abs_error\":%.9g,\"scalar_checksum\":\"0x%016" PRIx64
           "\",\"simd_checksum\":\"0x%016" PRIx64 "\"}\n",
           ic, oc, height, width, height * 2u, width * 2u, iterations, backend,
           scalar_ms, simd_ms, scalar_ms / simd_ms, error,
           checksum(scalar_output, output_count), checksum(simd_output, output_count));
    exit_code = 0;
cleanup:
    free(input); free(weights); free(bias); free(scalar_output); free(simd_output);
    return exit_code;
}
