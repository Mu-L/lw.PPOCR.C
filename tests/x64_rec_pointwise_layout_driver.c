#include "x64_rec_backend_internal.h"
#include "model_internal.h"
#include "simd_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static uint64_t clock_ns(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (uint64_t)((double)c.QuadPart * 1000000000.0 / (double)f.QuadPart);
}
#else
#include <time.h>
static uint64_t clock_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
#endif
static void fill_input(float* input, uint64_t count) {
    for (uint64_t i = 0u; i < count; ++i) input[i] = (float)((int)(i % 37u) - 18) * 0.0078125f;
}
static float max_difference(const float* a, const float* b, uint64_t count) {
    float max_value = 0.0f;
    for (uint64_t i = 0u; i < count; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_value) max_value = d;
    }
    return max_value;
}
static double measure_nhwc(const lw_x64_rec_op* op, const float* input, float* output,
                           uint32_t pixels, uint32_t ic, uint32_t oc, uint32_t iterations) {
    lw_nhwc_epilogue ep = { op->data.conv.bias, NULL, LW_NHWC_ACT_NONE, 0u, 0.0f, 0.0f, NULL };
    for (uint32_t i = 0u; i < 2u; ++i)
        lw_avx2_fma_nhwc_pointwise_f32(input, op->data.conv.packed_weights, &ep, output, pixels, ic, oc);
    uint64_t start = clock_ns();
    for (uint32_t i = 0u; i < iterations; ++i)
        lw_avx2_fma_nhwc_pointwise_f32(input, op->data.conv.packed_weights, &ep, output, pixels, ic, oc);
    return (double)(clock_ns() - start) / 1000000.0 / (double)iterations;
}
static double measure_nchw(const lw_x64_rec_op* op, uint8_t fma8, const float* input, float* output,
                           const int32_t input_dimensions[4], const int32_t output_dimensions[4],
                           uint32_t iterations) {
    for (uint32_t i = 0u; i < 2u; ++i) {
        if (fma8) lw_avx2_fma_packed_conv1x1_8x8_f32(input, op->data.conv.packed_weights, op->data.conv.bias, output, input_dimensions, output_dimensions);
        else lw_avx2_fma_packed_conv1x1_f32(input, op->data.conv.packed_weights, op->data.conv.bias, output, input_dimensions, output_dimensions);
    }
    uint64_t start = clock_ns();
    for (uint32_t i = 0u; i < iterations; ++i) {
        if (fma8) lw_avx2_fma_packed_conv1x1_8x8_f32(input, op->data.conv.packed_weights, op->data.conv.bias, output, input_dimensions, output_dimensions);
        else lw_avx2_fma_packed_conv1x1_f32(input, op->data.conv.packed_weights, op->data.conv.bias, output, input_dimensions, output_dimensions);
    }
    return (double)(clock_ns() - start) / 1000000.0 / (double)iterations;
}
static int parse_iterations(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL) return 0;
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0u || parsed > 1000u) return 0;
    *value = (uint32_t)parsed;
    return 1;
}
int main(int argc, char** argv) {
    const uint32_t width = 960u;
    uint32_t iterations = 5u;
    lw_model* model = NULL;
    lw_x64_rec_program* nhwc_program = NULL;
    lw_x64_rec_program* nchw_program = NULL;
    lw_x64_rec_instance* nhwc_instance = NULL;
    lw_x64_rec_instance* nchw_instance = NULL;
    lw_error error;
    int result = 1;
    if (argc < 2 || argc > 3 || (argc == 3 && !parse_iterations(argv[2], &iterations))) {
        fprintf(stderr, "usage: x64-rec-pointwise-layout-driver rec.lwm [iterations]\n");
        return 2;
    }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK ||
        lw_x64_rec_backend_compile_ex(model, width, LW_X64_REC_COMPILE_NHWC, &nhwc_program, &error) != LW_X64_REC_COMPILE_OK ||
        lw_x64_rec_backend_compile_ex(model, width, LW_X64_REC_COMPILE_NCHW, &nchw_program, &error) != LW_X64_REC_COMPILE_OK ||
        lw_x64_rec_instance_create(nhwc_program, &nhwc_instance, &error) != LW_STATUS_OK ||
        lw_x64_rec_instance_create(nchw_program, &nchw_instance, &error) != LW_STATUS_OK) {
        fprintf(stderr, "setup failed: %s\n", error.message);
        goto cleanup;
    }
    printf("{\"schema_version\":1,\"width\":%u,\"iterations\":%u,\"cases\":[\n", width, iterations);
    int first = 1;
    for (uint32_t oi = 0u; oi < nchw_program->op_count; ++oi) {
        const lw_x64_rec_op* nchw_op = &nchw_program->ops[oi];
        if (nchw_op->kind != LW_X64_REC_OP_POINTWISE_NCHW) continue;
        const lw_x64_rec_op* nhwc_op = NULL;
        for (uint32_t candidate = 0u; candidate < nhwc_program->op_count; ++candidate) {
            if (nhwc_program->ops[candidate].semantic_begin == nchw_op->semantic_begin &&
                nhwc_program->ops[candidate].kind == LW_X64_REC_OP_POINTWISE) {
                nhwc_op = &nhwc_program->ops[candidate];
                break;
            }
        }
        if (nhwc_op == NULL) continue;
        uint32_t h = nchw_op->data.conv.input_height;
        uint32_t w = nchw_op->data.conv.input_width;
        uint32_t ic = nchw_op->data.conv.input_channels;
        uint32_t oc = nchw_op->data.conv.output_channels;
        uint32_t pixels = h * w;
        uint64_t count = (uint64_t)ic * pixels;
        uint64_t out_count = (uint64_t)oc * pixels;
        float* nhwc_input = (float*)malloc((size_t)count * sizeof(float));
        float* nhwc_baseline = (float*)malloc((size_t)out_count * sizeof(float));
        float* nchw_input = (float*)malloc((size_t)count * sizeof(float));
        float* nchw_fma4 = (float*)malloc((size_t)out_count * sizeof(float));
        float* nchw_fma8 = (float*)malloc((size_t)out_count * sizeof(float));
        if (!nhwc_input || !nhwc_baseline || !nchw_input || !nchw_fma4 || !nchw_fma8) {
            free(nhwc_input); free(nhwc_baseline); free(nchw_input); free(nchw_fma4); free(nchw_fma8);
            fprintf(stderr, "allocation failed for semantic %u\n", nchw_op->semantic_begin);
            goto cleanup;
        }
        fill_input(nhwc_input, count);
        fill_input(nchw_input, count);
        lw_nhwc_epilogue ep = { nhwc_op->data.conv.bias, NULL, LW_NHWC_ACT_NONE, 0u, 0.0f, 0.0f, NULL };
        lw_avx2_fma_nhwc_pointwise_f32(nhwc_input, nhwc_op->data.conv.packed_weights, &ep, nhwc_baseline, pixels, ic, oc);
        int32_t input_dimensions[4] = { 1, (int32_t)ic, (int32_t)h, (int32_t)w };
        int32_t output_dimensions[4] = { 1, (int32_t)oc, (int32_t)h, (int32_t)w };
        lw_avx2_fma_packed_conv1x1_f32(nchw_input, nchw_op->data.conv.packed_weights, nchw_op->data.conv.bias, nchw_fma4, input_dimensions, output_dimensions);
        double nhwc_ms = measure_nhwc(nhwc_op, nhwc_input, nhwc_baseline, pixels, ic, oc, iterations);
        double nchw_fma4_ms = measure_nchw(nchw_op, 0u, nchw_input, nchw_fma4, input_dimensions, output_dimensions, iterations);
        double nchw_fma8_ms = -1.0;
        float fma8_error = 0.0f;
        if (oc >= 8u && (oc & 7u) == 0u && pixels >= 8u) {
            nchw_fma8_ms = measure_nchw(nchw_op, 1u, nchw_input, nchw_fma8, input_dimensions, output_dimensions, iterations);
            fma8_error = max_difference(nchw_fma4, nchw_fma8, out_count);
        }
        if (nchw_fma8_ms >= 0.0 && fma8_error > 1.0e-5f) {
            fprintf(stderr, "FMA8 mismatch for semantic %u: max_abs=%.9g\n", nchw_op->semantic_begin, fma8_error);
            free(nhwc_input); free(nhwc_baseline); free(nchw_input); free(nchw_fma4); free(nchw_fma8);
            goto cleanup;
        }
        const char* winner = "nhwc";
        double best = nhwc_ms;
        if (nchw_fma4_ms < best) { best = nchw_fma4_ms; winner = "nchw_fma4"; }
        if (nchw_fma8_ms >= 0.0 && nchw_fma8_ms < best) winner = "nchw_fma8";
        if (!first) printf(",\n");
        first = 0;
        printf("{\"semantic\":%u,\"input_channels\":%u,\"output_channels\":%u,\"height\":%u,\"width\":%u,\"nhwc_ms\":%.6f,\"nchw_fma4_ms\":%.6f,\"nchw_fma8_ms\":%.6f,\"winner\":\"%s\",\"nchw_fma8_error\":%.9g}",
               nchw_op->semantic_begin, ic, oc, h, w, nhwc_ms, nchw_fma4_ms, nchw_fma8_ms, winner, fma8_error);
        free(nhwc_input); free(nhwc_baseline); free(nchw_input); free(nchw_fma4); free(nchw_fma8);
    }
    printf("]}\n");
    result = 0;
cleanup:
    lw_x64_rec_instance_free(nchw_instance);
    lw_x64_rec_instance_free(nhwc_instance);
    lw_x64_rec_program_free(nchw_program);
    lw_x64_rec_program_free(nhwc_program);
    lw_model_free(model);
    return result;
}
