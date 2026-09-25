#include "x64_rec_backend_internal.h"
#include "model_internal.h"
#include "lw_infer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static uint64_t clock_ns(void) {
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 /
                      (double)frequency.QuadPart);
}
#else
#include <time.h>
static uint64_t clock_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}
#endif

static void fill_input(float* input, uint64_t count) {
    for (uint64_t index = 0u; index < count; ++index) {
        input[index] = (float)((int)(index % 37u) - 18) * 0.0078125f;
    }
}

static void run_kernel(uint32_t kind, const float* input, const float* weights,
                       const lw_nhwc_epilogue* epilogue, float* output,
                       uint32_t pixels, uint32_t input_channels, uint32_t output_channels) {
    switch (kind) {
    case 0u:
        lw_avx2_fma_nhwc_pointwise_f32(input, weights, epilogue, output, pixels,
                                       input_channels, output_channels);
        break;
    case 1u:
        lw_avx2_fma_nhwc_pointwise_4x16_f32(input, weights, epilogue, output, pixels,
                                             input_channels, output_channels);
        break;
    case 2u:
        lw_avx2_fma_nhwc_pointwise_3x32_f32(input, weights, epilogue, output, pixels,
                                             input_channels, output_channels);
        break;
    default:
        lw_avx2_fma_nhwc_pointwise_2x32_f32(input, weights, epilogue, output, pixels,
                                            input_channels, output_channels);
        break;
    }
}

static double measure(uint32_t kind, const float* input, const float* weights,
                      const lw_nhwc_epilogue* epilogue, float* output,
                      uint32_t pixels, uint32_t input_channels, uint32_t output_channels,
                      uint32_t iterations) {
    for (uint32_t warmup = 0u; warmup < 2u; ++warmup) {
        run_kernel(kind, input, weights, epilogue, output, pixels, input_channels, output_channels);
    }
    uint64_t started = clock_ns();
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        run_kernel(kind, input, weights, epilogue, output, pixels, input_channels, output_channels);
    }
    return (double)(clock_ns() - started) / 1000000.0 / (double)iterations;
}

static float max_difference(const float* expected, const float* actual, uint64_t count) {
    float maximum = 0.0f;
    for (uint64_t index = 0u; index < count; ++index) {
        float difference = fabsf(expected[index] - actual[index]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
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
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance* instance = NULL;
    lw_error error;
    int result = 1;

    if (argc < 2 || argc > 3 || (argc == 3 && !parse_iterations(argv[2], &iterations))) {
        fprintf(stderr, "usage: x64-rec-pointwise-tuning-driver rec.lwm [iterations]\n");
        return 2;
    }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK ||
        lw_x64_rec_backend_compile(model, width, &program, &error) != LW_X64_REC_COMPILE_OK ||
        program == NULL ||
        lw_x64_rec_instance_create(program, &instance, &error) != LW_STATUS_OK) {
        fprintf(stderr, "setup failed: %s\n", error.message);
        goto cleanup;
    }

    printf("{\"schema_version\":1,\"width\":%u,\"iterations\":%u,\"cases\":[\n", width, iterations);
    int first = 1;
    for (uint32_t op_index = 0u; op_index < program->op_count; ++op_index) {
        const lw_x64_rec_op* op = &program->ops[op_index];
        if (op->kind != LW_X64_REC_OP_POINTWISE) continue;
        uint32_t pixels = op->data.conv.input_height * op->data.conv.input_width;
        uint32_t input_channels = op->data.conv.input_channels;
        uint32_t output_channels = op->data.conv.output_channels;
        uint64_t input_count = (uint64_t)pixels * input_channels;
        uint64_t output_count = (uint64_t)pixels * output_channels;
        float* input = (float*)(void*)(instance->arena + (size_t)op->data.conv.input_offset);
        float* baseline = (float*)malloc((size_t)output_count * sizeof(float));
        float* candidate = (float*)malloc((size_t)output_count * sizeof(float));
        double timings[4] = {-1.0, -1.0, -1.0, -1.0};
        float errors[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const char* names[4] = {"6x16", "4x16", "3x32", "2x32"};
        uint32_t best = 0u;
        if (input == NULL || baseline == NULL || candidate == NULL) {
            free(candidate);
            free(baseline);
            fprintf(stderr, "allocation failed for op %u\n", op_index);
            goto cleanup;
        }
        fill_input(input, input_count);
        lw_nhwc_epilogue epilogue = {
            op->data.conv.bias, NULL, op->data.conv.activation, 0u, 0.0f, 0.0f, NULL
        };
        timings[0] = measure(0u, input, op->data.conv.packed_weights, &epilogue, baseline,
                             pixels, input_channels, output_channels, iterations);
        for (uint32_t kind = 1u; kind < 4u; ++kind) {
            if ((kind == 1u && output_channels % 16u != 0u) ||
                (kind >= 2u && output_channels % 32u != 0u)) continue;
            memset(candidate, 0, (size_t)output_count * sizeof(float));
            timings[kind] = measure(kind, input, op->data.conv.packed_weights, &epilogue, candidate,
                                    pixels, input_channels, output_channels, iterations);
            errors[kind] = max_difference(baseline, candidate, output_count);
            if (errors[kind] > 1.0e-5f) {
                fprintf(stderr, "pointwise candidate mismatch op=%u kind=%s error=%.9g\n",
                        op_index, names[kind], errors[kind]);
                free(candidate);
                free(baseline);
                goto cleanup;
            }
        }
        for (uint32_t kind = 1u; kind < 4u; ++kind) {
            if (timings[kind] > 0.0 && timings[kind] < timings[best]) best = kind;
        }
        if (!first) printf(",\n");
        first = 0;
        printf("{\"op\":%u,\"input_channels\":%u,\"output_channels\":%u,\"pixels\":%u,"
               "\"6x16_ms\":%.6f,\"4x16_ms\":%.6f,\"3x32_ms\":%.6f,\"2x32_ms\":%.6f,"
               "\"winner\":\"%s\",\"max_error\":%.9g}",
               op_index, input_channels, output_channels, pixels,
               timings[0], timings[1], timings[2], timings[3], names[best], errors[best]);
        free(candidate);
        free(baseline);
    }
    printf("]}\n");
    result = 0;
cleanup:
    lw_x64_rec_instance_free(instance);
    lw_x64_rec_program_free(program);
    lw_model_free(model);
    return result;
}
