#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "executor_internal.h"
#include "model_internal.h"
#include "rec_internal.h"
#include "session_internal.h"
#include "lw_infer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/mach_time.h>
#else
#  include <time.h>
#endif

#define DEFAULT_ITERATIONS 30u
#define WARMUP_ROUNDS 3u

static uint64_t clock_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency) ||
        frequency.QuadPart <= 0) {
        return 0u;
    }
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 /
                      (double)frequency.QuadPart);
#elif defined(__APPLE__)
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0u && mach_timebase_info(&timebase) != KERN_SUCCESS) {
        return 0u;
    }
    return (uint64_t)((double)mach_absolute_time() * (double)timebase.numer /
                      (double)timebase.denom);
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static int parse_iterations(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL) return 0;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u || parsed > 1000u) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static double median(double* values, uint32_t count) {
    uint32_t i;
    uint32_t j;
    if (count == 0u) return 0.0;
    for (i = 1u; i < count; ++i) {
        double current = values[i];
        j = i;
        while (j > 0u && values[j - 1u] > current) {
            values[j] = values[j - 1u];
            --j;
        }
        values[j] = current;
    }
    if ((count & 1u) != 0u) return values[count / 2u];
    return (values[count / 2u - 1u] + values[count / 2u]) * 0.5;
}

static void fill_source(uint8_t* source, uint32_t width, uint32_t height) {
    uint64_t count = (uint64_t)width * height * 3u;
    uint64_t i;
    for (i = 0u; i < count; ++i) {
        source[i] = (uint8_t)((i * 37u + i / (uint64_t)(width * 3u) * 13u + 17u) & 0xffu);
    }
}

static int compare_ctc(const uint32_t* canonical_indices, const float* canonical_probabilities,
                       const uint32_t* backend_indices, const float* backend_probabilities,
                       uint32_t rows, float* max_probability_difference) {
    uint32_t row;
    float maximum = 0.0f;
    for (row = 0u; row < rows; ++row) {
        float difference = fabsf(canonical_probabilities[row] - backend_probabilities[row]);
        if (difference > maximum) maximum = difference;
        if (canonical_indices[row] != backend_indices[row] || difference > 1.0e-6f) {
            fprintf(stderr, "backend mismatch at row=%u canonical=(%u,%.9g) backend=(%u,%.9g)\n",
                    row, canonical_indices[row], canonical_probabilities[row],
                    backend_indices[row], backend_probabilities[row]);
            if (max_probability_difference != NULL) *max_probability_difference = maximum;
            return 0;
        }
    }
    if (max_probability_difference != NULL) *max_probability_difference = maximum;
    return 1;
}

static int run_canonical(lw_session* session, const uint8_t* source, uint64_t source_bytes,
                         float* input, uint64_t input_count, uint32_t width, uint32_t height,
                         uint32_t* indices, float* probabilities, uint32_t rows,
                         uint32_t classes, double* preprocess_ms, double* graph_ms) {
    uint32_t resized_width = 0u;
    lw_error error;
    lw_status status;
    uint64_t start = clock_ns();
    lw_error_init(&error);
    status = lw_rec_preprocess_bgr_u8(source, source_bytes, width, height, width * 3u, 960u,
                                      input, input_count, &resized_width);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "canonical preprocess failed: %s\n", error.message);
        return 0;
    }
    if (preprocess_ms != NULL) {
        *preprocess_ms = (double)(clock_ns() - start) / 1000000.0;
    }
    start = clock_ns();
    status = lw_execute_session_f32_ctc_greedy(session, input, input_count, indices, probabilities,
                                               rows, classes, NULL, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "canonical execution failed: %s\n", error.message);
        return 0;
    }
    if (graph_ms != NULL) *graph_ms = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}

static uint32_t profile_slot(uint16_t kind) {
    switch (kind) {
    case LW_X64_REC_OP_POINTWISE: return 0u;
    case LW_X64_REC_OP_DENSE: return 1u;
    case LW_X64_REC_OP_DEPTHWISE: return 2u;
    case LW_X64_REC_OP_AFFINE: return 3u;
    case LW_X64_REC_OP_ADD:
    case LW_X64_REC_OP_MUL:
    case LW_X64_REC_OP_DIV: return 4u;
    case LW_X64_REC_OP_RELU:
    case LW_X64_REC_OP_ERF:
    case LW_X64_REC_OP_GELU:
    case LW_X64_REC_OP_HARD_SIGMOID: return 5u;
    case LW_X64_REC_OP_REDUCE_MEAN: return 6u;
    case LW_X64_REC_OP_AVG_POOL:
    case LW_X64_REC_OP_MAX_POOL: return 7u;
    case LW_X64_REC_OP_TRANSPOSE: return 8u;
    case LW_X64_REC_OP_MATMUL: return 9u;
    default: return 10u;
    }
}
static int run_backend_unprofiled(lw_x64_rec_program* program, lw_x64_rec_instance* instance,
                      const uint8_t* source, uint64_t source_bytes, float* input,
                      uint64_t input_count, uint32_t width, uint32_t height,
                      double* preprocess_ms, double* backbone_ms, double* ctc_ms) {
    uint32_t resized_width = 0u;
    lw_error error;
    lw_status status;
    uint64_t start = clock_ns();
    lw_error_init(&error);
    status = lw_rec_preprocess_bgr_u8_nhwc(source, source_bytes, width, height, width * 3u, 960u,
                                           input, input_count, &resized_width);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "backend preprocess failed: %s\n", error.message);
        return 0;
    }
    if (preprocess_ms != NULL) *preprocess_ms = (double)(clock_ns() - start) / 1000000.0;
    start = clock_ns();
    status = lw_x64_rec_instance_run_backbone(instance, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "backend execution failed: %s\n", error.message);
        return 0;
    }
    if (backbone_ms != NULL) *backbone_ms = (double)(clock_ns() - start) / 1000000.0;
    start = clock_ns();
    {
        float* activation = (float*)(void*)(instance->arena +
            (size_t)program->values[program->ctc.activation_value].offset);
        status = lw_x64_rec_ctc_execute(program, instance, activation, &error);
    }
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "backend CTC failed: %s\n", error.message);
        return 0;
    }
    if (ctc_ms != NULL) *ctc_ms = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}

static int run_backend_profiled(lw_x64_rec_program* program, lw_x64_rec_instance* instance,
                      const uint8_t* source, uint64_t source_bytes, float* input,
                      uint64_t input_count, uint32_t width, uint32_t height,
                      double* preprocess_ms, double* backbone_ms, double* ctc_ms,
                      double* category_totals_ms, double* profile_ctc_total_ms) {
    uint32_t resized_width = 0u;
    lw_error error;
    lw_status status;
    uint64_t start = clock_ns();
    lw_error_init(&error);
    status = lw_rec_preprocess_bgr_u8_nhwc(source, source_bytes, width, height, width * 3u, 960u,
                                           input, input_count, &resized_width);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "backend preprocess failed: %s\n", error.message);
        return 0;
    }
    if (preprocess_ms != NULL) *preprocess_ms = (double)(clock_ns() - start) / 1000000.0;
    start = clock_ns();
    for (uint32_t op_index = 0u; op_index < program->op_count; ++op_index) {
        uint64_t op_start = clock_ns();
        status = lw_x64_rec_instance_run_op(instance, op_index, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "backend op %u failed: %s\n", op_index, error.message);
            return 0;
        }
        if (category_totals_ms != NULL) {
            category_totals_ms[profile_slot(program->ops[op_index].kind)] +=
                (double)(clock_ns() - op_start) / 1000000.0;
        }
    }
    if (backbone_ms != NULL) *backbone_ms = (double)(clock_ns() - start) / 1000000.0;
    start = clock_ns();
    {
        float* activation = (float*)(void*)(instance->arena +
            (size_t)program->values[program->ctc.activation_value].offset);
        status = lw_x64_rec_ctc_execute(program, instance, activation, &error);
    }
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "backend CTC failed: %s\n", error.message);
        return 0;
    }
    {
        uint64_t elapsed = clock_ns() - start;
        if (ctc_ms != NULL) *ctc_ms = (double)elapsed / 1000000.0;
        if (profile_ctc_total_ms != NULL) *profile_ctc_total_ms += (double)elapsed / 1000000.0;
    }
    return 1;
}
static void count_conv_paths(const lw_x64_rec_program* program,
                             uint32_t* pointwise_ops, uint32_t* dense_ops,
                             uint32_t* depthwise_ops, uint32_t* pointwise_fallbacks,
                             uint32_t* dense_fallbacks, uint32_t* depthwise_fallbacks) {
    uint32_t index;
    uint32_t pointwise = 0u, dense = 0u, depthwise = 0u;
    uint32_t pointwise_fallback = 0u, dense_fallback = 0u, depthwise_fallback = 0u;
    if (program != NULL) {
        for (index = 0u; index < program->op_count; ++index) {
            const lw_x64_rec_op* op = &program->ops[index];
            if (op->kind == LW_X64_REC_OP_POINTWISE) {
                ++pointwise;
                if (op->data.conv.scalar_fallback ||
                    (op->data.conv.activation != LW_NHWC_ACT_NONE &&
                     op->data.conv.activation != LW_NHWC_ACT_RELU &&
                     op->data.conv.activation != LW_NHWC_ACT_HARDSWISH)) ++pointwise_fallback;
            } else if (op->kind == LW_X64_REC_OP_DENSE) {
                ++dense;
                if (op->data.conv.scalar_fallback) ++dense_fallback;
            } else if (op->kind == LW_X64_REC_OP_DEPTHWISE) {
                ++depthwise;
                if ((op->data.conv.input_channels & 7u) != 0u) ++depthwise_fallback;
            }
        }
    }
    if (pointwise_ops != NULL) *pointwise_ops = pointwise;
    if (dense_ops != NULL) *dense_ops = dense;
    if (depthwise_ops != NULL) *depthwise_ops = depthwise;
    if (pointwise_fallbacks != NULL) *pointwise_fallbacks = pointwise_fallback;
    if (dense_fallbacks != NULL) *dense_fallbacks = dense_fallback;
    if (depthwise_fallbacks != NULL) *depthwise_fallbacks = depthwise_fallback;
}

int main(int argc, char** argv) {
    const uint32_t width = 960u;
    const uint32_t height = 48u;
    uint32_t iterations = DEFAULT_ITERATIONS;
    uint64_t source_bytes = (uint64_t)width * height * 3u;
    uint64_t input_count = (uint64_t)width * height * 3u;
    lw_model* model = NULL;
    lw_session* canonical = NULL;
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance* instance = NULL;
    lw_tensor_desc input_desc;
    lw_error error;
    uint8_t* source = NULL;
    float* canonical_input = NULL;
    uint32_t* canonical_indices = NULL;
    float* canonical_probabilities = NULL;
    float* backend_input = NULL;
    double* canonical_samples = NULL;
    double* backend_samples = NULL;
    double* canonical_preprocess_samples = NULL;
    double* backend_preprocess_samples = NULL;
    double* backend_backbone_samples = NULL;
    double* backend_ctc_samples = NULL;
    double category_totals_ms[11] = {0.0};
    double profile_ctc_total_ms = 0.0;
    uint32_t profile_count = 0u;
    uint32_t pointwise_ops = 0u;
    uint32_t dense_ops = 0u;
    uint32_t depthwise_ops = 0u;
    uint32_t pointwise_fallbacks = 0u;
    uint32_t dense_fallbacks = 0u;
    uint32_t depthwise_fallbacks = 0u;
    uint32_t sample_count = 0u;
    uint32_t warmup;
    uint32_t iteration;
    int result = 1;

    if (argc > 3 || (argc == 3 && !parse_iterations(argv[2], &iterations))) {
        fprintf(stderr, "usage: x64-rec-backend-benchmark-driver rec.lwm [iterations]\n");
        return 2;
    }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
    if (lw_session_create_ctc_greedy(model, &input_desc, 1u, NULL, &canonical, &error) != LW_STATUS_OK) {
        fprintf(stderr, "canonical session create failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_rec_backend_compile(model, width, &program, &error) != LW_X64_REC_COMPILE_OK ||
        program == NULL || program->unsupported_nodes != 0u) {
        fprintf(stderr, "backend compile failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_rec_instance_create(program, &instance, &error) != LW_STATUS_OK || instance == NULL) {
        fprintf(stderr, "backend instance create failed: %s\n", error.message);
        goto cleanup;
    }
    backend_input = lw_x64_rec_instance_input(instance, &input_count);
    source = (uint8_t*)malloc((size_t)source_bytes);
    canonical_input = (float*)malloc((size_t)input_count * sizeof(float));
    canonical_indices = (uint32_t*)malloc((size_t)program->time_steps * sizeof(uint32_t));
    canonical_probabilities = (float*)malloc((size_t)program->time_steps * sizeof(float));
    canonical_samples = (double*)malloc((size_t)iterations * sizeof(double));
    backend_samples = (double*)malloc((size_t)iterations * sizeof(double));
    canonical_preprocess_samples = (double*)malloc((size_t)iterations * sizeof(double));
    backend_preprocess_samples = (double*)malloc((size_t)iterations * sizeof(double));
    backend_backbone_samples = (double*)malloc((size_t)iterations * sizeof(double));
    backend_ctc_samples = (double*)malloc((size_t)iterations * sizeof(double));
    if (backend_input == NULL || source == NULL || canonical_input == NULL ||
        canonical_indices == NULL || canonical_probabilities == NULL ||
        canonical_samples == NULL || backend_samples == NULL ||
        canonical_preprocess_samples == NULL || backend_preprocess_samples == NULL ||
        backend_backbone_samples == NULL || backend_ctc_samples == NULL) {
        fprintf(stderr, "benchmark allocation failed\n");
        goto cleanup;
    }
    fill_source(source, width, height);
    memset(backend_input, 0, (size_t)input_count * sizeof(float));
    for (warmup = 0u; warmup < WARMUP_ROUNDS; ++warmup) {
        double ignored;
        if (!run_canonical(canonical, source, source_bytes, canonical_input, input_count, width,
                           height, canonical_indices, canonical_probabilities, program->time_steps,
                           program->class_count, &ignored, &ignored) ||
            !run_backend_unprofiled(program, instance, source, source_bytes, backend_input, input_count,
                         width, height, &ignored, &ignored, &ignored) ||
            !compare_ctc(canonical_indices, canonical_probabilities, instance->best_indices,
                         instance->best_probabilities, program->time_steps, NULL)) {
            goto cleanup;
        }
    }
    for (iteration = 0u; iteration < iterations; ++iteration) {
        double canonical_preprocess;
        double canonical_graph;
        double backend_preprocess;
        double backend_backbone;
        double backend_ctc;
        int canonical_first = (iteration & 1u) == 0u;
        if (canonical_first) {
            if (!run_canonical(canonical, source, source_bytes, canonical_input, input_count,
                               width, height, canonical_indices, canonical_probabilities,
                               program->time_steps, program->class_count, &canonical_preprocess,
                               &canonical_graph) ||
                !run_backend_unprofiled(program, instance, source, source_bytes, backend_input, input_count,
                             width, height, &backend_preprocess, &backend_backbone, &backend_ctc)) {
                goto cleanup;
            }
        } else {
            if (!run_backend_unprofiled(program, instance, source, source_bytes, backend_input, input_count,
                             width, height, &backend_preprocess, &backend_backbone, &backend_ctc) ||
                !run_canonical(canonical, source, source_bytes, canonical_input, input_count,
                               width, height, canonical_indices, canonical_probabilities,
                               program->time_steps, program->class_count, &canonical_preprocess,
                               &canonical_graph)) {
                goto cleanup;
            }
        }
        if (!compare_ctc(canonical_indices, canonical_probabilities, instance->best_indices,
                         instance->best_probabilities, program->time_steps, NULL)) {
            goto cleanup;
        }
        canonical_samples[sample_count] = canonical_preprocess + canonical_graph;
        backend_samples[sample_count] = backend_preprocess + backend_backbone + backend_ctc;
        canonical_preprocess_samples[sample_count] = canonical_preprocess;
        backend_preprocess_samples[sample_count] = backend_preprocess;
        backend_backbone_samples[sample_count] = backend_backbone;
        backend_ctc_samples[sample_count] = backend_ctc;
        ++sample_count;
    }
    profile_count = iterations < 5u ? iterations : 5u;
    for (iteration = 0u; iteration < profile_count; ++iteration) {
        double ignored;
        if (!run_backend_profiled(program, instance, source, source_bytes, backend_input,
                                  input_count, width, height, &ignored, &ignored, &ignored,
                                  category_totals_ms, &profile_ctc_total_ms) ||
            !compare_ctc(canonical_indices, canonical_probabilities, instance->best_indices,
                         instance->best_probabilities, program->time_steps, NULL)) {
            goto cleanup;
        }
    }
    count_conv_paths(program, &pointwise_ops, &dense_ops, &depthwise_ops,
                     &pointwise_fallbacks, &dense_fallbacks, &depthwise_fallbacks);
    {
        double canonical_median = median(canonical_samples, sample_count);
        double backend_median = median(backend_samples, sample_count);
        double speedup = backend_median > 0.0 ? canonical_median / backend_median : 0.0;
        double canonical_preprocess = median(canonical_preprocess_samples, sample_count);
        double backend_preprocess = median(backend_preprocess_samples, sample_count);
        double backend_backbone = median(backend_backbone_samples, sample_count);
        double backend_ctc = median(backend_ctc_samples, sample_count);
        printf("{\"schema_version\":1,\"width\":%u,\"iterations\":%u,\"warmup\":%u,"
               "\"canonical_ms\":%.6f,\"backend_ms\":%.6f,\"speedup\":%.6f,"
               "\"canonical_preprocess_ms\":%.6f,\"backend_preprocess_ms\":%.6f,"
               "\"backend_backbone_ms\":%.6f,\"backend_ctc_ms\":%.6f,\"backend_graph_ms\":%.6f,"
               "\"backend_profile_runs\":%u,"
               "\"backend_profile_ms\":{\"pointwise\":%.6f,\"dense\":%.6f,\"depthwise\":%.6f,\"affine\":%.6f,"
               "\"binary\":%.6f,\"unary\":%.6f,\"reduce\":%.6f,\"pool\":%.6f,"
               "\"transpose\":%.6f,\"matmul\":%.6f,\"ctc\":%.6f},"
               "\"physical_ops\":%u,\"semantic_nodes\":%u,\"arena_bytes\":%llu,"
               "\"scratch_bytes\":%llu,\"unsupported_nodes\":%u,"
               "\"pointwise_ops\":%u,\"dense_ops\":%u,\"depthwise_ops\":%u,"
               "\"pointwise_fallbacks\":%u,\"dense_fallbacks\":%u,\"depthwise_fallbacks\":%u,"
               "\"scalar_conv_fallbacks\":%u,\"text_match\":true}\n",
                width, sample_count, WARMUP_ROUNDS, canonical_median, backend_median, speedup,
                canonical_preprocess, backend_preprocess, backend_backbone, backend_ctc,
                backend_backbone + backend_ctc, profile_count,
                category_totals_ms[0] / profile_count, category_totals_ms[1] / profile_count,
                category_totals_ms[2] / profile_count, category_totals_ms[3] / profile_count,
                category_totals_ms[4] / profile_count, category_totals_ms[5] / profile_count,
                category_totals_ms[6] / profile_count, category_totals_ms[7] / profile_count,
                category_totals_ms[8] / profile_count, category_totals_ms[9] / profile_count,
                profile_ctc_total_ms / profile_count,
                program->op_count, program->semantic_consumed + (program->ctc_fused ? 3u : 0u),
                (unsigned long long)program->arena_bytes,
                (unsigned long long)program->scratch_bytes, program->unsupported_nodes,
                pointwise_ops, dense_ops, depthwise_ops, pointwise_fallbacks,
                dense_fallbacks, depthwise_fallbacks,
                pointwise_fallbacks + dense_fallbacks + depthwise_fallbacks);
    }
    result = 0;

cleanup:
    free(backend_ctc_samples);
    free(backend_backbone_samples);
    free(backend_preprocess_samples);
    free(canonical_preprocess_samples);
    free(backend_samples);
    free(canonical_samples);
    free(canonical_probabilities);
    free(canonical_indices);
    free(canonical_input);
    free(source);
    lw_x64_rec_instance_free(instance);
    lw_x64_rec_program_free(program);
    lw_session_free(canonical);
    lw_model_free(model);
    return result;
}

