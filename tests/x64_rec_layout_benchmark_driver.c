#include "x64_rec_backend_internal.h"
#include "model_internal.h"
#include "rec_internal.h"
#include "session_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

static uint64_t clock_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (uint64_t)((double)c.QuadPart * 1000000000.0 / (double)f.QuadPart);
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
#endif
}
static double median(double* values, uint32_t count) {
    for (uint32_t i = 1u; i < count; ++i) {
        double x = values[i];
        uint32_t j = i;
        while (j > 0u && values[j - 1u] > x) { values[j] = values[j - 1u]; --j; }
        values[j] = x;
    }
    return (count & 1u) ? values[count / 2u] : (values[count / 2u - 1u] + values[count / 2u]) * 0.5;
}
static void fill_source(uint8_t* source, uint32_t width, uint32_t height) {
    uint64_t count = (uint64_t)width * height * 3u;
    for (uint64_t i = 0u; i < count; ++i) source[i] = (uint8_t)((i * 37u + i / (width * 3u) * 13u + 17u) & 0xffu);
}
static int compare_ctc(const uint32_t* a, const float* ap, const uint32_t* b, const float* bp, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) if (a[i] != b[i] || fabsf(ap[i] - bp[i]) > 1.0e-5f) return 0;
    return 1;
}
static int run_canonical(lw_session* session, const uint8_t* source, uint64_t source_bytes,
                          float* input, uint64_t input_count, uint32_t width, uint32_t height,
                          uint32_t* indices, float* probabilities, uint32_t rows,
                          uint32_t classes, double* elapsed) {
    uint64_t start = clock_ns();
    lw_error error;
    lw_error_init(&error);
    if (lw_rec_preprocess_bgr_u8(source, source_bytes, width, height, width * 3u, 960u,
                                 input, input_count, NULL) != LW_STATUS_OK ||
        lw_execute_session_f32_ctc_greedy(session, input, input_count, indices, probabilities,
                                          rows, classes, NULL, &error) != LW_STATUS_OK) {
        fprintf(stderr, "canonical run failed: %s\n", error.message);
        return 0;
    }
    if (elapsed != NULL) *elapsed = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}

static int run_backend(lw_x64_rec_instance* instance, int nchw, const uint8_t* source,
                       uint64_t source_bytes, uint32_t width, uint32_t height,
                       uint32_t* indices, float* probabilities, uint32_t rows,
                       double* elapsed) {
    uint64_t count = 0u;
    float* input = lw_x64_rec_instance_input(instance, &count);
    uint64_t start = clock_ns();
    if (input == NULL ||
        (nchw ? lw_rec_preprocess_bgr_u8(source, source_bytes, width, height, width * 3u, 960u, input, count, NULL)
              : lw_rec_preprocess_bgr_u8_nhwc(source, source_bytes, width, height, width * 3u, 960u, input, count, NULL)) != LW_STATUS_OK) return 0;
    lw_error error;
    lw_error_init(&error);
    if (lw_x64_rec_instance_run(instance, &error) != LW_STATUS_OK) { fprintf(stderr, "backend run failed: %s\n", error.message); return 0; }
    if (elapsed != NULL) *elapsed = (double)(clock_ns() - start) / 1000000.0;
    if (indices != NULL) for (uint32_t i = 0u; i < rows; ++i) { indices[i] = instance->best_indices[i]; probabilities[i] = instance->best_probabilities[i]; }
    return 1;
}
int main(int argc, char** argv) {
    const uint32_t width = 960u, height = 48u, warmups = 2u;
    uint32_t repeats = 10u;
    const uint64_t source_bytes = (uint64_t)width * height * 3u;
    const uint64_t input_count = (uint64_t)3u * height * width;
    lw_model* model = NULL;
    lw_x64_rec_program* nhwc_program = NULL, *nchw_program = NULL;
    lw_x64_rec_instance* nhwc = NULL, *nchw = NULL;
    lw_session* canonical = NULL;
    lw_tensor_desc desc;
    lw_error error;
    uint8_t* source = NULL;
    float* canonical_input = NULL;
    uint32_t* canonical_indices = NULL, *nhwc_indices = NULL, *backend_indices = NULL;
    float* canonical_probabilities = NULL, *nhwc_probabilities = NULL, *backend_probabilities = NULL;
    double* canonical_ms = NULL, *nhwc_ms = NULL, *nchw_ms = NULL;
    int code = 1;
    if (argc < 2 || argc > 3) { fprintf(stderr, "usage: x64-rec-layout-benchmark-driver rec.lwm [repeats]\n"); return 2; }
    if (argc == 3) { repeats = (uint32_t)strtoul(argv[2], NULL, 10); if (repeats == 0u || repeats > 100u) return 2; }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) { fprintf(stderr, "model load failed: %s\n", error.message); goto cleanup; }
    if (lw_x64_rec_backend_compile_ex(model, width, LW_X64_REC_COMPILE_NHWC, &nhwc_program, &error) != LW_X64_REC_COMPILE_OK ||
        lw_x64_rec_backend_compile_ex(model, width, LW_X64_REC_COMPILE_NCHW, &nchw_program, &error) != LW_X64_REC_COMPILE_OK) { fprintf(stderr, "backend compile failed: %s\n", error.message); goto cleanup; }
    if (lw_x64_rec_instance_create(nhwc_program, &nhwc, &error) != LW_STATUS_OK ||
        lw_x64_rec_instance_create(nchw_program, &nchw, &error) != LW_STATUS_OK) { fprintf(stderr, "instance create failed: %s\n", error.message); goto cleanup; }
    lw_tensor_desc_init(&desc);
    desc.dtype = LW_DTYPE_F32; desc.rank = 4u; desc.dimensions[0] = 1; desc.dimensions[1] = 3; desc.dimensions[2] = height; desc.dimensions[3] = width;
    if (lw_session_create_ctc_greedy(model, &desc, 1u, NULL, &canonical, &error) != LW_STATUS_OK) { fprintf(stderr, "canonical create failed: %s\n", error.message); goto cleanup; }
    source = (uint8_t*)malloc((size_t)source_bytes);
    canonical_input = (float*)malloc((size_t)input_count * sizeof(float));
    canonical_indices = (uint32_t*)malloc((size_t)nhwc_program->time_steps * sizeof(uint32_t));
    nhwc_indices = (uint32_t*)malloc((size_t)nhwc_program->time_steps * sizeof(uint32_t));
    backend_indices = (uint32_t*)malloc((size_t)nhwc_program->time_steps * sizeof(uint32_t));
    canonical_probabilities = (float*)malloc((size_t)nhwc_program->time_steps * sizeof(float));
    nhwc_probabilities = (float*)malloc((size_t)nhwc_program->time_steps * sizeof(float));
    backend_probabilities = (float*)malloc((size_t)nhwc_program->time_steps * sizeof(float));
    canonical_ms = (double*)malloc((size_t)repeats * sizeof(double));
    nhwc_ms = (double*)malloc((size_t)repeats * sizeof(double));
    nchw_ms = (double*)malloc((size_t)repeats * sizeof(double));
    if (!source || !canonical_input || !canonical_indices || !nhwc_indices || !backend_indices || !canonical_probabilities || !nhwc_probabilities || !backend_probabilities || !canonical_ms || !nhwc_ms || !nchw_ms) goto cleanup;
    fill_source(source, width, height);
    for (uint32_t i = 0u; i < warmups; ++i) {
        uint32_t order[3] = { i % 3u, (i + 1u) % 3u, (i + 2u) % 3u };
        for (uint32_t j = 0u; j < 3u; ++j) {
            if (order[j] == 0u) {
                if (!run_canonical(canonical, source, source_bytes, canonical_input, input_count,
                                   width, height, canonical_indices, canonical_probabilities,
                                   nhwc_program->time_steps, nhwc_program->class_count, NULL)) goto cleanup;
            } else if (order[j] == 1u) {
                if (!run_backend(nhwc, 0, source, source_bytes, width, height, NULL, NULL, 0u, NULL)) goto cleanup;
            } else {
                if (!run_backend(nchw, 1, source, source_bytes, width, height, NULL, NULL, 0u, NULL)) goto cleanup;
            }
        }
    }
    for (uint32_t i = 0u; i < repeats; ++i) {
        uint32_t order[3] = { i % 3u, (i + 1u) % 3u, (i + 2u) % 3u };
        for (uint32_t j = 0u; j < 3u; ++j) {
            if (order[j] == 0u) {
                if (!run_canonical(canonical, source, source_bytes, canonical_input, input_count,
                                   width, height, canonical_indices, canonical_probabilities,
                                   nhwc_program->time_steps, nhwc_program->class_count,
                                   &canonical_ms[i])) goto cleanup;
            } else if (order[j] == 1u) {
                if (!run_backend(nhwc, 0, source, source_bytes, width, height, nhwc_indices,
                                 nhwc_probabilities, nhwc_program->time_steps, &nhwc_ms[i])) goto cleanup;
            } else {
                if (!run_backend(nchw, 1, source, source_bytes, width, height, backend_indices,
                                 backend_probabilities, nchw_program->time_steps, &nchw_ms[i])) goto cleanup;
            }
        }
        if (!compare_ctc(canonical_indices, canonical_probabilities, nhwc_indices,
                         nhwc_probabilities, nhwc_program->time_steps) ||
            !compare_ctc(canonical_indices, canonical_probabilities, backend_indices,
                         backend_probabilities, nhwc_program->time_steps)) {
            fprintf(stderr, "rotated backend CTC mismatch at round %u\n", i);
            goto cleanup;
        }
    }
    {
        double canonical_median = median(canonical_ms, repeats);
        double nhwc_median = median(nhwc_ms, repeats);
        double nchw_median = median(nchw_ms, repeats);
        printf("{\"schema_version\":1,\"order_rotated\":true,\"width\":%u,\"height\":%u,\"repeats\":%u,\"canonical_ms\":%.3f,\"nhwc_ms\":%.3f,\"nchw_ms\":%.3f,\"nhwc_speedup\":%.6f,\"nchw_speedup\":%.6f,\"nhwc_arena_bytes\":%llu,\"nchw_arena_bytes\":%llu}\n",
               width, height, repeats, canonical_median, nhwc_median, nchw_median,
               canonical_median / nhwc_median, canonical_median / nchw_median,
               (unsigned long long)nhwc_program->arena_bytes, (unsigned long long)nchw_program->arena_bytes);
        /* The lifetime planner must reuse arena space aggressively; the
         * monotonic baseline was 99,806,656 bytes and the planned Tiny REC960
         * high-water mark is a few MiB. Guard against silent regressions back
         * to monotonic allocation. */
        if (nhwc_program->arena_bytes > 10000000u ||
            nchw_program->arena_bytes > 10000000u) {
            fprintf(stderr, "arena lifetime planner regression: nhwc=%llu nchw=%llu bytes\n",
                    (unsigned long long)nhwc_program->arena_bytes,
                    (unsigned long long)nchw_program->arena_bytes);
            goto cleanup;
        }
    }
    code = 0;
cleanup:
    free(nchw_ms); free(nhwc_ms); free(canonical_ms); free(backend_probabilities); free(nhwc_probabilities); free(canonical_probabilities); free(backend_indices); free(nhwc_indices); free(canonical_indices); free(canonical_input); free(source);
    lw_session_free(canonical); lw_x64_rec_instance_free(nchw); lw_x64_rec_instance_free(nhwc); lw_x64_rec_program_free(nchw_program); lw_x64_rec_program_free(nhwc_program); lw_model_free(model);
    return code;
}
