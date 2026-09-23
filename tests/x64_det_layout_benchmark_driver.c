/* A/B benchmark for the standalone x64 DET backend: canonical session vs the
 * NCHW arm vs the NHWC arm, rotated order, median-of-repeats, correctness and
 * determinism checks every round. The NHWC arm shards its convs over the
 * backend pool (same worker count as the canonical session); the NCHW arm
 * stays single-threaded as the control. */

#include "x64_det_backend_internal.h"
#include "model_internal.h"
#include "parallel_internal.h"
#include "session_internal.h"

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
static uint64_t profile_clock(void* context) {
    (void)context;
    return clock_ns();
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
static void fill_input(float* input, uint64_t count) {
    uint32_t state = 0x2545f491u;
    uint64_t i;
    for (i = 0u; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        input[i] = ((float)(state & 0xffffffu) / (float)0x1000000u) * 2.0f - 1.0f;
    }
}
static int compare_probabilities(const float* reference, const float* actual, uint64_t count) {
    uint64_t i;
    for (i = 0u; i < count; ++i) {
        double diff = fabs((double)reference[i] - (double)actual[i]);
        if (diff > 1.0e-3 * fabs((double)reference[i]) + 1.0e-5) {
            fprintf(stderr, "probability mismatch at %llu: ref %.9g got %.9g\n",
                    (unsigned long long)i, reference[i], actual[i]);
            return 0;
        }
    }
    return 1;
}
static int run_canonical(lw_session* session, const float* input, uint64_t input_count,
                         float* output, uint64_t output_count, double* elapsed) {
    lw_error error;
    uint64_t start = clock_ns();
    lw_error_init(&error);
    if (lw_execute_session_f32(session, input, input_count, output, output_count, &error) !=
        LW_STATUS_OK) {
        fprintf(stderr, "canonical run failed: %s\n", error.message);
        return 0;
    }
    if (elapsed != NULL) *elapsed = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}
static int run_backend(lw_x64_det_instance* instance, float* output, uint64_t output_count,
                       lw_thread_pool* pool, uint32_t worker_count, double* elapsed) {
    lw_error error;
    lw_execution_profile profile;
    uint64_t start = clock_ns();
    lw_error_init(&error);
    memset(&profile, 0, sizeof(profile));
    profile.struct_size = (uint32_t)sizeof(profile);
    profile.clock = profile_clock;
    if (lw_x64_det_instance_run_profiled_ex(instance, output, output_count, pool, worker_count,
                                            &profile, &error) != LW_STATUS_OK) {
        fprintf(stderr, "backend run failed: %s\n", error.message);
        return 0;
    }
    if (elapsed != NULL) *elapsed = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}
int main(int argc, char** argv) {
    uint32_t height = 640u;
    uint32_t width = 640u;
    uint32_t workers = 1u;
    uint32_t repeats = 8u;
    const uint32_t warmups = 2u;
    uint64_t plane;
    uint64_t input_count;
    lw_model* model = NULL;
    lw_x64_det_program* nhwc_program = NULL;
    lw_x64_det_program* nchw_program = NULL;
    lw_x64_det_instance* nhwc = NULL;
    lw_x64_det_instance* nchw = NULL;
    lw_session* canonical = NULL;
    lw_thread_pool* backend_pool = NULL;
    lw_tensor_desc desc;
    lw_error error;
    float* source_input = NULL;
    float* canonical_output = NULL;
    float* round_output = NULL;
    float* determinism_output = NULL;
    double* canonical_ms = NULL;
    double* nhwc_ms = NULL;
    double* nchw_ms = NULL;
    int code = 1;
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "usage: %s det.lwm [height] [width] [workers] [repeats]\n", argv[0]);
        return 2;
    }
    if (argc >= 3) { height = (uint32_t)strtoul(argv[2], NULL, 10); if (height == 0u) return 2; }
    if (argc >= 4) { width = (uint32_t)strtoul(argv[3], NULL, 10); if (width == 0u) return 2; }
    if (argc >= 5) { workers = (uint32_t)strtoul(argv[4], NULL, 10); if (workers == 0u || workers > 64u) return 2; }
    if (argc >= 6) { repeats = (uint32_t)strtoul(argv[5], NULL, 10); if (repeats == 0u || repeats > 100u) return 2; }
    plane = (uint64_t)height * width;
    input_count = plane * 3u;
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_det_backend_compile_ex(model, height, width, LW_X64_DET_COMPILE_NHWC,
                                      &nhwc_program, &error) != LW_X64_DET_COMPILE_OK ||
        lw_x64_det_backend_compile_ex(model, height, width, LW_X64_DET_COMPILE_NCHW,
                                      &nchw_program, &error) != LW_X64_DET_COMPILE_OK) {
        fprintf(stderr, "backend compile failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_x64_det_instance_create(nhwc_program, &nhwc, &error) != LW_STATUS_OK ||
        lw_x64_det_instance_create(nchw_program, &nchw, &error) != LW_STATUS_OK) {
        fprintf(stderr, "instance create failed: %s\n", error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&desc);
    desc.dtype = LW_DTYPE_F32;
    desc.rank = 4u;
    desc.dimensions[0] = 1;
    desc.dimensions[1] = 3;
    desc.dimensions[2] = (int32_t)height;
    desc.dimensions[3] = (int32_t)width;
    if (lw_session_create(model, &desc, 1u, NULL, &canonical, &error) != LW_STATUS_OK) {
        fprintf(stderr, "canonical create failed: %s\n", error.message);
        goto cleanup;
    }
    lw_session_set_intra_op_thread_count(canonical, workers);
    backend_pool = lw_thread_pool_create(workers);
    source_input = (float*)malloc((size_t)input_count * sizeof(float));
    canonical_output = (float*)malloc((size_t)plane * sizeof(float));
    round_output = (float*)malloc((size_t)plane * sizeof(float));
    determinism_output = (float*)malloc((size_t)plane * sizeof(float));
    canonical_ms = (double*)malloc((size_t)repeats * sizeof(double));
    nhwc_ms = (double*)malloc((size_t)repeats * sizeof(double));
    nchw_ms = (double*)malloc((size_t)repeats * sizeof(double));
    if (!source_input || !canonical_output || !round_output || !determinism_output ||
        !canonical_ms || !nhwc_ms || !nchw_ms) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    fill_input(source_input, input_count);
    {
        uint64_t count = 0u;
        float* nhwc_input = lw_x64_det_instance_input(nhwc, &count);
        float* nchw_input = lw_x64_det_instance_input(nchw, &count);
        if (nhwc_input == NULL || nchw_input == NULL || count != input_count) {
            fprintf(stderr, "backend input slot is invalid\n");
            goto cleanup;
        }
        memcpy(nhwc_input, source_input, (size_t)input_count * sizeof(float));
        memcpy(nchw_input, source_input, (size_t)input_count * sizeof(float));
    }
    /* Warmups, rotated order (same pattern as the REC layout benchmark). */
    for (uint32_t i = 0u; i < warmups; ++i) {
        uint32_t order[3] = { i % 3u, (i + 1u) % 3u, (i + 2u) % 3u };
        for (uint32_t j = 0u; j < 3u; ++j) {
            if (order[j] == 0u) {
                if (!run_canonical(canonical, source_input, input_count, canonical_output,
                                   plane, NULL)) goto cleanup;
            } else if (order[j] == 1u) {
                if (!run_backend(nhwc, round_output, plane, backend_pool, workers, NULL)) goto cleanup;
            } else {
                if (!run_backend(nchw, round_output, plane, backend_pool, workers, NULL)) goto cleanup;
            }
        }
    }
    for (uint32_t i = 0u; i < repeats; ++i) {
        uint32_t order[3] = { i % 3u, (i + 1u) % 3u, (i + 2u) % 3u };
        for (uint32_t j = 0u; j < 3u; ++j) {
            if (order[j] == 0u) {
                if (!run_canonical(canonical, source_input, input_count, canonical_output,
                                   plane, &canonical_ms[i])) goto cleanup;
            } else if (order[j] == 1u) {
                if (!run_backend(nhwc, round_output, plane, backend_pool, workers, &nhwc_ms[i])) goto cleanup;
                if (!compare_probabilities(canonical_output, round_output, plane)) {
                    fprintf(stderr, "NHWC mismatch at round %u\n", i);
                    goto cleanup;
                }
                if (!run_backend(nhwc, determinism_output, plane, backend_pool, workers, NULL)) goto cleanup;
                if (memcmp(round_output, determinism_output, (size_t)plane * sizeof(float)) != 0) {
                    fprintf(stderr, "NHWC backend is not bit-deterministic at round %u\n", i);
                    goto cleanup;
                }
            } else {
                if (!run_backend(nchw, round_output, plane, backend_pool, workers, &nchw_ms[i])) goto cleanup;
                if (!compare_probabilities(canonical_output, round_output, plane)) {
                    fprintf(stderr, "NCHW mismatch at round %u\n", i);
                    goto cleanup;
                }
                if (!run_backend(nchw, determinism_output, plane, backend_pool, workers, NULL)) goto cleanup;
                if (memcmp(round_output, determinism_output, (size_t)plane * sizeof(float)) != 0) {
                    fprintf(stderr, "NCHW backend is not bit-deterministic at round %u\n", i);
                    goto cleanup;
                }
            }
        }
    }
    {
        double canonical_median = median(canonical_ms, repeats);
        double nhwc_median = median(nhwc_ms, repeats);
        double nchw_median = median(nchw_ms, repeats);
        printf("{\"schema_version\":1,\"order_rotated\":true,\"height\":%u,\"width\":%u,"
               "\"workers\":%u,\"repeats\":%u,"
               "\"canonical_ms\":%.3f,\"nhwc_ms\":%.3f,\"nchw_ms\":%.3f,"
               "\"nhwc_speedup\":%.6f,\"nchw_speedup\":%.6f,"
               "\"nhwc_arena_bytes\":%llu,\"nchw_arena_bytes\":%llu,"
               "\"nhwc_conversions\":%u,\"nchw_conversions\":%u,"
               "\"nhwc_effective\":%u,\"nchw_effective\":%u}\n",
               height, width, workers, repeats,
               canonical_median, nhwc_median, nchw_median,
               canonical_median / nhwc_median, canonical_median / nchw_median,
               (unsigned long long)nhwc_program->arena_bytes,
               (unsigned long long)nchw_program->arena_bytes,
               nhwc_program->layout_conversions, nchw_program->layout_conversions,
               nhwc_program->nhwc_effective_nodes, nhwc_program->nchw_effective_nodes);
        /* The arena lifetime planner must reuse space aggressively. Monotonic
         * allocation would sum every tensor in the graph; a 256 MiB guard
         * catches silent regressions back to that. */
        if (nhwc_program->arena_bytes > 256u * 1024u * 1024u ||
            nchw_program->arena_bytes > 256u * 1024u * 1024u) {
            fprintf(stderr, "arena lifetime planner regression: nhwc=%llu nchw=%llu bytes\n",
                    (unsigned long long)nhwc_program->arena_bytes,
                    (unsigned long long)nchw_program->arena_bytes);
            goto cleanup;
        }
    }
    code = 0;
cleanup:
    free(nchw_ms); free(nhwc_ms); free(canonical_ms);
    free(determinism_output); free(round_output); free(canonical_output); free(source_input);
    lw_session_free(canonical);
    lw_thread_pool_free(backend_pool);
    lw_x64_det_instance_free(nchw); lw_x64_det_instance_free(nhwc);
    lw_x64_det_program_free(nchw_program); lw_x64_det_program_free(nhwc_program);
    lw_model_free(model);
    return code;
}
