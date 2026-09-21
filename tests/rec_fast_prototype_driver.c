#include "executor_internal.h"
#include "lw_infer.h"
#include "x64_rec_fast_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#include <stdlib.h>

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL) return 0;
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != 0 || parsed == 0u || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static uint64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)((counter.QuadPart * UINT64_C(1000000000)) / frequency.QuadPart);
#else
    struct timespec value;
    timespec_get(&value, TIME_UTC);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static int run_legacy(lw_session* session, const float* input, uint64_t input_count,
                      float* output, uint64_t output_count, lw_error* error) {
    return lw_execute_session_f32(session, input, input_count, output, output_count, error) == LW_STATUS_OK;
}

static int run_fast(lw_x64_rec_fast_plan* plan, const float* input, uint64_t input_count,
                    float* output, uint64_t output_count, lw_error* error) {
    return lw_x64_rec_fast_run(plan, input, input_count, output, output_count, error) == LW_STATUS_OK;
}

static double median_ms(double* values, uint32_t count) {
    uint32_t i;
    uint32_t j;
    for (i = 0u; i < count; ++i) {
        for (j = i + 1u; j < count; ++j) {
            if (values[j] < values[i]) {
                double swap = values[i];
                values[i] = values[j];
                values[j] = swap;
            }
        }
    }
    return values[count / 2u];
}

static uint64_t argmax_mismatch_count(const float* expected, const float* actual,
                                      uint64_t element_count, uint32_t columns) {
    uint64_t rows;
    uint64_t mismatch = 0u;
    if (expected == NULL || actual == NULL || columns == 0u || element_count % columns != 0u) return UINT64_MAX;
    rows = element_count / columns;
    for (uint64_t row = 0u; row < rows; ++row) {
        uint32_t expected_best = 0u;
        uint32_t actual_best = 0u;
        for (uint32_t column = 1u; column < columns; ++column) {
            if (expected[(size_t)row * columns + column] > expected[(size_t)row * columns + expected_best]) expected_best = column;
            if (actual[(size_t)row * columns + column] > actual[(size_t)row * columns + actual_best]) actual_best = column;
        }
        if (expected_best != actual_best) ++mismatch;
    }
    return mismatch;
}
static int test_layout_roundtrip(void) {
    const uint32_t dimensions[4] = {1u, 2u, 2u, 3u};
    float source[12];
    float nhwc[12];
    float roundtrip[12];
    uint32_t index;
    for (index = 0u; index < 12u; ++index) source[index] = (float)index - 4.0f;
    lw_x64_fast_nchw_to_nhwc(source, nhwc, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    lw_x64_fast_nhwc_to_nchw(nhwc, roundtrip, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    for (index = 0u; index < 12u; ++index) {
        if (source[index] != roundtrip[index]) return 0;
    }
    return 1;
}

static int element_count(const lw_tensor_desc* desc, uint64_t* count) {
    uint64_t value = 1u;
    uint32_t index;
    if (desc == NULL || count == NULL || desc->dtype != LW_DTYPE_F32 || desc->rank == 0u ||
        desc->rank > LW_MAX_DIMS) {
        return 0;
    }
    for (index = 0u; index < desc->rank; ++index) {
        if (desc->dimensions[index] <= 0 || value > UINT64_MAX / (uint64_t)desc->dimensions[index]) {
            return 0;
        }
        value *= (uint64_t)desc->dimensions[index];
    }
    *count = value;
    return 1;
}

int main(int argc, char** argv) {
    const uint32_t height = 48u;
    uint32_t width = 960u;
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_x64_rec_fast_plan* plan = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_error error;
    lw_status status;
    float* input = NULL;
    float* expected = NULL;
    float* actual = NULL;
    uint64_t input_count;
    uint64_t output_count;
    uint64_t index;
    float max_abs = 0.0f;
    uint64_t mismatch = 0u;
    uint64_t argmax_mismatch = 0u;
    int exit_code = 1;
    double legacy_samples[9];
    double fast_samples[9];
    uint32_t sample_index;
    double legacy_ms;
    double fast_ms;

    if (!test_layout_roundtrip()) {
        fprintf(stderr, "NCHW/NHWC layout roundtrip failed\n" );
        return 1;
    }
    if ((argc != 2 && argc != 3) || (argc == 3 && !parse_u32(argv[2], &width))) {
        fprintf(stderr, "usage: rec-fast-prototype-driver rec.lwm [width]\n");
        return 2;
    }
    input_count = UINT64_C(3) * height * width;
    if (input_count > SIZE_MAX / sizeof(*input)) {
        fprintf(stderr, "input is too large\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
    status = lw_session_create(model, &input_desc, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(session, 0u, &output_desc);
    if (status != LW_STATUS_OK || !element_count(&output_desc, &output_count)) {
        fprintf(stderr, "output descriptor query failed\n");
        goto cleanup;
    }
    input = (float*)malloc((size_t)input_count * sizeof(*input));
    expected = (float*)malloc((size_t)output_count * sizeof(*expected));
    actual = (float*)malloc((size_t)output_count * sizeof(*actual));
    if (input == NULL || expected == NULL || actual == NULL) {
        fprintf(stderr, "benchmark buffer allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < input_count; ++index) {
        input[index] = (float)((int32_t)((index * 17u) % 257u) - 128) / 128.0f;
    }
    status = lw_x64_rec_fast_plan_create(session, &plan, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "fast plan create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    status = lw_execute_session_f32(session, input, input_count, expected, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "reference execution failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    status = lw_x64_rec_fast_run(plan, input, input_count, actual, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "fast execution failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    for (index = 0u; index < output_count; ++index) {
        float difference = fabsf(expected[index] - actual[index]);
        if (difference > max_abs) max_abs = difference;
        if (difference > 1.0e-4f) ++mismatch;
    }
    argmax_mismatch = argmax_mismatch_count(expected, actual, output_count,
        output_desc.rank == 0u ? 0u : (uint32_t)output_desc.dimensions[output_desc.rank - 1u]);
    if (argmax_mismatch == UINT64_MAX) {
        fprintf(stderr, "unable to calculate argmax mismatch\\n");
        goto cleanup;
    }
    for (sample_index = 0u; sample_index < 2u; ++sample_index) {
        if (!run_legacy(session, input, input_count, expected, output_count, &error) ||
            !run_fast(plan, input, input_count, actual, output_count, &error)) {
            fprintf(stderr, "benchmark warmup failed: %s\n", error.message);
            goto cleanup;
        }
    }
    for (sample_index = 0u; sample_index < 9u; ++sample_index) {
        uint64_t started;
        uint64_t finished;
        if ((sample_index & 1u) == 0u) {
            started = now_ns();
            if (!run_legacy(session, input, input_count, expected, output_count, &error)) goto cleanup;
            finished = now_ns();
            legacy_samples[sample_index] = (double)(finished - started) / 1000000.0;
            started = now_ns();
            if (!run_fast(plan, input, input_count, actual, output_count, &error)) goto cleanup;
            finished = now_ns();
            fast_samples[sample_index] = (double)(finished - started) / 1000000.0;
        } else {
            started = now_ns();
            if (!run_fast(plan, input, input_count, actual, output_count, &error)) goto cleanup;
            finished = now_ns();
            fast_samples[sample_index] = (double)(finished - started) / 1000000.0;
            started = now_ns();
            if (!run_legacy(session, input, input_count, expected, output_count, &error)) goto cleanup;
            finished = now_ns();
            legacy_samples[sample_index] = (double)(finished - started) / 1000000.0;
        }
    }
    legacy_ms = median_ms(legacy_samples, 9u);
    fast_ms = median_ms(fast_samples, 9u);
    printf("{\"width\":%u,\"legacy_ms\":%.6f,\"fast_ms\":%.6f,\"speedup\":%.6f,\"max_abs\":%.9g,\"mismatch\":%llu,\"argmax_mismatch\":%llu,\"fast_nodes\":%u,\"generic_nodes\":%u,\"pointwise_nodes\":%u,\"dense_nodes\":%u,\"depthwise_nodes\":%u,\"binary_nodes\":%u,\"relu_nodes\":%u,\"unary_nodes\":%u,\"reduce_mean_nodes\":%u,\"pool_nodes\":%u,\"concat_nodes\":%u,\"batch_norm_nodes\":%u,\"planner_conversion_count\":%u,\"unsupported_nhwc_node_count\":%u,\"conversion_count\":%llu,\"conversion_bytes\":%llu,\"nhwc_workspace_bytes\":%llu}\n",
           width, legacy_ms, fast_ms, fast_ms > 0.0 ? legacy_ms / fast_ms : 0.0,
           (double)max_abs, (unsigned long long)mismatch, (unsigned long long)argmax_mismatch, plan->fast_node_count,
           plan->generic_node_count, plan->pointwise_node_count, plan->dense_node_count,
           plan->depthwise_node_count, plan->binary_node_count, plan->relu_node_count, plan->unary_node_count,
           plan->reduce_mean_node_count, plan->pool_node_count, plan->concat_node_count, plan->batch_norm_node_count,
           plan->layout.layout_conversion_count, plan->unsupported_nhwc_node_count,
           (unsigned long long)plan->conversion_count,
           (unsigned long long)plan->conversion_bytes,
           (unsigned long long)plan->nhwc_workspace_bytes);
    exit_code = mismatch == 0u && argmax_mismatch == 0u && plan->pointwise_node_count != 0u && plan->dense_node_count != 0u ? 0 : 1;

cleanup:
    lw_x64_rec_fast_plan_free(plan);
    free(actual);
    free(expected);
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
