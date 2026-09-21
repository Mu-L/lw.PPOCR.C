#include "executor_internal.h"
#include "lw_infer.h"
#include "x64_rec_fast_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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
    int exit_code = 1;

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
    printf("{\"width\":%u,\"max_abs\":%.9g,\"mismatch\":%llu,\"fast_nodes\":%u,\"generic_nodes\":%u,\"conversion_count\":%llu,\"conversion_bytes\":%llu,\"nhwc_workspace_bytes\":%llu}\n",
           width, (double)max_abs, (unsigned long long)mismatch, plan->fast_node_count,
           plan->generic_node_count, (unsigned long long)plan->conversion_count,
           (unsigned long long)plan->conversion_bytes,
           (unsigned long long)plan->nhwc_workspace_bytes);
    exit_code = mismatch == 0u ? 0 : 1;

cleanup:
    lw_x64_rec_fast_plan_free(plan);
    free(actual);
    free(expected);
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
