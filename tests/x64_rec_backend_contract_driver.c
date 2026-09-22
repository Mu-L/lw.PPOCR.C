#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "model_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance* instance = NULL;
    lw_error error;
    lw_x64_rec_compile_result result;
    uint64_t input_elements = 0u;
    float* input;

    if (argc != 2) {
        fprintf(stderr, "usage: x64-rec-backend-contract-driver rec.lwm\n");
        return 2;
    }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        return 1;
    }
    result = lw_x64_rec_backend_compile(model, 960u, &program, &error);
    if (result != LW_X64_REC_COMPILE_OK || program == NULL) {
        fprintf(stderr, "standalone compile failed: result=%u code=%d message=%s\n",
                (unsigned)result, error.code, error.message);
        lw_model_free(model);
        return 1;
    }
    printf("{\"compile_result\":%u,\"ops\":%u,\"values\":%u,\"arena_bytes\":%llu,\"scratch_bytes\":%llu,\"generic_ops\":%u,\"layout_conversions\":%u,\"direct_nhwc\":%s,\"ctc_fused\":%s}\n",
           (unsigned)result, (unsigned)program->op_count, (unsigned)program->value_count,
           (unsigned long long)program->arena_bytes,
           (unsigned long long)program->scratch_bytes, (unsigned)program->generic_ops,
           (unsigned)program->layout_conversions, program->direct_nhwc ? "true" : "false",
           program->ctc_fused ? "true" : "false");
    if (lw_x64_rec_instance_create(program, &instance, &error) != LW_STATUS_OK ||
        instance == NULL) {
        fprintf(stderr, "instance create failed: %s\n", error.message);
        lw_x64_rec_program_free(program);
        lw_model_free(model);
        return 1;
    }
    input = lw_x64_rec_instance_input(instance, &input_elements);
    if (input == NULL || input_elements == 0u) {
        fprintf(stderr, "instance input unavailable\n");
        lw_x64_rec_instance_free(instance);
        lw_x64_rec_program_free(program);
        lw_model_free(model);
        return 1;
    }
    memset(input, 0, (size_t)input_elements * sizeof(float));
    if (lw_x64_rec_instance_run(instance, &error) != LW_STATUS_OK) {
        fprintf(stderr, "backend execution failed: %s\n", error.message);
        lw_x64_rec_instance_free(instance);
        lw_x64_rec_program_free(program);
        lw_model_free(model);
        return 1;
    }
    lw_x64_rec_instance_free(instance);
    lw_x64_rec_program_free(program);
    lw_model_free(model);
    return 0;
}