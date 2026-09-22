#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_x64_rec_program* program = NULL;
    lw_error error;
    float* activation = NULL;
    lw_x64_rec_compile_result result;
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
    printf("{\"compile_result\":%u,\"ops\":%u,\"values\":%u,\"arena_bytes\":%llu,\"generic_ops\":%u,\"layout_conversions\":%u,\"direct_nhwc\":%s,\"ctc_fused\":%s}\n",
           (unsigned)result, (unsigned)program->op_count, (unsigned)program->value_count,
           (unsigned long long)program->arena_bytes, (unsigned)program->generic_ops,
           (unsigned)program->layout_conversions, program->direct_nhwc ? "true" : "false",
           program->ctc_fused ? "true" : "false");
    activation = (float*)calloc((size_t)program->ctc.rows * program->ctc.inner, sizeof(float));
    if (activation == NULL || lw_x64_rec_ctc_projection_run(program->ctc_projection, activation, &error) != LW_STATUS_OK) {
        free(activation);
        lw_x64_rec_program_free(program);
        lw_model_free(model);
        return 1;
    }
    free(activation);
    for (uint32_t value_index = 0u; value_index < program->value_count; ++value_index) {
        const lw_x64_rec_value* value = &program->values[value_index];
        if (value->constant_data == NULL && value->last_use < 0 && value_index != program->output_value) {
            lw_x64_rec_program_free(program);
            lw_model_free(model);
            return 1;
        }
        if (value->alias != 0u && value->alias_of >= program->value_count) {
            lw_x64_rec_program_free(program);
            lw_model_free(model);
            return 1;
        }
    }
    if (program->generic_ops != 0u ||
        program->direct_nhwc == 0u || program->ctc_fused == 0u || program->ctc_projection == NULL) {
        lw_x64_rec_program_free(program);
        lw_model_free(model);
        return 1;
    }
    lw_x64_rec_program_free(program);
    lw_model_free(model);
    return 0;
}
