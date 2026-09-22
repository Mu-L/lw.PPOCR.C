#include "x64_rec_backend_internal.h"

#include <stdio.h>

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_x64_rec_compiled_model* compiled = NULL;
    lw_error error;
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
    result = lw_x64_rec_backend_compile(model, 960u, &compiled, &error);
    printf("{\"compile_result\":%u,\"error_code\":%d,\"compiled\":%s}\n",
           (unsigned)result, error.code, compiled == NULL ? "false" : "true");
    if (result == LW_X64_REC_COMPILE_OK) {
        int valid = compiled != NULL && compiled->generic_ops == 0u &&
                    compiled->layout_conversions == 0u &&
                    compiled->direct_nhwc != 0u && compiled->ctc_fused != 0u;
        lw_x64_rec_compiled_model_release(compiled);
        lw_model_free(model);
        return valid ? 0 : 1;
    }
    lw_model_free(model);
    return result == LW_X64_REC_COMPILE_UNSUPPORTED &&
           compiled == NULL && error.code == LW_STATUS_UNSUPPORTED ? 0 : 1;
}