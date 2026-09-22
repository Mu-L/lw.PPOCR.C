#include "x64_rec_backend_internal.h"

#include <stdlib.h>
#include <string.h>

lw_status lw_x64_rec_instance_create(lw_x64_rec_program* program, lw_x64_rec_instance** out,
                                     lw_error* error) {
    lw_x64_rec_instance* instance;
    if (out == NULL || program == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC instance requires a compiled program");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    instance = (lw_x64_rec_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC instance allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    instance->program = program;
    if (program->scratch_bytes != 0u) {
        instance->scratch = (uint8_t*)malloc((size_t)program->scratch_bytes);
        if (instance->scratch == NULL) {
            free(instance);
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC scratch allocation failed");
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    *out = instance;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_x64_rec_instance_free(lw_x64_rec_instance* instance) {
    if (instance == NULL) return;
    free(instance->scratch);
    free(instance);
}

float* lw_x64_rec_instance_input(lw_x64_rec_instance* instance, uint64_t* element_count) {
    lw_x64_rec_value* value;
    if (element_count != NULL) *element_count = 0u;
    if (instance == NULL || instance->program == NULL || instance->program->arena == NULL) return NULL;
    value = &instance->program->values[instance->program->input_value];
    if (element_count != NULL) *element_count = value->bytes / sizeof(float);
    return (float*)(void*)(instance->program->arena + (size_t)value->offset);
}

lw_status lw_x64_rec_instance_run_backbone(lw_x64_rec_instance* instance, lw_error* error) {
    (void)instance;
    lw_set_error(error, LW_STATUS_UNSUPPORTED,
                 "standalone REC execution lowering is not enabled in this phase");
    return LW_STATUS_UNSUPPORTED;
}
