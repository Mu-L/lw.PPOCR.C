#include "x64_rec_backend_internal.h"

#include <stdlib.h>

static lw_status backend_create_session(const lw_model* model, uint32_t target_width,
                                        lw_session** out_session, lw_error* error) {
    lw_tensor_desc input;
    if (model == NULL || out_session == NULL || target_width == 0u ||
        target_width > (uint32_t)INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC instance requires a valid width");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out_session = NULL;
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = 1;
    input.dimensions[1] = 3;
    input.dimensions[2] = 48;
    input.dimensions[3] = (int32_t)target_width;
    return lw_session_create(model, &input, 1u, NULL, out_session, error);
}

lw_status lw_x64_rec_instance_create(
    lw_x64_rec_compiled_model* compiled,
    uint32_t target_width,
    lw_x64_rec_instance** out_instance,
    lw_error* error) {
    lw_x64_rec_instance* instance;
    lw_status status;
    if (out_instance == NULL || compiled == NULL || compiled->model == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC instance requires a compiled model");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out_instance = NULL;
    instance = (lw_x64_rec_instance*)calloc(1u, sizeof(*instance));
    if (instance == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "x64 REC instance allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    status = backend_create_session(compiled->model, target_width, &instance->session, error);
    if (status != LW_STATUS_OK) {
        free(instance);
        return status;
    }
    status = lw_x64_rec_fast_plan_create(instance->session, &instance->plan, error);
    if (status != LW_STATUS_OK ||
        instance->plan == NULL ||
        instance->plan->layout.graph_input_direct_nhwc == 0u ||
        instance->plan->generic_node_count != 0u ||
        instance->plan->layout.layout_conversion_count != 0u ||
        instance->plan->unsupported_nhwc_node_count != 0u) {
        if (status == LW_STATUS_OK) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED,
                         "x64 REC instance violates all-NHWC backend contract");
            status = LW_STATUS_UNSUPPORTED;
        }
        lw_x64_rec_instance_free(instance);
        return status;
    }
    instance->compiled = compiled;
    instance->target_width = target_width;
    instance->time_steps = 0u;
    lw_x64_rec_compiled_model_retain(compiled);
    *out_instance = instance;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_x64_rec_instance_free(lw_x64_rec_instance* instance) {
    if (instance == NULL) return;
    lw_x64_rec_fast_plan_free(instance->plan);
    lw_session_free(instance->session);
    lw_x64_rec_compiled_model_release(instance->compiled);
    free(instance);
}

float* lw_x64_rec_instance_input(
    lw_x64_rec_instance* instance,
    uint64_t* element_count) {
    if (element_count != NULL) *element_count = 0u;
    if (instance == NULL || instance->plan == NULL) return NULL;
    return lw_x64_rec_fast_graph_input_nhwc(instance->plan, element_count);
}

lw_status lw_x64_rec_instance_run_backbone(
    lw_x64_rec_instance* instance,
    lw_x64_fast_profile* profile,
    lw_error* error) {
    if (instance == NULL || instance->plan == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC instance is not initialized");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    return lw_x64_rec_fast_run_prepared_nhwc(instance->plan, profile, error);
}