#include "x64_rec_backend_internal.h"

#include <stdlib.h>

static lw_status backend_create_session(const lw_model* model, uint32_t target_width,
                                        lw_session** out_session, lw_error* error) {
    lw_tensor_desc input;
    if (model == NULL || out_session == NULL || target_width == 0u ||
        target_width > (uint32_t)INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC backend requires a model and width");
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

static lw_x64_rec_compile_result backend_compile_result(lw_status status) {
    if (status == LW_STATUS_OUT_OF_MEMORY) return LW_X64_REC_COMPILE_OUT_OF_MEMORY;
    if (status == LW_STATUS_INVALID_ARGUMENT || status == LW_STATUS_INVALID_SHAPE ||
        status == LW_STATUS_INVALID_FORMAT || status == LW_STATUS_OUT_OF_BOUNDS) {
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    return LW_X64_REC_COMPILE_UNSUPPORTED;
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model,
    uint32_t target_width,
    lw_x64_rec_compiled_model** out_compiled,
    lw_error* error) {
    lw_model_info info;
    lw_session* session = NULL;
    lw_x64_rec_fast_plan* plan = NULL;
    lw_x64_rec_compiled_model* compiled = NULL;
    lw_cpu_capabilities cpu;
    lw_status status;
    if (out_compiled == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC backend output is null");
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    *out_compiled = NULL;
    if (model == NULL || target_width == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "x64 REC backend requires a model and width");
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    lw_model_info_init(&info);
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK) {
        lw_set_error(error, status, "x64 REC backend cannot inspect model");
        return backend_compile_result(status);
    }
    cpu = lw_get_cpu_capabilities();
    if (!lw_simd_level_is_avx2(cpu.simd) || !cpu.has_avx2_fma) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "x64 REC backend requires AVX2 and FMA");
        return LW_X64_REC_COMPILE_UNSUPPORTED;
    }
    status = backend_create_session(model, target_width, &session, error);
    if (status != LW_STATUS_OK) return backend_compile_result(status);
    status = lw_x64_rec_fast_plan_create(session, &plan, error);
    if (status != LW_STATUS_OK) {
        lw_session_free(session);
        return backend_compile_result(status);
    }
    if (plan->layout.graph_input_direct_nhwc == 0u ||
        plan->generic_node_count != 0u ||
        plan->layout.layout_conversion_count != 0u ||
        plan->unsupported_nhwc_node_count != 0u) {
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "x64 REC backend requires an all-NHWC, generic-free graph");
        lw_x64_rec_fast_plan_free(plan);
        lw_session_free(session);
        return LW_X64_REC_COMPILE_UNSUPPORTED;
    }
    compiled = (lw_x64_rec_compiled_model*)calloc(1u, sizeof(*compiled));
    if (compiled == NULL) {
        lw_x64_rec_fast_plan_free(plan);
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "x64 REC compiled model allocation failed");
        return LW_X64_REC_COMPILE_OUT_OF_MEMORY;
    }
    lw_atomic_u32_init(&compiled->ref_count, 1u);
    compiled->model = model;
    compiled->cpu = cpu;
    compiled->tensor_count = info.tensor_count;
    compiled->node_count = info.node_count;
    compiled->model_signature = info.content_checksum;
    compiled->generic_ops = plan->generic_node_count;
    compiled->layout_conversions = plan->layout.layout_conversion_count;
    compiled->direct_nhwc = 1u;
    compiled->ctc_fused = 0u;
    lw_x64_rec_fast_plan_free(plan);
    lw_session_free(session);
    lw_set_error(error, LW_STATUS_OK, "");
    *out_compiled = compiled;
    return LW_X64_REC_COMPILE_OK;
}

void lw_x64_rec_compiled_model_retain(lw_x64_rec_compiled_model* compiled) {
    uint32_t observed;
    if (compiled == NULL) return;
    observed = lw_atomic_u32_load_acquire(&compiled->ref_count);
    while (observed != UINT32_MAX &&
           !lw_atomic_u32_compare_exchange_acq_rel(&compiled->ref_count, &observed, observed + 1u)) {
    }
}

void lw_x64_rec_compiled_model_release(lw_x64_rec_compiled_model* compiled) {
    uint32_t observed;
    if (compiled == NULL) return;
    observed = lw_atomic_u32_load_acquire(&compiled->ref_count);
    while (observed != 0u &&
           !lw_atomic_u32_compare_exchange_acq_rel(&compiled->ref_count, &observed, observed - 1u)) {
    }
    if (observed == 1u) free(compiled);
}