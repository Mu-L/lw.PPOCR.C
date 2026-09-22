#ifndef LW_X64_REC_BACKEND_INTERNAL_H
#define LW_X64_REC_BACKEND_INTERNAL_H

#include "atomic_internal.h"
#include "cpu_features.h"
#include "lw_infer.h"
#include "x64_rec_fast_internal.h"

#include <stddef.h>
#include <stdint.h>

typedef enum lw_x64_rec_compile_result {
    LW_X64_REC_COMPILE_OK = 0,
    LW_X64_REC_COMPILE_UNSUPPORTED = 1,
    LW_X64_REC_COMPILE_INVALID_GRAPH = 2,
    LW_X64_REC_COMPILE_OUT_OF_MEMORY = 3
} lw_x64_rec_compile_result;

typedef enum lw_x64_rec_storage_layout {
    LW_X64_REC_LAYOUT_NHWC = 0,
    LW_X64_REC_LAYOUT_TC = 1,
    LW_X64_REC_LAYOUT_C = 2,
    LW_X64_REC_LAYOUT_SCALAR = 3
} lw_x64_rec_storage_layout;

typedef struct lw_x64_rec_compiled_model {
    lw_atomic_u32 ref_count;
    const lw_model* model;
    lw_cpu_capabilities cpu;
    uint32_t tensor_count;
    uint32_t node_count;
    uint64_t model_signature;
    uint32_t generic_ops;
    uint32_t layout_conversions;
    uint8_t direct_nhwc;
    uint8_t ctc_fused;
    uint16_t reserved;
} lw_x64_rec_compiled_model;

typedef struct lw_x64_rec_instance {
    lw_x64_rec_compiled_model* compiled;
    lw_session* session;
    lw_x64_rec_fast_plan* plan;
    uint32_t target_width;
    uint32_t time_steps;
} lw_x64_rec_instance;

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model,
    uint32_t target_width,
    lw_x64_rec_compiled_model** out_compiled,
    lw_error* error);

void lw_x64_rec_compiled_model_retain(lw_x64_rec_compiled_model* compiled);
void lw_x64_rec_compiled_model_release(lw_x64_rec_compiled_model* compiled);

lw_status lw_x64_rec_instance_create(
    lw_x64_rec_compiled_model* compiled,
    uint32_t target_width,
    lw_x64_rec_instance** out_instance,
    lw_error* error);

void lw_x64_rec_instance_free(lw_x64_rec_instance* instance);

float* lw_x64_rec_instance_input(
    lw_x64_rec_instance* instance,
    uint64_t* element_count);

lw_status lw_x64_rec_instance_run_backbone(
    lw_x64_rec_instance* instance,
    lw_x64_fast_profile* profile,
    lw_error* error);

#endif