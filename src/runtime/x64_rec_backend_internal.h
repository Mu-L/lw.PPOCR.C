#ifndef LW_X64_REC_BACKEND_INTERNAL_H
#define LW_X64_REC_BACKEND_INTERNAL_H

#include "atomic_internal.h"
#include "cpu_features.h"
#include "model_internal.h"
#include "session_internal.h"
#include "../kernels/nhwc_internal.h"

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
    LW_X64_REC_LAYOUT_VECTOR = 2,
    LW_X64_REC_LAYOUT_SCALAR = 3
} lw_x64_rec_storage_layout;

typedef enum lw_x64_rec_op_kind {
    LW_X64_REC_OP_DENSE = 1,
    LW_X64_REC_OP_POINTWISE = 2,
    LW_X64_REC_OP_DEPTHWISE = 3,
    LW_X64_REC_OP_AFFINE = 4,
    LW_X64_REC_OP_ADD = 5,
    LW_X64_REC_OP_MUL = 6,
    LW_X64_REC_OP_DIV = 7,
    LW_X64_REC_OP_RELU = 8,
    LW_X64_REC_OP_GELU = 9,
    LW_X64_REC_OP_HARD_SIGMOID = 10,
    LW_X64_REC_OP_REDUCE_MEAN = 11,
    LW_X64_REC_OP_AVG_POOL = 12,
    LW_X64_REC_OP_MAX_POOL = 13,
    LW_X64_REC_OP_CONCAT = 14,
    LW_X64_REC_OP_RESIZE = 15,
    LW_X64_REC_OP_TRANSPOSE_COPY = 16,
    LW_X64_REC_OP_CTC_PROJECTION = 17,
    LW_X64_REC_OP_CTC_BIAS = 18,
    LW_X64_REC_OP_CTC_SOFTMAX = 19,
    LW_X64_REC_OP_GENERIC_UNSUPPORTED = 0xffff
} lw_x64_rec_op_kind;

typedef struct lw_x64_rec_value {
    uint64_t offset;
    uint64_t bytes;
    const float* constant_data;
    int32_t producer;
    int32_t last_use;
    int32_t dimensions[4];
    uint32_t rank;
    uint8_t layout;
    uint8_t alias;
    uint16_t reserved;
    uint32_t alias_of;
} lw_x64_rec_value;

typedef struct lw_x64_rec_op {
    uint16_t kind;
    uint16_t semantic_op;
    uint32_t node_index;
    uint32_t input_count;
    uint32_t inputs[8];
    uint32_t output;
    uint32_t weight;
    uint32_t bias;
    uint32_t flags;
} lw_x64_rec_op;

typedef struct lw_x64_rec_ctc_tail {
    uint8_t enabled;
    uint8_t reserved[3];
    uint32_t activation_value;
    uint32_t classes;
    uint32_t rows;
    uint32_t inner;
    uint32_t weight_tensor;
    uint32_t bias_tensor;
} lw_x64_rec_ctc_tail;

struct lw_x64_rec_ctc_projection;
typedef struct lw_x64_rec_program {
    const lw_model* model;
    lw_cpu_capabilities cpu;
    uint64_t model_signature;
    uint32_t target_width;
    uint32_t time_steps;
    uint32_t class_count;
    uint32_t input_value;
    uint32_t output_value;
    uint32_t op_count;
    uint32_t value_count;
    uint32_t generic_ops;
    uint32_t layout_conversions;
    uint8_t direct_nhwc;
    uint8_t ctc_fused;
    uint16_t reserved;
    lw_x64_rec_op* ops;
    lw_x64_rec_value* values;
    uint8_t* arena;
    uint64_t arena_bytes;
    uint8_t* scratch;
    uint64_t scratch_bytes;
    lw_x64_rec_ctc_tail ctc;
    struct lw_x64_rec_ctc_projection* ctc_projection;
} lw_x64_rec_program;

typedef struct lw_x64_rec_instance {
    lw_x64_rec_program* program;
    lw_session* shape_session;
    uint8_t* scratch;
} lw_x64_rec_instance;

uint64_t lw_x64_rec_arena_align(uint64_t value, uint64_t alignment);
lw_status lw_x64_rec_arena_alloc(uint64_t* cursor, uint64_t bytes, uint64_t alignment,
                                 uint64_t* out_offset, lw_error* error);

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model, uint32_t target_width, lw_x64_rec_program** out_program,
    lw_error* error);
void lw_x64_rec_program_free(lw_x64_rec_program* program);

lw_status lw_x64_rec_instance_create(lw_x64_rec_program* program, lw_x64_rec_instance** out,
                                     lw_error* error);
void lw_x64_rec_instance_free(lw_x64_rec_instance* instance);
float* lw_x64_rec_instance_input(lw_x64_rec_instance* instance, uint64_t* element_count);
lw_status lw_x64_rec_instance_run_backbone(lw_x64_rec_instance* instance, lw_error* error);

#endif
