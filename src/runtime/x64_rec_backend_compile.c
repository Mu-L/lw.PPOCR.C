#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static lw_status make_shape_session(const lw_model* model, uint32_t width, lw_session** out,
                                    lw_error* error) {
    lw_tensor_desc input;
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = 1;
    input.dimensions[1] = 3;
    input.dimensions[2] = 48;
    input.dimensions[3] = (int32_t)width;
    return lw_session_create(model, &input, 1u, NULL, out, error);
}

static const uint8_t* node_bytes(const lw_model* model, uint32_t index) {
    return model->bytes + (size_t)model->node_offset + (size_t)index * LWM_V0_NODE_SIZE;
}

static lw_x64_rec_op_kind classify(uint16_t op) {
    switch (op) {
    case LW_OP_CONV: return LW_X64_REC_OP_DENSE;
    case LW_OP_BATCH_NORMALIZATION: return LW_X64_REC_OP_AFFINE;
    case LW_OP_ADD: return LW_X64_REC_OP_ADD;
    case LW_OP_MUL: return LW_X64_REC_OP_MUL;
    case LW_OP_DIV: return LW_X64_REC_OP_DIV;
    case LW_OP_RELU: return LW_X64_REC_OP_RELU;
    case LW_OP_ERF: return LW_X64_REC_OP_GELU;
    case LW_OP_HARD_SIGMOID: return LW_X64_REC_OP_HARD_SIGMOID;
    case LW_OP_REDUCE_MEAN: return LW_X64_REC_OP_REDUCE_MEAN;
    case LW_OP_AVERAGE_POOL: return LW_X64_REC_OP_AVG_POOL;
    case LW_OP_MAX_POOL: return LW_X64_REC_OP_MAX_POOL;
    case LW_OP_CONCAT: return LW_X64_REC_OP_CONCAT;
    case LW_OP_RESIZE: return LW_X64_REC_OP_RESIZE;
    case LW_OP_SQUEEZE:
    case LW_OP_UNSQUEEZE:
    case LW_OP_RESHAPE:
        return LW_X64_REC_OP_TRANSPOSE_COPY;
    case LW_OP_TRANSPOSE: return LW_X64_REC_OP_TRANSPOSE_COPY;
    case LW_OP_MATMUL: return LW_X64_REC_OP_DENSE;
    case LW_OP_SOFTMAX: return LW_X64_REC_OP_GENERIC_UNSUPPORTED;
    default: return LW_X64_REC_OP_GENERIC_UNSUPPORTED;
    }
}

static int detect_ctc(const lw_model* model, const lw_session* session, lw_x64_rec_ctc_tail* tail) {
    uint32_t n = model->info.node_count;
    const uint8_t* matmul;
    const uint8_t* add;
    const uint8_t* softmax;
    uint32_t activation;
    uint32_t weights;
    uint32_t output;
    uint32_t add_output;
    if (n < 3u) return 0;
    matmul = node_bytes(model, n - 3u);
    add = node_bytes(model, n - 2u);
    softmax = node_bytes(model, n - 1u);
    if (lwm_read_u16(matmul) != LW_OP_MATMUL || lwm_read_u16(add) != LW_OP_ADD ||
        lwm_read_u16(softmax) != LW_OP_SOFTMAX || lwm_read_u16(matmul + 2u) != 2u ||
        lwm_read_u16(add + 2u) != 2u || lwm_read_u16(softmax + 2u) != 1u) return 0;
    activation = lwm_read_u32(matmul + 8u);
    weights = lwm_read_u32(matmul + 12u);
    output = lwm_read_u32(matmul + 40u);
    add_output = lwm_read_u32(add + 40u);
    if (lwm_read_u32(add + 8u) != output || session->tensors[weights].rank != 2u) return 0;
    tail->enabled = 1u;
    tail->activation_value = activation;
    tail->weight_tensor = weights;
    tail->bias_tensor = lwm_read_u32(add + 12u);
    tail->classes = (uint32_t)session->tensors[weights].dimensions[1];
    tail->inner = (uint32_t)session->tensors[weights].dimensions[0];
    tail->rows = (uint32_t)session->tensors[activation].dimensions[session->tensors[activation].rank - 2u];
    return add_output == lwm_read_u32(softmax + 8u);
}

static void fill_value(const lw_session* session, uint32_t i, lw_x64_rec_value* value) {
    const lw_runtime_tensor* tensor = &session->tensors[i];
    uint32_t j;
    memset(value, 0, sizeof(*value));
    value->bytes = tensor->byte_size;
    value->rank = tensor->rank > 4u ? 4u : tensor->rank;
    for (j = 0u; j < value->rank; ++j) value->dimensions[j] = tensor->dimensions[j];
    value->layout = tensor->rank == 4u ? LW_X64_REC_LAYOUT_NHWC : LW_X64_REC_LAYOUT_SCALAR;
    value->producer = -1;
    value->last_use = -1;
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) {
        const uint8_t* disk = session->model->bytes + (size_t)session->model->tensor_offset +
            (size_t)i * LWM_V0_TENSOR_SIZE;
        value->constant_data = (const float*)(const void*)(session->model->bytes +
            (size_t)lwm_read_u64(disk + 48u));
    }
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model, uint32_t target_width, lw_x64_rec_program** out_program,
    lw_error* error) {
    lw_session* session = NULL;
    lw_x64_rec_program* program = NULL;
    lw_model_info info;
    uint64_t cursor = 0u;
    uint32_t i;
    lw_status status;
    if (out_program == NULL || model == NULL || target_width == 0u || target_width > INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC backend requires model, width and output");
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    *out_program = NULL;
    lw_model_info_init(&info);
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK) return LW_X64_REC_COMPILE_INVALID_GRAPH;
    status = make_shape_session(model, target_width, &session, error);
    if (status != LW_STATUS_OK) return status == LW_STATUS_OUT_OF_MEMORY ? LW_X64_REC_COMPILE_OUT_OF_MEMORY : LW_X64_REC_COMPILE_INVALID_GRAPH;
    program = (lw_x64_rec_program*)calloc(1u, sizeof(*program));
    if (program == NULL) { lw_session_free(session); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC program allocation failed"); return LW_X64_REC_COMPILE_OUT_OF_MEMORY; }
    program->model = model;
    program->cpu = lw_get_cpu_capabilities();
    if (!lw_simd_level_is_avx2(program->cpu.simd) || !program->cpu.has_avx2_fma) {
        lw_session_free(session);
        free(program);
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "standalone x64 REC backend requires AVX2 and FMA");
        return LW_X64_REC_COMPILE_UNSUPPORTED;
    }
    program->model_signature = info.content_checksum;
    program->target_width = target_width;
    program->value_count = info.tensor_count;
    program->op_count = info.node_count;
    program->direct_nhwc = 1u;
    program->ctc_fused = detect_ctc(model, session, &program->ctc) ? 1u : 0u;
    program->ops = (lw_x64_rec_op*)calloc(program->op_count, sizeof(*program->ops));
    program->values = (lw_x64_rec_value*)calloc(program->value_count, sizeof(*program->values));
    if (program->ops == NULL || program->values == NULL) { lw_session_free(session); lw_x64_rec_program_free(program); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC program tables allocation failed"); return LW_X64_REC_COMPILE_OUT_OF_MEMORY; }
    for (i = 0u; i < program->value_count; ++i) {
        fill_value(session, i, &program->values[i]);
        if ((session->tensors[i].flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u) {
            status = lw_x64_rec_arena_alloc(&cursor, session->tensors[i].byte_size, 64u, &program->values[i].offset, error);
            if (status != LW_STATUS_OK) { lw_session_free(session); lw_x64_rec_program_free(program); return LW_X64_REC_COMPILE_INVALID_GRAPH; }
            if ((session->tensors[i].flags & LWM_V0_TENSOR_FLAG_INPUT) != 0u) {
                program->input_value = i;
            }
        }
    }
    for (i = 0u; i < program->op_count; ++i) {
        const uint8_t* node = node_bytes(model, i);
        lw_x64_rec_op* op = &program->ops[i];
        uint32_t j;
        op->kind = (uint16_t)classify(lwm_read_u16(node));
        if (program->ctc_fused != 0u && i == info.node_count - 3u) op->kind = LW_X64_REC_OP_CTC_PROJECTION;
        if (program->ctc_fused != 0u && i == info.node_count - 2u) op->kind = LW_X64_REC_OP_CTC_BIAS;
        if (program->ctc_fused != 0u && i == info.node_count - 1u) op->kind = LW_X64_REC_OP_CTC_SOFTMAX;
        op->semantic_op = lwm_read_u16(node);
        op->node_index = i;
        op->input_count = lwm_read_u16(node + 2u);
        op->output = lwm_read_u32(node + 40u);
        if (op->input_count > 8u) { lw_session_free(session); lw_x64_rec_program_free(program); lw_set_error(error, LW_STATUS_INVALID_FORMAT, "REC node has too many inputs"); return LW_X64_REC_COMPILE_INVALID_GRAPH; }
        for (j = 0u; j < op->input_count; ++j) op->inputs[j] = lwm_read_u32(node + 8u + j * 4u);
        if (op->kind == LW_X64_REC_OP_GENERIC_UNSUPPORTED) ++program->generic_ops;
        if (op->output < program->value_count) {
            program->values[op->output].producer = (int32_t)i;
        }
        for (j = 0u; j < op->input_count; ++j) {
            if (op->inputs[j] < program->value_count && program->values[op->inputs[j]].last_use < (int32_t)i) {
                program->values[op->inputs[j]].last_use = (int32_t)i;
            }
        }
        if ((op->semantic_op == LW_OP_RESHAPE || op->semantic_op == LW_OP_SQUEEZE ||
             op->semantic_op == LW_OP_UNSQUEEZE) && op->input_count == 1u &&
            op->output < program->value_count) {
            program->values[op->output].alias = 1u;
            program->values[op->output].alias_of = op->inputs[0];
            program->values[op->output].offset = program->values[op->inputs[0]].offset;
        }
    }

    if (program->ctc_fused != 0u) {
        program->ctc_projection = (struct lw_x64_rec_ctc_projection*)calloc(1u, sizeof(*program->ctc_projection));
        if (program->ctc_projection == NULL || lw_x64_rec_ctc_projection_prepare(model, session, &program->ctc, program->ctc_projection, error) != LW_STATUS_OK) {
            lw_session_free(session); lw_x64_rec_program_free(program);
            if (error != NULL && error->code == LW_STATUS_OK) lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "CTC projection setup failed");
            return LW_X64_REC_COMPILE_OUT_OF_MEMORY;
        }
    }
    program->class_count = program->ctc.classes;
    program->time_steps = program->ctc.rows;
    program->output_value = info.output_count == 0u ? 0u : lwm_read_u32(model->bytes + (size_t)model->output_offset);
    program->arena_bytes = cursor;
    if (cursor != 0u) {
        program->arena = (uint8_t*)calloc(1u, (size_t)cursor);
        if (program->arena == NULL) { lw_session_free(session); lw_x64_rec_program_free(program); lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC arena allocation failed"); return LW_X64_REC_COMPILE_OUT_OF_MEMORY; }
    }
    lw_session_free(session);
    lw_set_error(error, LW_STATUS_OK, "");
    *out_program = program;
    return LW_X64_REC_COMPILE_OK;
}

void lw_x64_rec_program_free(lw_x64_rec_program* program) {
    if (program == NULL) return;
    if (program->ctc_projection != NULL) { lw_x64_rec_ctc_projection_free(program->ctc_projection); free(program->ctc_projection); }
    free(program->scratch);
    free(program->arena);
    free(program->ops);
    free(program->values);
    free(program);
}
