#include "ctc_projection_internal.h"
#include "ctc_head_parallel_internal.h"
#include "packed_matmul_internal.h"
#include "../simd/simd_kernels.h"
#include "lwm_read.h"

#include <stdlib.h>
#include <string.h>

static const float* constant_f32(const lw_model* model, uint32_t index) {
    const uint8_t* tensor = model->bytes + (size_t)model->tensor_offset +
                            (size_t)index * LWM_V0_TENSOR_SIZE;
    return (const float*)(const void*)(model->bytes + (size_t)lwm_read_u64(tensor + 48u));
}

lw_status lw_x64_rec_ctc_prepare(const lw_model* model, const lw_session* session,
                                 lw_x64_rec_ctc_tail* tail, lw_error* error) {
    uint64_t count;
    (void)session;
    if (model == NULL || tail == NULL || !tail->enabled ||
        !lw_packed_matmul_weight_count(tail->inner, tail->classes, &count) ||
        count > SIZE_MAX / sizeof(float)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid CTC projection descriptor");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    tail->packed_weights = (float*)malloc((size_t)count * sizeof(float));
    if (tail->packed_weights == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "CTC packed weight allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    lw_pack_matmul_weights_f32(constant_f32(model, tail->weight_tensor), tail->inner,
                               tail->classes, tail->packed_weights);
    tail->bias = constant_f32(model, tail->bias_tensor);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_x64_rec_ctc_free(lw_x64_rec_ctc_tail* tail) {
    if (tail == NULL) return;
    free(tail->packed_weights);
    tail->packed_weights = NULL;
    tail->bias = NULL;
}

lw_status lw_x64_rec_ctc_execute(const lw_x64_rec_program* program,
                                 lw_x64_rec_instance* instance, const float* activation,
                                 lw_error* error) {
    if (program == NULL || instance == NULL || activation == NULL || !program->ctc.enabled ||
        instance->ctc_scores == NULL || instance->best_indices == NULL ||
        instance->best_probabilities == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid CTC runtime state");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    memset(instance->best_probabilities, 0, (size_t)program->ctc.rows * sizeof(*instance->best_probabilities));
    /* The logit tensor is never materialized: the argmax pass keeps only the
     * per-row maximum and index, and the probability pass recomputes one row
     * of logits per emitted step.  Rows are independent, so the argmax pass
     * may fan out over the borrowed intra-op pool bit-identically. */
    {
        uint32_t head_workers = lw_ctc_head_parallel_worker_count(
            instance->thread_pool, instance->intra_op_workers, program->ctc.rows,
            program->ctc.inner, program->ctc.classes);
        lw_ctc_head_argmax_scores_parallel_f32(
            instance->thread_pool, head_workers, activation, program->ctc.packed_weights,
            program->ctc.bias, instance->best_indices, instance->ctc_scores, program->ctc.rows,
            program->ctc.inner, program->ctc.classes);
    }
    lw_avx2_ctc_row_probabilities_f32(
        activation, program->ctc.packed_weights, program->ctc.bias,
        instance->best_indices, instance->ctc_scores, instance->best_probabilities,
        program->ctc.rows, program->ctc.inner, program->ctc.classes);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
