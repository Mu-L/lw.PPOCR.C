#include "ctc_projection_internal.h"
#include "lwm_read.h"
#include "packed_matmul_internal.h"
#include "../simd/simd_kernels.h"

#include <stdlib.h>
#include <string.h>

static const float* constant_f32(const lw_model* model, uint32_t index) {
    const uint8_t* tensor = model->bytes + (size_t)model->tensor_offset + (size_t)index * LWM_V0_TENSOR_SIZE;
    return (const float*)(const void*)(model->bytes + (size_t)lwm_read_u64(tensor + 48u));
}

lw_status lw_x64_rec_ctc_projection_prepare(const lw_model* model, const lw_session* session,
                                            const lw_x64_rec_ctc_tail* tail,
                                            lw_x64_rec_ctc_projection* projection,
                                            lw_error* error) {
    uint64_t packed_count;
    if (model == NULL || session == NULL || tail == NULL || projection == NULL || !tail->enabled ||
        tail->rows == 0u || tail->inner == 0u || tail->classes == 0u ||
        !lw_packed_matmul_weight_count(tail->inner, tail->classes, &packed_count)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid CTC projection descriptor");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    memset(projection, 0, sizeof(*projection));
    projection->rows = tail->rows;
    projection->inner = tail->inner;
    projection->classes = tail->classes;
    projection->packed_weights = (float*)malloc((size_t)packed_count * sizeof(float));
    projection->logits = (float*)malloc((size_t)tail->rows * tail->classes * sizeof(float));
    projection->emitted_probabilities = (float*)malloc((size_t)tail->rows * sizeof(float));
    projection->best_indices = (uint32_t*)malloc((size_t)tail->rows * sizeof(uint32_t));
    if (projection->packed_weights == NULL || projection->logits == NULL ||
        projection->emitted_probabilities == NULL || projection->best_indices == NULL) {
        lw_x64_rec_ctc_projection_free(projection);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "CTC projection allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    lw_pack_matmul_weights_f32(constant_f32(model, tail->weight_tensor), tail->inner,
                               tail->classes, projection->packed_weights);
    projection->bias = constant_f32(model, tail->bias_tensor);
    projection->enabled = 1u;
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

void lw_x64_rec_ctc_projection_free(lw_x64_rec_ctc_projection* projection) {
    if (projection == NULL) return;
    free(projection->packed_weights);
    free(projection->logits);
    free(projection->emitted_probabilities);
    free(projection->best_indices);
    memset(projection, 0, sizeof(*projection));
}

lw_status lw_x64_rec_ctc_projection_run(const lw_x64_rec_ctc_projection* projection,
                                        const float* activation, lw_error* error) {
    if (projection == NULL || !projection->enabled || activation == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid CTC projection input");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    lw_avx2_fma_packed_matmul_bias_argmax_f32(
        activation, projection->packed_weights, projection->bias, projection->logits,
        projection->best_indices, 1u, projection->rows, projection->inner, projection->classes);
    lw_avx2_ctc_emitted_softmax_contiguous_f32(
        projection->logits, projection->best_indices, projection->emitted_probabilities,
        projection->rows, projection->classes);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}
