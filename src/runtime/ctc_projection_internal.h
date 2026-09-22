#ifndef LW_X64_REC_CTC_PROJECTION_INTERNAL_H
#define LW_X64_REC_CTC_PROJECTION_INTERNAL_H

#include "x64_rec_backend_internal.h"

#include <stdint.h>

typedef struct lw_x64_rec_ctc_projection {
    uint8_t enabled;
    uint8_t reserved[3];
    uint32_t rows;
    uint32_t inner;
    uint32_t classes;
    float* packed_weights;
    const float* bias;
    float* logits;
    float* emitted_probabilities;
    uint32_t* best_indices;
} lw_x64_rec_ctc_projection;

lw_status lw_x64_rec_ctc_projection_prepare(const lw_model* model, const lw_session* session,
                                            const lw_x64_rec_ctc_tail* tail,
                                            lw_x64_rec_ctc_projection* projection,
                                            lw_error* error);
void lw_x64_rec_ctc_projection_free(lw_x64_rec_ctc_projection* projection);
lw_status lw_x64_rec_ctc_projection_run(const lw_x64_rec_ctc_projection* projection,
                                        const float* activation, lw_error* error);

#endif
