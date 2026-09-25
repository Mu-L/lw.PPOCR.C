#ifndef LW_X64_REC_CTC_PROJECTION_INTERNAL_H
#define LW_X64_REC_CTC_PROJECTION_INTERNAL_H

#include "x64_rec_backend_internal.h"

lw_status lw_x64_rec_ctc_prepare(const lw_model* model, const lw_session* session,
                                 lw_x64_rec_ctc_tail* tail, lw_error* error);
lw_status lw_x64_rec_ctc_prepare_shared(const lw_model* model, const lw_session* session,
                                        const lw_x64_rec_ctc_tail* source,
                                        lw_x64_rec_ctc_tail* tail, lw_error* error);
void lw_x64_rec_ctc_free(lw_x64_rec_ctc_tail* tail);
lw_status lw_x64_rec_ctc_execute(const lw_x64_rec_program* program,
                                 lw_x64_rec_instance* instance, const float* activation,
                                 lw_error* error);

#endif
