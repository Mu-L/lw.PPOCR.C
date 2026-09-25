#ifndef LW_REC_BACKEND_KERNELS_INTERNAL_H
#define LW_REC_BACKEND_KERNELS_INTERNAL_H

#include "../kernels/nhwc_internal.h"

typedef struct lw_rec_backend_kernels {
    const char* name;
    uint32_t vector_width;
    uint32_t oc_block;
    uint32_t has_fma;
    void (*pointwise)(const float*, const float*, const lw_nhwc_epilogue*,
                      float*, uint32_t, uint32_t, uint32_t);
    lw_status (*dense)(const float*, const float*, const lw_nhwc_epilogue*,
                       float*, const lw_nhwc_dense_desc*, void*, uint64_t);
    lw_status (*depthwise)(const float*, const float*, const float*,
                           float*, const lw_nhwc_depthwise_desc*);
} lw_rec_backend_kernels;

const lw_rec_backend_kernels* lw_rec_backend_kernels_current(void);

#endif
