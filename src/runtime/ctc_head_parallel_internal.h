#ifndef LW_CTC_HEAD_PARALLEL_INTERNAL_H
#define LW_CTC_HEAD_PARALLEL_INTERNAL_H

/* Row-parallel wrappers for the CTC head packed-matmul kernels.  Each output
 * element is accumulated in exactly the same order as in the serial kernel
 * (rows are independent), so results are bit-identical; only the row range
 * is split across pool workers. */

#include "parallel_internal.h"
#include "rec_backend_kernels_internal.h"

#include <stdint.h>

/* Effective worker count for a CTC head projection: 1 (serial) when the pool
 * is missing, the cap is 1, the row count is tiny, or the total MAC count is
 * too small to amortize synchronization. */
uint32_t lw_ctc_head_parallel_worker_count(const lw_thread_pool* pool,
                                           uint32_t max_workers, uint32_t rows,
                                           uint32_t inner_dimension, uint32_t columns);

void lw_ctc_head_argmax_scores_parallel_f32(lw_thread_pool* pool, uint32_t worker_count,
                                            lw_rec_ctc_argmax_scores_fn kernel,
                                            const float* input, const float* packed_weights,
                                            const float* bias, uint32_t* best_indices,
                                            float* scores, uint32_t rows,
                                            uint32_t inner_dimension, uint32_t columns);

void lw_ctc_head_bias_argmax_parallel_f32(lw_thread_pool* pool, uint32_t worker_count,
                                          const float* input, const float* packed_weights,
                                          const float* bias, float* output,
                                          uint32_t* best_indices, uint32_t rows,
                                          uint32_t inner_dimension, uint32_t columns);

#endif
