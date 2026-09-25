#include "ctc_head_parallel_internal.h"

#include "../simd/simd_kernels.h"

#include <stddef.h>

/* Minimum total MACs before worker fan-out pays for synchronization. */
#define LW_CTC_HEAD_PARALLEL_MIN_WORK (4ull * 1000ull * 1000ull)

uint32_t lw_ctc_head_parallel_worker_count(const lw_thread_pool* pool,
                                           uint32_t max_workers, uint32_t rows,
                                           uint32_t inner_dimension, uint32_t columns) {
    uint32_t pool_workers;
    uint64_t work;
    if (pool == NULL || max_workers <= 1u || rows < 8u) {
        return 1u;
    }
    work = (uint64_t)rows * inner_dimension * columns;
    if (work < LW_CTC_HEAD_PARALLEL_MIN_WORK) {
        return 1u;
    }
    pool_workers = lw_thread_pool_worker_count(pool);
    if (max_workers > pool_workers) {
        max_workers = pool_workers;
    }
    /* Keep at least one 4-row block per worker. */
    if (max_workers > rows / 4u) {
        max_workers = rows / 4u;
    }
    return max_workers <= 1u ? 1u : max_workers;
}

/* Row ranges are split in units of 4-row blocks: the packed kernels process
 * rows in blocks of 4 and silently drop a tail that is not a multiple of 4,
 * so slice boundaries must stay block-aligned. */
static uint32_t row_range_begin(uint32_t rows, uint32_t worker_count, uint32_t worker_index) {
    uint32_t blocks = rows / 4u;
    uint32_t base = blocks / worker_count;
    uint32_t remainder = blocks % worker_count;
    return (worker_index * base + (worker_index < remainder ? worker_index : remainder)) * 4u;
}

typedef struct lw_ctc_head_argmax_context {
    lw_rec_ctc_argmax_scores_fn kernel;
    const float* input;
    const float* packed_weights;
    const float* bias;
    uint32_t* best_indices;
    float* scores;
    uint32_t rows;
    uint32_t inner_dimension;
    uint32_t columns;
} lw_ctc_head_argmax_context;

static void execute_argmax_scores_rows(void* context_void, uint32_t worker_index,
                                       uint32_t worker_count) {
    lw_ctc_head_argmax_context* context = (lw_ctc_head_argmax_context*)context_void;
    uint32_t row_begin = row_range_begin(context->rows, worker_count, worker_index);
    uint32_t row_end = row_range_begin(context->rows, worker_count, worker_index + 1u);
    if (row_end <= row_begin) {
        return;
    }
    /* Rows are independent and each output element accumulates in the serial
     * kernel's order, so the row split is bit-identical. */
    context->kernel(
        context->input + (size_t)row_begin * context->inner_dimension,
        context->packed_weights, context->bias, context->best_indices + row_begin,
        context->scores + row_begin, row_end - row_begin, context->inner_dimension,
        context->columns);
}

void lw_ctc_head_argmax_scores_parallel_f32(lw_thread_pool* pool, uint32_t worker_count,
                                            lw_rec_ctc_argmax_scores_fn kernel,
                                            const float* input, const float* packed_weights,
                                            const float* bias, uint32_t* best_indices,
                                            float* scores, uint32_t rows,
                                            uint32_t inner_dimension, uint32_t columns) {
    lw_ctc_head_argmax_context context;
    if (pool == NULL || worker_count <= 1u) {
        kernel(input, packed_weights, bias, best_indices, scores, rows,
               inner_dimension, columns);
        return;
    }
    context.kernel = kernel;
    context.input = input;
    context.packed_weights = packed_weights;
    context.bias = bias;
    context.best_indices = best_indices;
    context.scores = scores;
    context.rows = rows;
    context.inner_dimension = inner_dimension;
    context.columns = columns;
    lw_thread_pool_run(pool, worker_count, execute_argmax_scores_rows, &context);
}

typedef struct lw_ctc_head_bias_argmax_context {
    const float* input;
    const float* packed_weights;
    const float* bias;
    float* output;
    uint32_t* best_indices;
    uint32_t rows;
    uint32_t inner_dimension;
    uint32_t columns;
} lw_ctc_head_bias_argmax_context;

static void execute_bias_argmax_rows(void* context_void, uint32_t worker_index,
                                     uint32_t worker_count) {
    lw_ctc_head_bias_argmax_context* context = (lw_ctc_head_bias_argmax_context*)context_void;
    uint32_t row_begin = row_range_begin(context->rows, worker_count, worker_index);
    uint32_t row_end = row_range_begin(context->rows, worker_count, worker_index + 1u);
    if (row_end <= row_begin) {
        return;
    }
    lw_avx2_packed_matmul_bias_argmax_f32(
        context->input + (size_t)row_begin * context->inner_dimension, context->packed_weights,
        context->bias, context->output + (size_t)row_begin * context->columns,
        context->best_indices + row_begin, 1u, row_end - row_begin, context->inner_dimension,
        context->columns);
}

void lw_ctc_head_bias_argmax_parallel_f32(lw_thread_pool* pool, uint32_t worker_count,
                                          const float* input, const float* packed_weights,
                                          const float* bias, float* output,
                                          uint32_t* best_indices, uint32_t rows,
                                          uint32_t inner_dimension, uint32_t columns) {
    lw_ctc_head_bias_argmax_context context;
    if (pool == NULL || worker_count <= 1u) {
        lw_avx2_packed_matmul_bias_argmax_f32(input, packed_weights, bias, output, best_indices,
                                              1u, rows, inner_dimension, columns);
        return;
    }
    context.input = input;
    context.packed_weights = packed_weights;
    context.bias = bias;
    context.output = output;
    context.best_indices = best_indices;
    context.rows = rows;
    context.inner_dimension = inner_dimension;
    context.columns = columns;
    lw_thread_pool_run(pool, worker_count, execute_bias_argmax_rows, &context);
}
