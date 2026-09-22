#include "det_internal.h"
#include "resize_fixed_internal.h"

/* OpenCV-parity fixed-point DET resize + ImageNet LUT normalization. The
 * 11-bit coefficient path reproduces OpenCV's INTER_LINEAR 8-bit result
 * before normalization, so threshold-boundary detector pixels agree with the
 * reference implementation. Destination rows are independent; with enough
 * rows per worker they are sharded over the detector's intra-op pool. */

#include "parallel_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LW_FIXED_MIN_ROWS_PER_WORKER 24u

typedef struct fixed_row_context {
    const uint8_t* source;
    uint32_t source_stride;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t resized_width;
    uint32_t resized_height;
    const int32_t* x_offsets;
    const int16_t* x_coefficients;
    int32_t* row_scratch; /* per-worker pairs, strided */
    const float* lut;
    float* output;
    uint32_t plane;
} fixed_row_context;

static const double det_mean[3] = {0.485, 0.456, 0.406};
static const double det_inverse_std[3] = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};

/* One destination row: two horizontal passes into the worker row scratch,
 * then the vertical fixed-point blend and the normalization LUT. */
static void fixed_row(const fixed_row_context* context, uint32_t oy, int32_t* row0,
                      int32_t* row1) {
    int32_t source_y;
    int16_t beta0;
    int16_t beta1;
    uint32_t source_y0;
    uint32_t source_y1;
    uint32_t ox;
    uint32_t destination = oy * context->resized_width;
    lw_fixed_get_linear_coordinate(oy, context->source_height, context->resized_height,
                                   &source_y, &beta0, &beta1);
    source_y0 = source_y < 0 ? 0u
                             : ((uint32_t)source_y >= context->source_height
                                    ? context->source_height - 1u
                                    : (uint32_t)source_y);
    {
        int32_t clamped = source_y + 1;
        source_y1 = clamped < 0 ? 0u
                                : ((uint32_t)clamped >= context->source_height
                                       ? context->source_height - 1u
                                       : (uint32_t)clamped);
    }
    lw_fixed_build_horizontal_row(context->source, context->source_stride, context->source_width,
                                  source_y0, context->resized_width, context->x_offsets,
                                  context->x_coefficients, row0);
    lw_fixed_build_horizontal_row(context->source, context->source_stride, context->source_width,
                                  source_y1, context->resized_width, context->x_offsets,
                                  context->x_coefficients, row1);
    for (ox = 0u; ox < context->resized_width; ++ox) {
        uint32_t channel;
        uint32_t row_offset = ox * 3u;
        for (channel = 0u; channel < 3u; ++channel) {
            int32_t value = lw_fixed_blend(row0[row_offset + channel], row1[row_offset + channel],
                                           beta0, beta1);
            context->output[(size_t)channel * context->plane + destination + ox] =
                context->lut[(size_t)channel * 256u + (uint32_t)value];
        }
    }
}

static void fixed_row_worker(void* context_void, uint32_t worker_index, uint32_t worker_count) {
    fixed_row_context* context = (fixed_row_context*)context_void;
    int32_t* row0 = context->row_scratch +
                    (size_t)worker_index * 2u * context->resized_width * 3u;
    int32_t* row1 = row0 + (size_t)context->resized_width * 3u;
    uint32_t oy;
    (void)worker_count;
    for (oy = worker_index; oy < context->resized_height; oy += worker_count) {
        fixed_row(context, oy, row0, row1);
    }
}

static void workspace_ensure(lw_det_preprocess_workspace* workspace, uint32_t width,
                             uint32_t workers) {
    uint32_t slot_count = workers + 1u;
    if (workspace->width != width || workspace->row_slot_count < slot_count) {
        int32_t* x_offsets = (int32_t*)malloc((size_t)width * sizeof(*x_offsets));
        int16_t* x_coefficients = (int16_t*)malloc((size_t)width * 2u * sizeof(*x_coefficients));
        int32_t* rows = (int32_t*)malloc((size_t)slot_count * 2u * width * 3u * sizeof(*rows));
        if (x_offsets == NULL || x_coefficients == NULL || rows == NULL) {
            free(rows);
            free(x_coefficients);
            free(x_offsets);
            return;
        }
        free(workspace->rows);
        free(workspace->x_coefficients);
        free(workspace->x_offsets);
        workspace->x_offsets = x_offsets;
        workspace->x_coefficients = x_coefficients;
        workspace->rows = rows;
        workspace->width = width;
        workspace->row_slot_count = slot_count;
    }
}

static void workspace_free(lw_det_preprocess_workspace* workspace) {
    if (workspace == NULL) return;
    free(workspace->rows);
    free(workspace->x_coefficients);
    free(workspace->x_offsets);
    memset(workspace, 0, sizeof(*workspace));
}

/* The 768-entry ImageNet LUT is built with the same double expressions as
 * the legacy path, so normalization stays bit-identical for any byte; only
 * the resize rounding differs (OpenCV parity). */
static void build_normalized_lut(float* lut) {
    uint32_t channel;
    for (channel = 0u; channel < 3u; ++channel) {
        uint32_t value;
        for (value = 0u; value < 256u; ++value) {
            lut[(size_t)channel * 256u + value] =
                (float)(((double)value / 255.0 - det_mean[channel]) *
                        det_inverse_std[channel]);
        }
    }
}

lw_status lw_det_preprocess_bgr_u8_fixed(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, uint32_t resized_width,
    uint32_t resized_height, float* output, uint64_t output_element_count,
    lw_det_preprocess_workspace* workspace, lw_thread_pool* pool,
    uint32_t intra_op_thread_count) {
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint64_t plane;
    uint64_t required_output_elements;
    uint32_t workers = 1u;
    fixed_row_context context;
    float lut[3u * 256u];
    if (source == NULL || output == NULL || workspace == NULL || source_width == 0u ||
        source_height == 0u || resized_width == 0u || resized_height == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes) {
        return LW_STATUS_INVALID_SHAPE;
    }
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    plane = (uint64_t)resized_width * resized_height;
    if (plane > UINT64_MAX / 3u) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    required_output_elements = plane * 3u;
    if (required_source_bytes > SIZE_MAX || source_byte_count < required_source_bytes ||
        output_element_count != required_output_elements ||
        required_output_elements > SIZE_MAX / sizeof(float)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    if (pool != NULL && intra_op_thread_count > 1u &&
        resized_height / LW_FIXED_MIN_ROWS_PER_WORKER >= 2u) {
        workers = intra_op_thread_count;
        if (workers > resized_height / LW_FIXED_MIN_ROWS_PER_WORKER) {
            workers = resized_height / LW_FIXED_MIN_ROWS_PER_WORKER;
        }
        if (workers < 2u) workers = 1u;
    }
    /* Serial pre-allocation: the parallel body must not grow shared buffers. */
    workspace_ensure(workspace, resized_width, workers);
    if (workspace->x_offsets == NULL || workspace->rows == NULL) {
        return LW_STATUS_OUT_OF_MEMORY;
    }
    lw_fixed_build_linear_coefficients(source_width, resized_width, workspace->x_offsets,
                                       workspace->x_coefficients);
    build_normalized_lut(lut);
    context.source = source;
    context.source_stride = source_stride;
    context.source_width = source_width;
    context.source_height = source_height;
    context.resized_width = resized_width;
    context.resized_height = resized_height;
    context.x_offsets = workspace->x_offsets;
    context.x_coefficients = workspace->x_coefficients;
    context.row_scratch = workspace->rows;
    context.lut = lut;
    context.output = output;
    context.plane = (uint32_t)plane;
    if (workers > 1u) {
        uint32_t worker;
        lw_thread_pool_run(pool, workers, fixed_row_worker, &context);
        (void)worker;
    } else {
        fixed_row_worker(&context, 0u, 1u);
    }
    return LW_STATUS_OK;
}

void lw_det_preprocess_workspace_free(lw_det_preprocess_workspace* workspace) {
    workspace_free(workspace);
}

void lw_det_preprocess_workspace_init(lw_det_preprocess_workspace* workspace) {
    if (workspace != NULL) memset(workspace, 0, sizeof(*workspace));
}
