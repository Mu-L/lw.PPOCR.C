#include "cls_internal.h"
#include "resize_fixed_internal.h"

/* Resize the left 4H crop window and normalize it for the direction classifier. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t clamp_coordinate(int64_t coordinate, uint32_t limit) {
    if (coordinate < 0) {
        return 0u;
    }
    if ((uint64_t)coordinate >= limit) {
        return limit - 1u;
    }
    return (uint32_t)coordinate;
}

lw_status lw_cls_preprocess_bgr_u8(const uint8_t* source, uint64_t source_byte_count,
                                   uint32_t source_width, uint32_t source_height,
                                   uint32_t source_stride, float* output,
                                   uint64_t output_element_count, uint32_t* resized_width) {
    const uint64_t required_output_elements =
        (uint64_t)3u * LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    const double normalize_scale = 2.0 / 255.0;
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint64_t source_window_width;
    uint64_t channel_plane;
    uint32_t channel;
    uint32_t output_y;

    if (resized_width != NULL) {
        *resized_width = 0u;
    }
    if (source == NULL || output == NULL || source_width == 0u || source_height == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes) {
        return LW_STATUS_INVALID_SHAPE;
    }
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    if (required_source_bytes > SIZE_MAX || source_byte_count < required_source_bytes ||
        output_element_count != required_output_elements ||
        required_output_elements > SIZE_MAX / sizeof(float)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    source_window_width = (uint64_t)source_height * LW_CLS_SOURCE_WINDOW_MAX_WIDTH_PER_HEIGHT;
    if (source_window_width > source_width) {
        source_window_width = source_width;
    }
    if (source_window_width == 0u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    channel_plane = (uint64_t)LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;

    for (output_y = 0u; output_y < LW_CLS_INPUT_HEIGHT; ++output_y) {
        const double source_y =
            ((double)output_y + 0.5) * source_height / LW_CLS_INPUT_HEIGHT - 0.5;
        const int64_t source_y0_raw = (int64_t)floor(source_y);
        const int64_t source_y1_raw = source_y0_raw + 1;
        const uint32_t source_y0 = clamp_coordinate(source_y0_raw, source_height);
        const uint32_t source_y1 = clamp_coordinate(source_y1_raw, source_height);
        const double weight_y = source_y - (double)source_y0_raw;
        uint32_t output_x;
        for (output_x = 0u; output_x < LW_CLS_INPUT_WIDTH; ++output_x) {
            const double source_x =
                ((double)output_x + 0.5) * source_window_width / LW_CLS_INPUT_WIDTH - 0.5;
            const int64_t source_x0_raw = (int64_t)floor(source_x);
            const int64_t source_x1_raw = source_x0_raw + 1;
            const uint32_t source_x0 = clamp_coordinate(source_x0_raw, (uint32_t)source_window_width);
            const uint32_t source_x1 = clamp_coordinate(source_x1_raw, (uint32_t)source_window_width);
            const double weight_x = source_x - (double)source_x0_raw;
            for (channel = 0u; channel < 3u; ++channel) {
                const double top_left = source[(size_t)((uint64_t)source_y0 * source_stride +
                                                        (uint64_t)source_x0 * 3u + channel)];
                const double top_right = source[(size_t)((uint64_t)source_y0 * source_stride +
                                                         (uint64_t)source_x1 * 3u + channel)];
                const double bottom_left =
                    source[(size_t)((uint64_t)source_y1 * source_stride +
                                    (uint64_t)source_x0 * 3u + channel)];
                const double bottom_right =
                    source[(size_t)((uint64_t)source_y1 * source_stride +
                                    (uint64_t)source_x1 * 3u + channel)];
                const double top = top_left + (top_right - top_left) * weight_x;
                const double bottom = bottom_left + (bottom_right - bottom_left) * weight_x;
                const double value = top + (bottom - top) * weight_y;
                const uint64_t output_index =
                    (uint64_t)channel * channel_plane +
                    (uint64_t)output_y * LW_CLS_INPUT_WIDTH + output_x;
                output[(size_t)output_index] = (float)(value * normalize_scale - 1.0);
            }
        }
    }
    if (resized_width != NULL) {
        *resized_width = LW_CLS_INPUT_WIDTH;
    }
    return LW_STATUS_OK;
}

static void cls_fixed_workspace_ensure(lw_cls_preprocess_workspace* workspace,
                                       uint32_t width) {
    if (workspace->width != width) {
        int32_t* x_offsets = (int32_t*)malloc((size_t)width * sizeof(*x_offsets));
        int16_t* x_coefficients = (int16_t*)malloc((size_t)width * 2u * sizeof(*x_coefficients));
        int32_t* row0 = (int32_t*)malloc((size_t)width * 3u * sizeof(*row0));
        int32_t* row1 = (int32_t*)malloc((size_t)width * 3u * sizeof(*row1));
        if (x_offsets == NULL || x_coefficients == NULL || row0 == NULL || row1 == NULL) {
            free(row1);
            free(row0);
            free(x_coefficients);
            free(x_offsets);
            return;
        }
        free(workspace->row1);
        free(workspace->row0);
        free(workspace->x_coefficients);
        free(workspace->x_offsets);
        workspace->x_offsets = x_offsets;
        workspace->x_coefficients = x_coefficients;
        workspace->row0 = row0;
        workspace->row1 = row1;
        workspace->width = width;
    }
}

void lw_cls_preprocess_workspace_init(lw_cls_preprocess_workspace* workspace) {
    if (workspace != NULL) memset(workspace, 0, sizeof(*workspace));
}

void lw_cls_preprocess_workspace_free(lw_cls_preprocess_workspace* workspace) {
    if (workspace == NULL) return;
    free(workspace->row1);
    free(workspace->row0);
    free(workspace->x_coefficients);
    free(workspace->x_offsets);
    memset(workspace, 0, sizeof(*workspace));
}

static lw_status cls_preprocess_bgr_u8_fixed_layout(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, float* output,
    uint64_t output_element_count, uint32_t* resized_width,
    lw_cls_preprocess_workspace* workspace, int nhwc) {
    const uint64_t required_output_elements =
        (uint64_t)3u * LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    uint64_t row_bytes;
    uint64_t required_source_bytes;
    uint32_t window_width;
    uint32_t actual_width;
    uint32_t channel_plane;
    uint32_t oy;
    uint32_t ox;
    uint32_t channel;
    float rec_lut[256];
    if (resized_width != NULL) {
        *resized_width = 0u;
    }
    if (source == NULL || output == NULL || workspace == NULL || source_width == 0u ||
        source_height == 0u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    row_bytes = (uint64_t)source_width * 3u;
    if (row_bytes > UINT32_MAX || source_stride < row_bytes ||
        (uint64_t)(source_height - 1u) * source_stride > UINT64_MAX - row_bytes) {
        return LW_STATUS_INVALID_SHAPE;
    }
    required_source_bytes = (uint64_t)(source_height - 1u) * source_stride + row_bytes;
    if (required_source_bytes > SIZE_MAX || source_byte_count < required_source_bytes ||
        output_element_count != required_output_elements ||
        required_output_elements > SIZE_MAX / sizeof(float)) {
        return LW_STATUS_INVALID_SHAPE;
    }
    /* Left 4:1 window on the original buffer (no allocation, stride
     * unchanged), then keep-aspect to height 80 with trailing -1 columns. */
    window_width = source_width;
    if ((uint64_t)LW_CLS_INPUT_HEIGHT * window_width >
        (uint64_t)(LW_CLS_INPUT_WIDTH * 2u) * source_height) {
        window_width = (uint32_t)(((uint64_t)(LW_CLS_INPUT_WIDTH * 2u) * source_height) /
                                  LW_CLS_INPUT_HEIGHT);
    }
    if (window_width == 0u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    actual_width = (uint32_t)(((uint64_t)LW_CLS_INPUT_HEIGHT * window_width +
                               source_height - 1u) /
                              source_height);
    if (actual_width > LW_CLS_INPUT_WIDTH) actual_width = LW_CLS_INPUT_WIDTH;
    if (actual_width < LW_CLS_INPUT_WIDTH) {
        uint64_t count = required_output_elements;
        uint64_t i;
        for (i = 0u; i < count; ++i) output[(size_t)i] = -1.0f;
    }
    channel_plane = (uint64_t)LW_CLS_INPUT_HEIGHT * LW_CLS_INPUT_WIDTH;
    cls_fixed_workspace_ensure(workspace, actual_width);
    if (workspace->x_offsets == NULL || workspace->row0 == NULL) {
        return LW_STATUS_OUT_OF_MEMORY;
    }
    lw_fixed_build_linear_coefficients(window_width, actual_width, workspace->x_offsets,
                                       workspace->x_coefficients);
    for (ox = 0u; ox < 256u; ++ox) {
        rec_lut[ox] = (float)((double)ox * (2.0 / 255.0) - 1.0);
    }
    for (oy = 0u; oy < LW_CLS_INPUT_HEIGHT; ++oy) {
        int32_t source_y;
        int16_t beta0;
        int16_t beta1;
        uint32_t source_y0;
        uint32_t source_y1;
        uint32_t destination = oy * LW_CLS_INPUT_WIDTH;
        lw_fixed_get_linear_coordinate(oy, source_height, LW_CLS_INPUT_HEIGHT, &source_y,
                                       &beta0, &beta1);
        source_y0 = source_y < 0
                        ? 0u
                        : ((uint32_t)source_y >= source_height ? source_height - 1u
                                                               : (uint32_t)source_y);
        {
            int32_t clamped = source_y + 1;
            source_y1 = clamped < 0
                            ? 0u
                            : ((uint32_t)clamped >= source_height ? source_height - 1u
                                                                  : (uint32_t)clamped);
        }
        lw_fixed_build_horizontal_row(source, source_stride, window_width, source_y0,
                                      actual_width, workspace->x_offsets,
                                      workspace->x_coefficients, workspace->row0);
        lw_fixed_build_horizontal_row(source, source_stride, window_width, source_y1,
                                      actual_width, workspace->x_offsets,
                                      workspace->x_coefficients, workspace->row1);
        for (ox = 0u; ox < actual_width; ++ox) {
            uint32_t row_offset = ox * 3u;
            for (channel = 0u; channel < 3u; ++channel) {
                int32_t value = lw_fixed_blend(workspace->row0[row_offset + channel],
                                               workspace->row1[row_offset + channel], beta0,
                                               beta1);
                size_t output_index = nhwc != 0
                    ? ((size_t)destination + ox) * 3u + channel
                    : (size_t)channel * channel_plane + destination + ox;
                output[output_index] = rec_lut[value];
            }
        }
    }
    if (resized_width != NULL) {
        *resized_width = actual_width;
    }
    return LW_STATUS_OK;
}

lw_status lw_cls_preprocess_bgr_u8_fixed(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, float* output,
    uint64_t output_element_count, uint32_t* resized_width,
    lw_cls_preprocess_workspace* workspace) {
    return cls_preprocess_bgr_u8_fixed_layout(
        source, source_byte_count, source_width, source_height, source_stride,
        output, output_element_count, resized_width, workspace, 0);
}

lw_status lw_cls_preprocess_bgr_u8_fixed_nhwc(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, float* output,
    uint64_t output_element_count, uint32_t* resized_width,
    lw_cls_preprocess_workspace* workspace) {
    return cls_preprocess_bgr_u8_fixed_layout(
        source, source_byte_count, source_width, source_height, source_stride,
        output, output_element_count, resized_width, workspace, 1);
}
