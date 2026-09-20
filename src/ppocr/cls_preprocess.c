#include "cls_internal.h"

/* Resize the left 4H crop window and normalize it for the direction classifier. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

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
