#ifndef LW_RESIZE_FIXED_INTERNAL_H
#define LW_RESIZE_FIXED_INTERNAL_H

/* Shared 11-bit fixed-point OpenCV INTER_LINEAR replica used by the DET and
 * CLS preprocessing paths. The coordinate math stays in double (OpenCV
 * softdouble parity) and the coefficients use round-half-even
 * (CvRoundToShort). */

#include <math.h>
#include <stdint.h>

#define LW_FIXED_RESIZE_SCALE 2048

static int16_t lw_fixed_round_to_short(double value) {
    double rounded = floor(value + 0.5);
    if (rounded - value == 0.5 && ((int64_t)rounded & 1) != 0) {
        rounded -= 1.0;
    }
    return (int16_t)rounded;
}

/* Per-destination-column coefficients with edge clamping: both source
 * indices clamp, and the fraction zeroes at the edges. */
static void lw_fixed_build_linear_coefficients(uint32_t source_size, uint32_t destination_size,
                                               int32_t* offsets, int16_t* coefficients) {
    uint32_t d;
    for (d = 0u; d < destination_size; ++d) {
        double coordinate = ((double)d + 0.5) * source_size / destination_size - 0.5;
        int32_t source_index = (int32_t)floor(coordinate);
        double fraction = coordinate - source_index;
        if (source_index < 0) {
            source_index = 0;
            fraction = 0.0;
        }
        if (source_index >= (int32_t)source_size - 1) {
            source_index = (int32_t)source_size - 1;
            fraction = 0.0;
        }
        offsets[d] = source_index;
        coefficients[(size_t)d * 2u] =
            lw_fixed_round_to_short((1.0 - fraction) * LW_FIXED_RESIZE_SCALE);
        coefficients[(size_t)d * 2u + 1u] =
            lw_fixed_round_to_short(fraction * LW_FIXED_RESIZE_SCALE);
    }
}

/* Vertical coordinate: raw floor/fraction without edge clamping (both source
 * rows point at the edge pixel but the two fixed-point products are still
 * evaluated separately). */
static void lw_fixed_get_linear_coordinate(uint32_t destination, uint32_t source_size,
                                           uint32_t destination_size, int32_t* source_index,
                                           int16_t* coefficient0, int16_t* coefficient1) {
    double coordinate = ((double)destination + 0.5) * source_size / destination_size - 0.5;
    *source_index = (int32_t)floor(coordinate);
    {
        double fraction = coordinate - *source_index;
        *coefficient0 = lw_fixed_round_to_short((1.0 - fraction) * LW_FIXED_RESIZE_SCALE);
        *coefficient1 = lw_fixed_round_to_short(fraction * LW_FIXED_RESIZE_SCALE);
    }
}

/* One horizontal pass into int32 row scratch (BGR24 source). */
static void lw_fixed_build_horizontal_row(const uint8_t* source, uint32_t source_stride,
                                          uint32_t source_width, uint32_t source_y,
                                          uint32_t destination_width, const int32_t* offsets,
                                          const int16_t* coefficients, int32_t* destination) {
    const uint8_t* row = source + (size_t)source_y * source_stride;
    uint32_t x;
    for (x = 0u; x < destination_width; ++x) {
        int32_t sx = offsets[x];
        int32_t sx1 = sx + 1;
        int16_t coefficient0 = coefficients[(size_t)x * 2u];
        int16_t coefficient1 = coefficients[(size_t)x * 2u + 1u];
        if (sx1 >= (int32_t)source_width) sx1 = (int32_t)source_width - 1;
        {
            size_t source_offset = (size_t)sx * 3u;
            size_t source_offset1 = (size_t)sx1 * 3u;
            size_t destination_offset = (size_t)x * 3u;
            destination[destination_offset] = (int32_t)row[source_offset] * coefficient0 +
                                               (int32_t)row[source_offset1] * coefficient1;
            destination[destination_offset + 1u] =
                (int32_t)row[source_offset + 1u] * coefficient0 +
                (int32_t)row[source_offset1 + 1u] * coefficient1;
            destination[destination_offset + 2u] =
                (int32_t)row[source_offset + 2u] * coefficient0 +
                (int32_t)row[source_offset1 + 2u] * coefficient1;
        }
    }
}

/* VResizeLinearVec_32s8u replica: vertical fixed-point blend of two
 * horizontal accumulators, clamped to [0, 255]. */
static int32_t lw_fixed_blend(int32_t h0, int32_t h1, int16_t beta0, int16_t beta1) {
    int32_t value = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return value;
}

#endif
