#ifndef LW_CLS_INTERNAL_H
#define LW_CLS_INTERNAL_H

/* Private classifier preprocessing contract. */

#include "lw_infer.h"

#include <stdint.h>

#define LW_CLS_INPUT_HEIGHT 80u
#define LW_CLS_INPUT_WIDTH 160u
#define LW_CLS_CLASS_COUNT 2u
#define LW_CLS_SOURCE_WINDOW_MAX_WIDTH_PER_HEIGHT 4u

lw_status lw_cls_preprocess_bgr_u8(const uint8_t* source, uint64_t source_byte_count,
                                   uint32_t source_width, uint32_t source_height,
                                   uint32_t source_stride, float* output,
                                   uint64_t output_element_count, uint32_t* resized_width);

/* Reference keep-aspect variant: resize the left 4H window to
 * min(160, ceil(80*w/h)) x 80 with the OpenCV-parity fixed-point path, pad
 * unused trailing columns with -1, and normalize through the RecNorm LUT. */
typedef struct lw_cls_preprocess_workspace {
    int32_t* x_offsets;
    int16_t* x_coefficients;
    int32_t* row0;
    int32_t* row1;
    uint32_t width;
} lw_cls_preprocess_workspace;

lw_status lw_cls_preprocess_bgr_u8_fixed(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, float* output,
    uint64_t output_element_count, uint32_t* resized_width,
    lw_cls_preprocess_workspace* workspace);

/* Same fixed-point samples and LUT as the NCHW variant, directly interleaved
 * into the compiled backend's NHWC input arena. */
lw_status lw_cls_preprocess_bgr_u8_fixed_nhwc(
    const uint8_t* source, uint64_t source_byte_count, uint32_t source_width,
    uint32_t source_height, uint32_t source_stride, float* output,
    uint64_t output_element_count, uint32_t* resized_width,
    lw_cls_preprocess_workspace* workspace);

void lw_cls_preprocess_workspace_init(lw_cls_preprocess_workspace* workspace);
void lw_cls_preprocess_workspace_free(lw_cls_preprocess_workspace* workspace);

#endif
