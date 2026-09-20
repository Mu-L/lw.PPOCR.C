#ifndef LW_NHWC_INTERNAL_H
#define LW_NHWC_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

#define LW_NHWC_OC_BLOCK 16u
#define LW_NHWC_PIXEL_TILE 6u
#define LW_NHWC_POINTWISE_GROUP_TILES 8u

typedef enum lw_nhwc_activation {
    LW_NHWC_ACT_NONE = 0,
    LW_NHWC_ACT_RELU = 1,
    LW_NHWC_ACT_HARDSWISH = 2,
    LW_NHWC_ACT_GELU = 3
} lw_nhwc_activation;

typedef struct lw_nhwc_epilogue {
    const float* bias;
    const float* residual;
    uint16_t activation;
    uint16_t reserved;
    float alpha;
    float beta;
} lw_nhwc_epilogue;

int lw_nhwc_dense_packed_weight_count(uint32_t input_channels,
                                      uint32_t output_channels,
                                      uint32_t kernel_h,
                                      uint32_t kernel_w,
                                      uint64_t* element_count);

void lw_pack_nhwc_dense_f32(const float* weights,
                            uint32_t input_channels,
                            uint32_t output_channels,
                            uint32_t kernel_h,
                            uint32_t kernel_w,
                            float* packed_weights);

/* Input/output are NHWC. This API is experimental and production-disabled. */
void lw_avx2_fma_nhwc_pointwise_grouped_f32(const float* input,
                                             const float* packed_weights,
                                             const lw_nhwc_epilogue* epilogue,
                                             float* output,
                                             uint32_t pixels,
                                             uint32_t input_channels,
                                             uint32_t output_channels,
                                             uint32_t group_tiles);

void lw_avx2_fma_nhwc_pointwise_f32(const float* input,
                                    const float* packed_weights,
                                    const lw_nhwc_epilogue* epilogue,
                                    float* output,
                                    uint32_t pixels,
                                    uint32_t input_channels,
                                    uint32_t output_channels);

#endif
