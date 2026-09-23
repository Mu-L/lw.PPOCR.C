#ifndef LW_NHWC_INTERNAL_H
#define LW_NHWC_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

#define LW_NHWC_OC_BLOCK 16u
#define LW_NHWC_DEPTHWISE_BLOCK 32u
#define LW_NHWC_PIXEL_TILE 6u
#define LW_NHWC_POINTWISE_GROUP_TILES 8u
#define LW_NHWC_DENSE_KC 512u
#define LW_NHWC_PARTIAL_FLOATS (LW_NHWC_PIXEL_TILE * LW_NHWC_OC_BLOCK)

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

typedef struct lw_nhwc_dense_desc {
    uint32_t batch;
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t dense_kc;
    /* Optional sharded-row range: output rows [offset, offset + output_height)
     * of the full output. Zero keeps the whole-output serial semantics. */
    uint32_t output_row_offset;
} lw_nhwc_dense_desc;

int lw_nhwc_dense_scratch_bytes(const lw_nhwc_dense_desc* desc, uint64_t* scratch_bytes);

int lw_nhwc_dense_prepared_scratch_bytes(const lw_nhwc_dense_desc* desc, uint64_t* scratch_bytes);

lw_status lw_avx2_fma_nhwc_dense_prepared_f32(const float* input,
                                             const float* packed_weights,
                                             const int32_t* input_offsets_k,
                                             const int32_t* patch_offsets_k,
                                             const lw_nhwc_epilogue* epilogue,
                                             float* output,
                                             const lw_nhwc_dense_desc* desc,
                                             void* scratch,
                                             uint64_t scratch_bytes);

lw_status lw_avx2_fma_nhwc_dense_f32(const float* input,
                                     const float* packed_weights,
                                     const lw_nhwc_epilogue* epilogue,
                                     float* output,
                                     const lw_nhwc_dense_desc* desc,
                                     void* scratch,
                                     uint64_t scratch_bytes);
/* Input/output are NHWC. This API is experimental and production-disabled. */
typedef struct lw_nhwc_depthwise_desc {
    uint32_t batch;
    uint32_t channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    /* Optional sharded-row range: output rows [offset, offset + output_height)
     * of the full output. Zero keeps the whole-output serial semantics. */
    uint32_t output_row_offset;
} lw_nhwc_depthwise_desc;

int lw_nhwc_depthwise_packed_weight_count(uint32_t channels,
                                          uint32_t kernel_h,
                                          uint32_t kernel_w,
                                          uint64_t* element_count);

void lw_pack_nhwc_depthwise_f32(const float* weights,
                                uint32_t channels,
                                uint32_t kernel_h,
                                uint32_t kernel_w,
                                float* packed_weights);

lw_status lw_avx2_fma_nhwc_depthwise_f32(const float* input,
                                         const float* packed_weights,
                                         const float* bias,
                                         float* output,
                                         const lw_nhwc_depthwise_desc* desc);
typedef struct lw_nhwc_depthwise_stats {
    uint64_t x1_invocations;
    uint64_t x2_invocations;
} lw_nhwc_depthwise_stats;

lw_status lw_avx2_fma_nhwc_depthwise_profiled_f32(const float* input,
                                                   const float* packed_weights,
                                                   const float* bias,
                                                   float* output,
                                                   const lw_nhwc_depthwise_desc* desc,
                                                   lw_nhwc_depthwise_stats* stats);
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
void lw_avx2_fma_nhwc_pointwise_4x16_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels);
void lw_avx2_fma_nhwc_pointwise_3x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels);
void lw_avx2_fma_nhwc_pointwise_2x32_f32(const float* input,
                                           const float* packed_weights,
                                           const lw_nhwc_epilogue* epilogue,
                                           float* output,
                                           uint32_t pixels,
                                           uint32_t input_channels,
                                           uint32_t output_channels);

void lw_avx2_nhwc_affine_f32(const float* input, const float* mul, const float* add,
                             float* output, uint32_t pixels, uint32_t channels);
void lw_avx2_nchw_affine_f32(const float* input, const float* mul, const float* add,
                             float* output, uint32_t channels, uint32_t spatial);
void lw_avx2_nhwc_reduce_mean_hw_f32(const float* input, float* output,
                                     uint32_t batch, uint32_t height,
                                     uint32_t width, uint32_t channels);
void lw_avx2_nhwc_pool_f32(const float* input, float* output,
                           uint32_t batch, uint32_t input_height,
                           uint32_t input_width, uint32_t output_height,
                           uint32_t output_width, uint32_t channels,
                           uint32_t kernel_h, uint32_t kernel_w,
                           uint32_t stride_h, uint32_t stride_w,
                           uint32_t pad_top, uint32_t pad_left,
                           uint8_t count_include_pad, uint8_t is_max);
/* Integer-scale nearest resize (planner-guaranteed scales): each output row
 * is a whole-row copy of the source row at oy / scale_h, each source pixel
 * repeated scale_w times. Pure data movement, bit-identical to the scalar
 * nearest path. */
void lw_avx2_nhwc_resize_nearest_f32(const float* input, float* output,
                                     uint32_t batch, uint32_t channels,
                                     uint32_t input_height, uint32_t input_width,
                                     uint32_t output_height, uint32_t output_width);
/* 2x2 stride-2 no-pad ConvTranspose as four per-tap 1x1 GEMMs. Weights are
 * packed [tap][oc/16][ic][16] from the ONNX [ic, oc, 2, 2] layout. The
 * epilogue supports bias and ReLU only; Sigmoid is deliberately excluded so
 * callers can defuse it into a separate bit-identical pass. */
typedef struct lw_nhwc_convtranspose_desc {
    uint32_t batch;
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_channels;
    uint32_t output_height;
    uint32_t output_width;
} lw_nhwc_convtranspose_desc;

int lw_nhwc_convtranspose_packed_weight_count(uint32_t input_channels,
                                              uint32_t output_channels,
                                              uint64_t* element_count);
void lw_pack_nhwc_convtranspose2x2_f32(const float* weights, uint32_t input_channels,
                                       uint32_t output_channels, float* packed_weights);
lw_status lw_avx2_fma_nhwc_convtranspose2x2_s2_f32(
    const float* input, const float* packed_weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc);
/* Single-output-channel probability-map variant; unpaced ONNX weights
 * [ic, 1, 2, 2], ReLU-only epilogue. */
lw_status lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32(
    const float* input, const float* weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc);
#endif
