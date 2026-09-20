#include "nhwc_internal.h"

#include <limits.h>
#include <stddef.h>

int lw_nhwc_dense_packed_weight_count(uint32_t input_channels,
                                      uint32_t output_channels,
                                      uint32_t kernel_h,
                                      uint32_t kernel_w,
                                      uint64_t* element_count) {
    uint64_t output_blocks;
    uint64_t count;

    if (input_channels == 0u || output_channels == 0u || kernel_h == 0u ||
        kernel_w == 0u || element_count == NULL) {
        return 0;
    }
    output_blocks = ((uint64_t)output_channels + LW_NHWC_OC_BLOCK - 1u) /
                   LW_NHWC_OC_BLOCK;
    count = output_blocks;
    if (count > UINT64_MAX / kernel_h) {
        return 0;
    }
    count *= kernel_h;
    if (count > UINT64_MAX / kernel_w) {
        return 0;
    }
    count *= kernel_w;
    if (count > UINT64_MAX / input_channels) {
        return 0;
    }
    count *= input_channels;
    if (count > UINT64_MAX / LW_NHWC_OC_BLOCK) {
        return 0;
    }
    count *= LW_NHWC_OC_BLOCK;
    if (count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return 0;
    }
    *element_count = count;
    return 1;
}

void lw_pack_nhwc_dense_f32(const float* weights,
                            uint32_t input_channels,
                            uint32_t output_channels,
                            uint32_t kernel_h,
                            uint32_t kernel_w,
                            float* packed_weights) {
    uint32_t output_blocks = output_channels / LW_NHWC_OC_BLOCK +
                             (output_channels % LW_NHWC_OC_BLOCK == 0u ? 0u : 1u);
    uint32_t output_block;
    uint32_t kh;
    uint32_t kw;

    for (output_block = 0u; output_block < output_blocks; ++output_block) {
        for (kh = 0u; kh < kernel_h; ++kh) {
            for (kw = 0u; kw < kernel_w; ++kw) {
                uint32_t input_channel;
                for (input_channel = 0u; input_channel < input_channels;
                     ++input_channel) {
                    uint32_t lane;
                    for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                        uint64_t output_channel =
                            (uint64_t)output_block * LW_NHWC_OC_BLOCK + lane;
                        uint64_t packed_index =
                            (((((uint64_t)output_block * kernel_h + kh) * kernel_w + kw) *
                              input_channels + input_channel) *
                             LW_NHWC_OC_BLOCK) +
                            lane;
                        packed_weights[(size_t)packed_index] =
                            output_channel < output_channels
                                ? weights[((((output_channel * input_channels +
                                             input_channel) * kernel_h + kh) *
                                            kernel_w) +
                                           kw)]
                                : 0.0f;
                    }
                }
            }
        }
    }
}
