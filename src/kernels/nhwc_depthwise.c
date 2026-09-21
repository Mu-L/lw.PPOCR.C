#include "nhwc_internal.h"

#include <limits.h>
#include <stddef.h>

int lw_nhwc_depthwise_packed_weight_count(uint32_t channels,
                                          uint32_t kernel_h,
                                          uint32_t kernel_w,
                                          uint64_t* element_count) {
    uint64_t blocks;
    uint64_t taps;
    if (channels == 0u || kernel_h == 0u || kernel_w == 0u || element_count == NULL) return 0;
    blocks = ((uint64_t)channels + LW_NHWC_DEPTHWISE_BLOCK - 1u) / LW_NHWC_DEPTHWISE_BLOCK;
    taps = (uint64_t)kernel_h * kernel_w;
    if (blocks > UINT64_MAX / taps || blocks * taps > UINT64_MAX / LW_NHWC_DEPTHWISE_BLOCK) return 0;
    *element_count = blocks * taps * LW_NHWC_DEPTHWISE_BLOCK;
    return *element_count <= SIZE_MAX / sizeof(float);
}

void lw_pack_nhwc_depthwise_f32(const float* weights, uint32_t channels,
                                uint32_t kernel_h, uint32_t kernel_w,
                                float* packed_weights) {
    uint32_t blocks;
    uint32_t taps;
    if (weights == NULL || packed_weights == NULL || channels == 0u || kernel_h == 0u || kernel_w == 0u) return;
    blocks = (channels + LW_NHWC_DEPTHWISE_BLOCK - 1u) / LW_NHWC_DEPTHWISE_BLOCK;
    taps = kernel_h * kernel_w;
    for (uint32_t block = 0u; block < blocks; ++block) {
        for (uint32_t tap = 0u; tap < taps; ++tap) {
            uint32_t ky = tap / kernel_w;
            uint32_t kx = tap - ky * kernel_w;
            for (uint32_t lane = 0u; lane < LW_NHWC_DEPTHWISE_BLOCK; ++lane) {
                uint32_t channel = block * LW_NHWC_DEPTHWISE_BLOCK + lane;
                size_t dst = ((size_t)block * taps + tap) * LW_NHWC_DEPTHWISE_BLOCK + lane;
                packed_weights[dst] = channel < channels
                    ? weights[((size_t)channel * kernel_h + ky) * kernel_w + kx]
                    : 0.0f;
            }
        }
    }
}
