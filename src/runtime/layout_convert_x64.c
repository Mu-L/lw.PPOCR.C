#include "x64_rec_fast_internal.h"

#include <stddef.h>

void lw_x64_fast_nchw_to_nhwc(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width) {
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t y = 0u; y < height; ++y) {
            for (uint32_t x = 0u; x < width; ++x) {
                float* dst = destination + (((size_t)n * height + y) * width + x) * channels;
                for (uint32_t c = 0u; c < channels; ++c) {
                    dst[c] = source[(((size_t)n * channels + c) * height + y) * width + x];
                }
            }
        }
    }
}

void lw_x64_fast_nhwc_to_nchw(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width) {
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t c = 0u; c < channels; ++c) {
            float* dst = destination + ((size_t)n * channels + c) * height * width;
            for (uint32_t y = 0u; y < height; ++y) {
                for (uint32_t x = 0u; x < width; ++x) {
                    dst[(size_t)y * width + x] =
                        source[(((size_t)n * height + y) * width + x) * channels + c];
                }
            }
        }
    }
}
