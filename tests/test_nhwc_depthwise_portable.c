#include "nhwc_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { CHANNELS = 8, HEIGHT = 3, WIDTH = 4, TAPS = 9 };
    float input[HEIGHT * WIDTH * CHANNELS];
    float packed[TAPS * LW_NHWC_DEPTHWISE_BLOCK] = {0};
    float bias[CHANNELS];
    float post_bias[CHANNELS];
    float output[HEIGHT * WIDTH * CHANNELS];
    float sharded[HEIGHT * WIDTH * CHANNELS];
    lw_nhwc_depthwise_desc desc = {0};
    for (uint32_t i = 0u; i < HEIGHT * WIDTH * CHANNELS; ++i) {
        input[i] = (float)(i % 17u) * 0.0625f;
    }
    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        bias[channel] = (float)channel * 0.03125f;
        post_bias[channel] = (float)channel * -0.015625f;
        for (uint32_t tap = 0u; tap < TAPS; ++tap) {
            packed[tap * LW_NHWC_DEPTHWISE_BLOCK + channel] =
                (float)((tap + channel) % 7u) * 0.125f;
        }
    }
    desc.batch = 1u;
    desc.channels = CHANNELS;
    desc.input_height = HEIGHT;
    desc.input_width = WIDTH;
    desc.output_height = HEIGHT;
    desc.output_width = WIDTH;
    desc.kernel_h = 3u;
    desc.kernel_w = 3u;
    desc.stride_h = 1u;
    desc.stride_w = 1u;
    desc.pad_top = desc.pad_left = desc.pad_bottom = desc.pad_right = 1u;
    desc.post_bias = post_bias;
    if (lw_avx2_fma_nhwc_depthwise_f32(input, packed, bias, output, &desc) != LW_STATUS_OK) {
        return 1;
    }
    for (uint32_t oy = 0u; oy < HEIGHT; ++oy) {
        desc.output_height = 1u;
        desc.output_row_offset = oy;
        if (lw_avx2_fma_nhwc_depthwise_f32(input, packed, bias,
                sharded + (size_t)oy * WIDTH * CHANNELS, &desc) != LW_STATUS_OK) {
            return 2;
        }
    }
    if (memcmp(output, sharded, sizeof(output)) != 0) return 3;
    for (uint32_t oy = 0u; oy < HEIGHT; ++oy) {
        for (uint32_t ox = 0u; ox < WIDTH; ++ox) {
            for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
                float expected = bias[channel];
                for (uint32_t ky = 0u; ky < 3u; ++ky) {
                    int32_t iy = (int32_t)oy + (int32_t)ky - 1;
                    if (iy < 0 || iy >= HEIGHT) continue;
                    for (uint32_t kx = 0u; kx < 3u; ++kx) {
                        int32_t ix = (int32_t)ox + (int32_t)kx - 1;
                        if (ix < 0 || ix >= WIDTH) continue;
                        expected += input[((size_t)(uint32_t)iy * WIDTH +
                            (uint32_t)ix) * CHANNELS + channel] *
                            packed[(ky * 3u + kx) * LW_NHWC_DEPTHWISE_BLOCK + channel];
                    }
                }
                expected += post_bias[channel];
                if (fabsf(output[((size_t)oy * WIDTH + ox) * CHANNELS + channel] -
                          expected) > 1e-5f) {
                    fprintf(stderr, "portable depthwise mismatch at %u,%u,%u\n",
                            oy, ox, channel);
                    return 4;
                }
            }
        }
    }
    return 0;
}
