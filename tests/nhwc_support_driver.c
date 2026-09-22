#include "nhwc_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void reference_pool(const float* input, float* output,
                           uint32_t input_height, uint32_t input_width,
                           uint32_t output_height, uint32_t output_width,
                           uint32_t channels, uint32_t kernel_h, uint32_t kernel_w,
                           uint32_t pad_top, uint32_t pad_left, uint8_t include_pad,
                           uint8_t is_max) {
    for (uint32_t oy = 0u; oy < output_height; ++oy) {
        for (uint32_t ox = 0u; ox < output_width; ++ox) {
            int32_t iy0 = (int32_t)oy - (int32_t)pad_top;
            int32_t ix0 = (int32_t)ox - (int32_t)pad_left;
            for (uint32_t channel = 0u; channel < channels; ++channel) {
                float value = is_max ? -INFINITY : 0.0f;
                uint32_t valid = 0u;
                for (uint32_t ky = 0u; ky < kernel_h; ++ky) {
                    int32_t iy = iy0 + (int32_t)ky;
                    if (iy < 0 || iy >= (int32_t)input_height) continue;
                    for (uint32_t kx = 0u; kx < kernel_w; ++kx) {
                        int32_t ix = ix0 + (int32_t)kx;
                        float sample;
                        if (ix < 0 || ix >= (int32_t)input_width) continue;
                        sample = input[((uint32_t)iy * input_width + (uint32_t)ix) * channels + channel];
                        if (is_max) {
                            if (sample > value) value = sample;
                        } else {
                            value += sample;
                        }
                        ++valid;
                    }
                }
                if (!is_max) {
                    uint32_t denominator = include_pad ? kernel_h * kernel_w : valid;
                    value /= (float)(denominator == 0u ? 1u : denominator);
                }
                output[(oy * output_width + ox) * channels + channel] = value;
            }
        }
    }
}

static int check_case(uint32_t channels, uint8_t include_pad, uint8_t is_max) {
    const uint32_t input_height = 2u;
    const uint32_t input_width = 2u;
    const uint32_t output_height = 3u;
    const uint32_t output_width = 3u;
    const uint32_t elements = output_height * output_width * channels;
    float input[2u * 2u * 8u];
    float expected[3u * 3u * 8u];
    float actual[3u * 3u * 8u];
    for (uint32_t index = 0u; index < input_height * input_width * channels; ++index) {
        input[index] = (float)(index + 1u) * 0.25f;
    }
    reference_pool(input, expected, input_height, input_width, output_height, output_width,
                   channels, 3u, 3u, 1u, 1u, include_pad, is_max);
    lw_avx2_nhwc_pool_f32(input, actual, 1u, input_height, input_width, output_height,
                          output_width, channels, 3u, 3u, 1u, 1u, 1u, 1u, include_pad, is_max);
    for (uint32_t index = 0u; index < elements; ++index) {
        if (fabsf(expected[index] - actual[index]) > 1.0e-6f) {
            fprintf(stderr, "pool mismatch channels=%u include_pad=%u max=%u index=%u expected=%.9g actual=%.9g\n",
                    channels, include_pad, is_max, index, expected[index], actual[index]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    if (!check_case(3u, 0u, 0u) || !check_case(3u, 1u, 0u) ||
        !check_case(3u, 0u, 1u) || !check_case(8u, 0u, 0u) ||
        !check_case(8u, 1u, 1u)) {
        return 1;
    }
    puts("NHWC Pool support cases passed");
    return 0;
}
