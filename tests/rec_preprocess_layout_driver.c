#include "rec_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int check_width(uint32_t target_width) {
    const uint32_t source_width = 13u;
    const uint32_t source_height = 7u;
    const uint32_t source_stride = source_width * 3u + 5u;
    const uint64_t source_bytes = (uint64_t)source_stride * source_height;
    const uint64_t plane = (uint64_t)LW_REC_INPUT_HEIGHT * target_width;
    const uint64_t elements = plane * 3u;
    uint8_t* source = (uint8_t*)malloc((size_t)source_bytes);
    float* nchw = (float*)malloc((size_t)elements * sizeof(float));
    float* nhwc = (float*)malloc((size_t)elements * sizeof(float));
    uint32_t nchw_width = 0u;
    uint32_t nhwc_width = 0u;
    uint64_t index;
    float max_difference = 0.0f;
    lw_status status;
    if (source == NULL || nchw == NULL || nhwc == NULL) {
        free(source);
        free(nchw);
        free(nhwc);
        return 0;
    }
    for (index = 0u; index < source_bytes; ++index) {
        source[index] = (uint8_t)((index * 37u + 11u) & 0xffu);
    }
    status = lw_rec_preprocess_bgr_u8(source, source_bytes, source_width, source_height,
                                      source_stride, target_width, nchw, elements, &nchw_width);
    if (status != LW_STATUS_OK) {
        free(source);
        free(nchw);
        free(nhwc);
        return 0;
    }
    status = lw_rec_preprocess_bgr_u8_nhwc(source, source_bytes, source_width, source_height,
                                           source_stride, target_width, nhwc, elements,
                                           &nhwc_width);
    if (status != LW_STATUS_OK || nchw_width != nhwc_width) {
        free(source);
        free(nchw);
        free(nhwc);
        return 0;
    }
    for (index = 0u; index < plane; ++index) {
        uint32_t channel;
        for (channel = 0u; channel < 3u; ++channel) {
            float expected = nchw[(size_t)((uint64_t)channel * plane + index)];
            float actual = nhwc[(size_t)(index * 3u + channel)];
            float difference = fabsf(expected - actual);
            if (difference > max_difference) {
                max_difference = difference;
            }
        }
    }
    printf("{\"target_width\":%u,\"resized_width\":%u,\"max_abs_difference\":%.9g}\n",
           target_width, nchw_width, max_difference);
    free(source);
    free(nchw);
    free(nhwc);
    return max_difference <= 1.0e-6f;
}

int main(void) {
    return check_width(320u) && check_width(960u) ? 0 : 1;
}
