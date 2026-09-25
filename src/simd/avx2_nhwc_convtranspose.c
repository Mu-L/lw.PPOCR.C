#include "nhwc_internal.h"

#include <stddef.h>
#include <stdint.h>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#define LW_CT_X86 1
#else
#define LW_CT_X86 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LW_CT_TARGET __attribute__((target("avx2,fma")))
#else
#define LW_CT_TARGET
#endif

#define LW_CT_TILE 6u
#define LW_CT_TAPS 4u

int lw_nhwc_convtranspose_packed_weight_count(uint32_t input_channels,
                                              uint32_t output_channels,
                                              uint64_t* element_count) {
    uint64_t blocks;
    uint64_t count;
    if (input_channels == 0u || output_channels == 0u ||
        (output_channels & 7u) != 0u || element_count == NULL) {
        return 0;
    }
    blocks = (output_channels + LW_NHWC_OC_BLOCK - 1u) / LW_NHWC_OC_BLOCK;
    count = blocks;
    if (count > UINT64_MAX / input_channels) return 0;
    count *= input_channels;
    if (count > UINT64_MAX / LW_CT_TAPS) return 0;
    count *= LW_CT_TAPS;
    if (count > UINT64_MAX / LW_NHWC_OC_BLOCK) return 0;
    count *= LW_NHWC_OC_BLOCK;
    if (count > (uint64_t)(SIZE_MAX / sizeof(float))) return 0;
    *element_count = count;
    return 1;
}

void lw_pack_nhwc_convtranspose2x2_f32(const float* weights, uint32_t input_channels,
                                       uint32_t output_channels, float* packed_weights) {
    /* ONNX ConvTranspose weights [ic, oc, 2, 2] -> [tap][oc/16][ic][16] with
     * tap = dy*2+dx. */
    uint32_t blocks = (output_channels + LW_NHWC_OC_BLOCK - 1u) / LW_NHWC_OC_BLOCK;
    uint32_t tap;
    for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
        uint32_t block;
        for (block = 0u; block < blocks; ++block) {
            uint32_t input_channel;
            for (input_channel = 0u; input_channel < input_channels; ++input_channel) {
                uint32_t lane;
                for (lane = 0u; lane < LW_NHWC_OC_BLOCK; ++lane) {
                    uint32_t output_channel = block * LW_NHWC_OC_BLOCK + lane;
                    size_t packed_index =
                        ((((size_t)tap * blocks + block) * input_channels + input_channel) *
                         LW_NHWC_OC_BLOCK) + lane;
                    size_t weight_index = ((size_t)input_channel * output_channels +
                                           output_channel) * LW_CT_TAPS + tap;
                    packed_weights[packed_index] = output_channel < output_channels
                        ? weights[weight_index] : 0.0f;
                }
            }
        }
    }
}

#if LW_CT_X86
LW_CT_TARGET
static void convtranspose_tile_16(const float* input_row, const float* packed_weights,
                                  const lw_nhwc_epilogue* epilogue,
                                  float* output_row0, float* output_row1,
                                  const lw_nhwc_convtranspose_desc* desc,
                                  uint32_t oc_blocks, uint32_t input_x,
                                  uint32_t pixels) {
    uint32_t block;
    for (block = 0u; block < oc_blocks; ++block) {
        uint32_t block_channels = desc->output_channels - block * LW_NHWC_OC_BLOCK;
        uint32_t tap;
        __m256 bias_lo = _mm256_setzero_ps();
        __m256 bias_hi = _mm256_setzero_ps();
        if (block_channels > LW_NHWC_OC_BLOCK) block_channels = LW_NHWC_OC_BLOCK;
        if (epilogue != NULL && epilogue->bias != NULL) {
            if (block_channels < LW_NHWC_OC_BLOCK) {
                float bias_values[LW_NHWC_OC_BLOCK] = {0.0f};
                uint32_t lane;
                for (lane = 0u; lane < block_channels; ++lane) {
                    bias_values[lane] = epilogue->bias[block * LW_NHWC_OC_BLOCK + lane];
                }
                bias_lo = _mm256_loadu_ps(bias_values);
                bias_hi = _mm256_loadu_ps(bias_values + 8u);
            } else {
                bias_lo = _mm256_loadu_ps(epilogue->bias + block * LW_NHWC_OC_BLOCK);
                bias_hi = _mm256_loadu_ps(epilogue->bias + block * LW_NHWC_OC_BLOCK + 8u);
            }
        }
        for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
            __m256 acc_lo[LW_CT_TILE];
            __m256 acc_hi[LW_CT_TILE];
            uint32_t input_channel;
            uint32_t pixel;
            for (pixel = 0u; pixel < LW_CT_TILE; ++pixel) {
                acc_lo[pixel] = bias_lo;
                acc_hi[pixel] = bias_hi;
            }
            for (input_channel = 0u; input_channel < desc->input_channels; ++input_channel) {
                const float* weight = packed_weights +
                    ((((size_t)tap * oc_blocks + block) * desc->input_channels +
                      input_channel) * LW_NHWC_OC_BLOCK);
                __m256 w_lo = _mm256_loadu_ps(weight);
                __m256 w_hi = _mm256_loadu_ps(weight + 8u);
                for (pixel = 0u; pixel < pixels; ++pixel) {
                    __m256 value = _mm256_set1_ps(
                        input_row[(size_t)(input_x + pixel) * desc->input_channels +
                                  input_channel]);
                    acc_lo[pixel] = _mm256_fmadd_ps(value, w_lo, acc_lo[pixel]);
                    acc_hi[pixel] = _mm256_fmadd_ps(value, w_hi, acc_hi[pixel]);
                }
            }
            {
                uint32_t dy = tap >> 1u;
                uint32_t dx = tap & 1u;
                float* output_row = dy == 0u ? output_row0 : output_row1;
                for (pixel = 0u; pixel < pixels; ++pixel) {
                    float* destination = output_row +
                        ((size_t)((input_x + pixel) * 2u + dx) * desc->output_channels) +
                        block * LW_NHWC_OC_BLOCK;
                    if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_RELU) {
                        acc_lo[pixel] = _mm256_max_ps(acc_lo[pixel], _mm256_setzero_ps());
                        acc_hi[pixel] = _mm256_max_ps(acc_hi[pixel], _mm256_setzero_ps());
                    }
                    if (block_channels < LW_NHWC_OC_BLOCK) {
                        float values[LW_NHWC_OC_BLOCK];
                        uint32_t lane;
                        _mm256_storeu_ps(values, acc_lo[pixel]);
                        _mm256_storeu_ps(values + 8u, acc_hi[pixel]);
                        for (lane = 0u; lane < block_channels; ++lane) {
                            destination[lane] = values[lane];
                        }
                    } else {
                        _mm256_storeu_ps(destination, acc_lo[pixel]);
                        _mm256_storeu_ps(destination + 8u, acc_hi[pixel]);
                    }
                }
            }
        }
    }
}
#endif

lw_status lw_avx2_fma_nhwc_convtranspose2x2_s2_f32(
    const float* input, const float* packed_weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc) {
    uint32_t oc_blocks;
    uint32_t batch;
    uint32_t input_y;
    if (input == NULL || packed_weights == NULL || output == NULL || desc == NULL ||
        desc->batch == 0u || desc->input_channels == 0u || desc->input_height == 0u ||
        desc->input_width == 0u || desc->output_channels == 0u ||
        (desc->output_channels & 7u) != 0u || desc->output_height != desc->input_height * 2u ||
        desc->output_width != desc->input_width * 2u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
#if !LW_CT_X86
    (void)epilogue;
    return LW_STATUS_UNSUPPORTED;
#else
    oc_blocks = (desc->output_channels + LW_NHWC_OC_BLOCK - 1u) / LW_NHWC_OC_BLOCK;
    for (batch = 0u; batch < desc->batch; ++batch) {
        const float* input_batch =
            input + (size_t)batch * desc->input_height * desc->input_width * desc->input_channels;
        float* output_batch = output +
            (size_t)batch * desc->output_height * desc->output_width * desc->output_channels;
        for (input_y = 0u; input_y < desc->input_height; ++input_y) {
            const float* input_row =
                input_batch + (size_t)input_y * desc->input_width * desc->input_channels;
            float* output_row0 = output_batch +
                (size_t)(input_y * 2u) * desc->output_width * desc->output_channels;
            float* output_row1 = output_row0 +
                (size_t)desc->output_width * desc->output_channels;
            uint32_t input_x = 0u;
            while (input_x < desc->input_width) {
                uint32_t pixels = desc->input_width - input_x;
                if (pixels > LW_CT_TILE) pixels = LW_CT_TILE;
                convtranspose_tile_16(input_row, packed_weights, epilogue, output_row0,
                                      output_row1, desc, oc_blocks, input_x, pixels);
                input_x += pixels;
            }
        }
    }
    return LW_STATUS_OK;
#endif
}

#if LW_CT_X86
LW_CT_TARGET
#endif
lw_status lw_avx2_fma_nhwc_convtranspose2x2_s2_c1_f32(
    const float* input, const float* weights, const lw_nhwc_epilogue* epilogue,
    float* output, const lw_nhwc_convtranspose_desc* desc) {
    uint32_t batch;
    uint32_t input_y;
    if (input == NULL || weights == NULL || output == NULL || desc == NULL ||
        desc->batch == 0u || desc->input_channels == 0u || desc->input_height == 0u ||
        desc->input_width == 0u || desc->output_channels != 1u ||
        desc->output_height != desc->input_height * 2u ||
        desc->output_width != desc->input_width * 2u) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
#if !LW_CT_X86
    (void)epilogue;
    return LW_STATUS_UNSUPPORTED;
#else
    {
        /* Strided per-tap weight columns (ONNX [ic, 1, 2, 2]: w[tap][ic] =
         * weights[ic*4+tap]) gathered once into contiguous vectors. The first
         * two 8-lane chunks are vectorized; any remainder runs scalar. */
        uint32_t vector_channels = desc->input_channels >= 16u ? 16u : desc->input_channels;
        __m256 weight_vectors[LW_CT_TAPS][2];
        uint32_t chunk;
        for (chunk = 0u; chunk * 8u < vector_channels; ++chunk) {
            uint32_t tap;
            for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                float gathered[8];
                uint32_t lane;
                for (lane = 0u; lane < 8u; ++lane) {
                    gathered[lane] = weights[(size_t)(chunk * 8u + lane) * LW_CT_TAPS + tap];
                }
                weight_vectors[tap][chunk] = _mm256_loadu_ps(gathered);
            }
        }
        for (batch = 0u; batch < desc->batch; ++batch) {
            const float* input_batch =
                input + (size_t)batch * desc->input_height * desc->input_width * desc->input_channels;
            float* output_batch = output + (size_t)batch * desc->output_height * desc->output_width;
            for (input_y = 0u; input_y < desc->input_height; ++input_y) {
                const float* input_row =
                    input_batch + (size_t)input_y * desc->input_width * desc->input_channels;
                float* output_row0 = output_batch + (size_t)(input_y * 2u) * desc->output_width;
                float* output_row1 = output_row0 + desc->output_width;
                uint32_t input_x;
                for (input_x = 0u; input_x < desc->input_width; ++input_x) {
                    const float* pixel = input_row + (size_t)input_x * desc->input_channels;
                    float sums[LW_CT_TAPS];
                    uint32_t tap;
                    __m256 acc[LW_CT_TAPS];
                    for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                        acc[tap] = _mm256_setzero_ps();
                    }
                    if (vector_channels >= 8u) {
                        __m256 in0 = _mm256_loadu_ps(pixel);
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                            acc[tap] = _mm256_fmadd_ps(in0, weight_vectors[tap][0], acc[tap]);
                        }
                    }
                    if (vector_channels >= 16u) {
                        __m256 in1 = _mm256_loadu_ps(pixel + 8u);
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                            acc[tap] = _mm256_fmadd_ps(in1, weight_vectors[tap][1], acc[tap]);
                        }
                    }
                    /* Channels beyond the vectorized prefix accumulate into
                     * lane 0 of a scalar sum per tap. */
                    if (desc->input_channels > vector_channels) {
                        float extra[LW_CT_TAPS];
                        uint32_t input_channel;
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) extra[tap] = 0.0f;
                        for (input_channel = vector_channels; input_channel < desc->input_channels;
                             ++input_channel) {
                            float value = pixel[input_channel];
                            for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                                extra[tap] += value * weights[(size_t)input_channel * LW_CT_TAPS + tap];
                            }
                        }
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                            float lane_value[8];
                            _mm256_storeu_ps(lane_value, acc[tap]);
                            lane_value[0] += extra[tap];
                            acc[tap] = _mm256_loadu_ps(lane_value);
                        }
                    }
                    for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                        float lanes[8];
                        _mm256_storeu_ps(lanes, acc[tap]);
                        sums[tap] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                                    lanes[5] + lanes[6] + lanes[7];
                    }
                    if (epilogue != NULL && epilogue->bias != NULL) {
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) sums[tap] += epilogue->bias[0];
                    }
                    if (epilogue != NULL && epilogue->activation == LW_NHWC_ACT_RELU) {
                        for (tap = 0u; tap < LW_CT_TAPS; ++tap) {
                            if (sums[tap] < 0.0f) sums[tap] = 0.0f;
                        }
                    }
                    output_row0[(size_t)input_x * 2u] = sums[0];
                    output_row0[(size_t)input_x * 2u + 1u] = sums[1];
                    output_row1[(size_t)input_x * 2u] = sums[2];
                    output_row1[(size_t)input_x * 2u + 1u] = sums[3];
                }
            }
        }
        return LW_STATUS_OK;
    }
#endif
}
