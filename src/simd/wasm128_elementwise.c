#include "simd_kernels.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

static v128_t binary_vector(lw_scalar_binary_op operation, v128_t left, v128_t right) {
    switch (operation) {
    case LW_SCALAR_BINARY_ADD: return wasm_f32x4_add(left, right);
    case LW_SCALAR_BINARY_SUB: return wasm_f32x4_sub(left, right);
    case LW_SCALAR_BINARY_MUL: return wasm_f32x4_mul(left, right);
    case LW_SCALAR_BINARY_DIV: return wasm_f32x4_div(left, right);
    default: return left;
    }
}

static float binary_scalar(lw_scalar_binary_op operation, float left, float right) {
    switch (operation) {
    case LW_SCALAR_BINARY_ADD: return left + right;
    case LW_SCALAR_BINARY_SUB: return left - right;
    case LW_SCALAR_BINARY_MUL: return left * right;
    case LW_SCALAR_BINARY_DIV: return left / right;
    default: return powf(left, right);
    }
}

void lw_wasm128_binary_contiguous_f32(lw_scalar_binary_op operation,
    const float* left, const float* right, float* output, uint64_t count) {
    uint64_t index = 0u;
    if (operation != LW_SCALAR_BINARY_POW) {
        for (; index + 4u <= count; index += 4u) {
            wasm_v128_store(output + (size_t)index,
                binary_vector(operation, wasm_v128_load(left + (size_t)index),
                    wasm_v128_load(right + (size_t)index)));
        }
    }
    for (; index < count; ++index)
        output[(size_t)index] = binary_scalar(operation, left[(size_t)index], right[(size_t)index]);
}

void lw_wasm128_binary_scalar_f32(lw_scalar_binary_op operation,
    const float* tensor, float scalar, float* output, uint64_t count, int scalar_is_left) {
    uint64_t index = 0u;
    if (operation != LW_SCALAR_BINARY_POW) {
        v128_t broadcast = wasm_f32x4_splat(scalar);
        for (; index + 4u <= count; index += 4u) {
            v128_t value = wasm_v128_load(tensor + (size_t)index);
            wasm_v128_store(output + (size_t)index,
                binary_vector(operation, scalar_is_left != 0 ? broadcast : value,
                    scalar_is_left != 0 ? value : broadcast));
        }
    }
    for (; index < count; ++index) {
        float value = tensor[(size_t)index];
        output[(size_t)index] = binary_scalar(operation,
            scalar_is_left != 0 ? scalar : value,
            scalar_is_left != 0 ? value : scalar);
    }
}

void lw_wasm128_binary_channel_nhwc_f32(lw_scalar_binary_op operation,
    const float* tensor, const float* channel, float* output,
    uint64_t pixels, uint32_t channels, int channel_is_left) {
    for (uint64_t pixel = 0u; pixel < pixels; ++pixel) {
        const float* source = tensor + (size_t)pixel * channels;
        float* destination = output + (size_t)pixel * channels;
        uint32_t c = 0u;
        if (operation != LW_SCALAR_BINARY_POW) {
            for (; c + 4u <= channels; c += 4u) {
                v128_t value = wasm_v128_load(source + c);
                v128_t scale = wasm_v128_load(channel + c);
                wasm_v128_store(destination + c,
                    binary_vector(operation, channel_is_left != 0 ? scale : value,
                        channel_is_left != 0 ? value : scale));
            }
        }
        for (; c < channels; ++c) {
            destination[c] = binary_scalar(operation,
                channel_is_left != 0 ? channel[c] : source[c],
                channel_is_left != 0 ? source[c] : channel[c]);
        }
    }
}

void lw_wasm128_affine_nhwc_f32(const float* input, const float* mul,
    const float* add, float* output, uint32_t pixels, uint32_t channels) {
    for (uint32_t pixel = 0u; pixel < pixels; ++pixel) {
        const float* source = input + (size_t)pixel * channels;
        float* destination = output + (size_t)pixel * channels;
        uint32_t channel = 0u;
        for (; channel + 4u <= channels; channel += 4u) {
            v128_t value = wasm_f32x4_mul(wasm_v128_load(source + channel),
                wasm_v128_load(mul + channel));
            wasm_v128_store(destination + channel,
                wasm_f32x4_add(value, wasm_v128_load(add + channel)));
        }
        for (; channel < channels; ++channel)
            destination[channel] = source[channel] * mul[channel] + add[channel];
    }
}

void lw_wasm128_relu_f32(const float* input, float* output, uint64_t count) {
    const v128_t zero = wasm_f32x4_splat(0.0f);
    uint64_t index = 0u;
    for (; index + 4u <= count; index += 4u) {
        v128_t value = wasm_v128_load(input + (size_t)index);
        wasm_v128_store(output + (size_t)index,
            wasm_v128_bitselect(value, zero, wasm_f32x4_gt(value, zero)));
    }
    for (; index < count; ++index) {
        float value = input[(size_t)index];
        output[(size_t)index] = value > 0.0f ? value : 0.0f;
    }
}

void lw_wasm128_reduce_mean_hw_f32(const float* input, float* output,
    uint32_t batch, uint32_t height, uint32_t width, uint32_t channels) {
    const float denominator = (float)height * (float)width;
    for (uint32_t n = 0u; n < batch; ++n) {
        uint32_t channel = 0u;
        for (; channel + 4u <= channels; channel += 4u) {
            v128_t sum = wasm_f32x4_splat(0.0f);
            for (uint32_t y = 0u; y < height; ++y) {
                for (uint32_t x = 0u; x < width; ++x) {
                    const float* source = input +
                        ((((size_t)n * height + y) * width + x) * channels + channel);
                    sum = wasm_f32x4_add(sum, wasm_v128_load(source));
                }
            }
            wasm_v128_store(output + (size_t)n * channels + channel,
                wasm_f32x4_div(sum, wasm_f32x4_splat(denominator)));
        }
        for (; channel < channels; ++channel) {
            float sum = 0.0f;
            for (uint32_t y = 0u; y < height; ++y)
                for (uint32_t x = 0u; x < width; ++x)
                    sum += input[(((size_t)n * height + y) * width + x) * channels + channel];
            output[(size_t)n * channels + channel] = sum / denominator;
        }
    }
}

static void pool_window(int32_t iy0, int32_t ix0, uint32_t input_height,
    uint32_t input_width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t* ky_begin, uint32_t* ky_end, uint32_t* kx_begin, uint32_t* kx_end) {
    int32_t y_begin = iy0 < 0 ? -iy0 : 0;
    int32_t x_begin = ix0 < 0 ? -ix0 : 0;
    int32_t y_end = (int32_t)kernel_h;
    int32_t x_end = (int32_t)kernel_w;
    if (iy0 + y_end > (int32_t)input_height) y_end = (int32_t)input_height - iy0;
    if (ix0 + x_end > (int32_t)input_width) x_end = (int32_t)input_width - ix0;
    if (y_end < y_begin) y_end = y_begin;
    if (x_end < x_begin) x_end = x_begin;
    if (y_begin > (int32_t)kernel_h) y_begin = (int32_t)kernel_h;
    if (x_begin > (int32_t)kernel_w) x_begin = (int32_t)kernel_w;
    if (y_end > (int32_t)kernel_h) y_end = (int32_t)kernel_h;
    if (x_end > (int32_t)kernel_w) x_end = (int32_t)kernel_w;
    *ky_begin = (uint32_t)y_begin;
    *ky_end = (uint32_t)y_end;
    *kx_begin = (uint32_t)x_begin;
    *kx_end = (uint32_t)x_end;
}

void lw_wasm128_pool_nhwc_f32(const float* input, float* output,
    uint32_t batch, uint32_t input_height, uint32_t input_width,
    uint32_t output_height, uint32_t output_width, uint32_t channels,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_top, uint32_t pad_left, uint8_t count_include_pad, uint8_t is_max) {
    for (uint32_t n = 0u; n < batch; ++n) {
        for (uint32_t oy = 0u; oy < output_height; ++oy) {
            for (uint32_t ox = 0u; ox < output_width; ++ox) {
                int32_t iy0 = (int32_t)((uint64_t)oy * stride_h) - (int32_t)pad_top;
                int32_t ix0 = (int32_t)((uint64_t)ox * stride_w) - (int32_t)pad_left;
                uint32_t ky_begin, ky_end, kx_begin, kx_end;
                uint32_t channel = 0u;
                uint32_t denominator;
                float* destination = output +
                    ((((size_t)n * output_height + oy) * output_width + ox) * channels);
                pool_window(iy0, ix0, input_height, input_width, kernel_h, kernel_w,
                    &ky_begin, &ky_end, &kx_begin, &kx_end);
                denominator = count_include_pad ? kernel_h * kernel_w
                    : (ky_end - ky_begin) * (kx_end - kx_begin);
                if (denominator == 0u) denominator = 1u;
                for (; channel + 4u <= channels; channel += 4u) {
                    v128_t value = wasm_f32x4_splat(is_max ? -FLT_MAX : 0.0f);
                    for (uint32_t ky = ky_begin; ky < ky_end; ++ky) {
                        uint32_t iy = (uint32_t)(iy0 + (int32_t)ky);
                        for (uint32_t kx = kx_begin; kx < kx_end; ++kx) {
                            uint32_t ix = (uint32_t)(ix0 + (int32_t)kx);
                            const float* source = input +
                                ((((size_t)n * input_height + iy) * input_width + ix) *
                                 channels + channel);
                            v128_t sample = wasm_v128_load(source);
                            value = is_max
                                ? wasm_v128_bitselect(sample, value,
                                    wasm_f32x4_gt(sample, value))
                                : wasm_f32x4_add(value, sample);
                        }
                    }
                    if (!is_max)
                        value = wasm_f32x4_div(value,
                            wasm_f32x4_splat((float)denominator));
                    wasm_v128_store(destination + channel, value);
                }
                for (; channel < channels; ++channel) {
                    float value = is_max ? -FLT_MAX : 0.0f;
                    for (uint32_t ky = ky_begin; ky < ky_end; ++ky) {
                        uint32_t iy = (uint32_t)(iy0 + (int32_t)ky);
                        for (uint32_t kx = kx_begin; kx < kx_end; ++kx) {
                            uint32_t ix = (uint32_t)(ix0 + (int32_t)kx);
                            float sample = input[(((size_t)n * input_height + iy) *
                                input_width + ix) * channels + channel];
                            if (is_max) {
                                if (sample > value) value = sample;
                            } else value += sample;
                        }
                    }
                    if (!is_max) value /= (float)denominator;
                    destination[channel] = value;
                }
            }
        }
    }
}
