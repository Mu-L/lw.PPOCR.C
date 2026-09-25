#include "packed_matmul_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void lw_wasm128_packed_matmul_argmax_scores_f32(
    const float*, const float*, const float*, uint32_t*, float*,
    uint32_t, uint32_t, uint32_t);
void lw_wasm128_ctc_row_probabilities_f32(
    const float*, const float*, const float*, const uint32_t*, const float*,
    float*, uint32_t, uint32_t, uint32_t);

#define ROWS 7u
#define INNER 3u
#define CLASSES 19u
#define PACKED_FLOATS (2u * INNER * 16u)

static int run_case(const float* bias) {
    static const float input[ROWS][INNER] = {
        {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    static const uint32_t expected_indices[ROWS] = {1u, 1u, 2u, 3u, 0u, 4u, 17u};
    float weights[INNER * CLASSES] = {0.0f};
    float packed[PACKED_FLOATS];
    float logits[ROWS][CLASSES];
    uint32_t indices[ROWS];
    float scores[ROWS];
    float probabilities[ROWS];
    weights[1u] = 4.0f;
    weights[2u] = -4.0f;
    weights[CLASSES + 3u] = 3.0f;
    weights[CLASSES + 4u] = -3.0f;
    weights[2u * CLASSES + 17u] = 5.0f;
    lw_pack_matmul_weights_f32(weights, INNER, CLASSES, packed);
    for (uint32_t row = 0u; row < ROWS; ++row) {
        for (uint32_t column = 0u; column < CLASSES; ++column) {
            float value = bias != NULL ? bias[column] : 0.0f;
            for (uint32_t inner = 0u; inner < INNER; ++inner) {
                value += input[row][inner] * weights[inner * CLASSES + column];
            }
            logits[row][column] = value;
        }
    }
    memset(indices, 0xff, sizeof(indices));
    memset(scores, 0, sizeof(scores));
    memset(probabilities, 0, sizeof(probabilities));
    lw_wasm128_packed_matmul_argmax_scores_f32(
        &input[0][0], packed, bias, indices, scores, ROWS, INNER, CLASSES);
    lw_wasm128_ctc_row_probabilities_f32(
        &input[0][0], packed, bias, indices, scores, probabilities,
        ROWS, INNER, CLASSES);
    for (uint32_t row = 0u; row < ROWS; ++row) {
        float maximum = logits[row][0];
        uint32_t best = 0u;
        float expected_probability = 0.0f;
        for (uint32_t column = 1u; column < CLASSES; ++column) {
            if (logits[row][column] > maximum) {
                maximum = logits[row][column];
                best = column;
            }
        }
        if (best != expected_indices[row] || indices[row] != best ||
            fabsf(scores[row] - maximum) > 1.0e-6f) {
            fprintf(stderr, "CTC argmax mismatch at row %u: expected %u got %u\n",
                    row, best, indices[row]);
            return 1;
        }
        if (best != 0u && (row == 0u || best != indices[row - 1u])) {
            float sum = 0.0f;
            for (uint32_t column = 0u; column < CLASSES; ++column) {
                sum += expf(logits[row][column] - maximum);
            }
            expected_probability = 1.0f / sum;
        }
        if (fabsf(probabilities[row] - expected_probability) > 1.0e-6f) {
            fprintf(stderr, "CTC probability mismatch at row %u: expected %.9g got %.9g\n",
                    row, expected_probability, probabilities[row]);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    float bias[CLASSES] = {0.0f};
    bias[0] = 0.2f;
    if (run_case(bias) != 0 || run_case(NULL) != 0) return 1;
    puts("portable CTC projection matches reference, including row/column tails");
    return 0;
}
