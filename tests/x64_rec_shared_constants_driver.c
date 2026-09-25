#include "x64_rec_backend_internal.h"
#include "model_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const lw_x64_rec_op* source_op(const lw_x64_rec_program* root, uint32_t semantic) {
    uint32_t i;
    for (i = 0u; i < root->op_count; ++i) {
        if (root->ops[i].semantic_begin == semantic) return &root->ops[i];
    }
    return NULL;
}

static int run_pair(const lw_x64_rec_program* independent,
                    const lw_x64_rec_program* shared, lw_error* error) {
    lw_x64_rec_instance* a = NULL;
    lw_x64_rec_instance* b = NULL;
    float* input_a;
    float* input_b;
    uint64_t count_a = 0u;
    uint64_t count_b = 0u;
    uint32_t t;
    int ok = 0;
    if (lw_x64_rec_instance_create(independent, &a, error) != LW_STATUS_OK ||
        lw_x64_rec_instance_create(shared, &b, error) != LW_STATUS_OK) goto done;
    input_a = lw_x64_rec_instance_input(a, &count_a);
    input_b = lw_x64_rec_instance_input(b, &count_b);
    if (input_a == NULL || input_b == NULL || count_a != count_b) goto done;
    for (uint64_t i = 0u; i < count_a; ++i) {
        input_a[i] = (float)((int)(i % 251u) - 125) / 127.0f;
        input_b[i] = input_a[i];
    }
    if (lw_x64_rec_instance_run(a, error) != LW_STATUS_OK ||
        lw_x64_rec_instance_run(b, error) != LW_STATUS_OK ||
        independent->time_steps != shared->time_steps) goto done;
    for (t = 0u; t < shared->time_steps; ++t) {
        if (a->best_indices[t] != b->best_indices[t] ||
            fabsf(a->best_probabilities[t] - b->best_probabilities[t]) > 1.0e-6f)
            goto done;
    }
    ok = 1;
done:
    lw_x64_rec_instance_free(a);
    lw_x64_rec_instance_free(b);
    return ok;
}

int main(int argc, char** argv) {
    static const uint32_t widths[] = {192u, 320u, 480u, 640u};
    lw_model* model = NULL;
    lw_x64_rec_program* root = NULL;
    lw_x64_rec_program* independent = NULL;
    lw_x64_rec_program* shared[4] = {NULL, NULL, NULL, NULL};
    lw_error error;
    uint64_t independent_total = 0u;
    uint64_t shared_total = 0u;
    uint32_t borrowed_conv = 0u;
    uint32_t borrowed_affine = 0u;
    uint32_t borrowed_matmul = 0u;
    uint32_t i;
    int result = 1;
    if (argc != 2) return 2;
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK ||
        lw_x64_rec_backend_compile(model, 960u, &root, &error) != LW_X64_REC_COMPILE_OK ||
        root == NULL) goto done;
    independent_total = root->owned_constant_bytes;
    shared_total = root->owned_constant_bytes;
    for (i = 0u; i < 4u; ++i) {
        lw_x64_rec_program* own = NULL;
        uint32_t op_index;
        if (lw_x64_rec_backend_compile(model, widths[i], &own, &error) != LW_X64_REC_COMPILE_OK ||
            own == NULL) goto done;
        independent_total += own->owned_constant_bytes;
        if (lw_x64_rec_backend_compile_shared(model, widths[i], root, &shared[i], &error) !=
                LW_X64_REC_COMPILE_OK || shared[i] == NULL) {
            lw_x64_rec_program_free(own);
            goto done;
        }
        shared_total += shared[i]->owned_constant_bytes;
        if (shared[i]->packed_source != root || shared[i]->borrowed_constant_count == 0u ||
            shared[i]->owned_constant_bytes >= own->owned_constant_bytes ||
            shared[i]->ctc.packed_weights != root->ctc.packed_weights ||
            shared[i]->ctc.packed_weights_borrowed == 0u) {
            lw_x64_rec_program_free(own);
            goto done;
        }
        for (op_index = 0u; op_index < shared[i]->op_count; ++op_index) {
            const lw_x64_rec_op* op = &shared[i]->ops[op_index];
            const lw_x64_rec_op* base = source_op(root, op->semantic_begin);
            if (base == NULL || base->kind != op->kind) continue;
            if ((op->kind == LW_X64_REC_OP_POINTWISE ||
                 op->kind == LW_X64_REC_OP_DENSE ||
                 op->kind == LW_X64_REC_OP_DEPTHWISE) &&
                base->data.conv.packed_weights != NULL &&
                base->data.conv.original_weights == op->data.conv.original_weights) {
                if (op->data.conv.packed_weights != base->data.conv.packed_weights) {
                    lw_x64_rec_program_free(own);
                    goto done;
                }
                ++borrowed_conv;
            }
            if (op->kind == LW_X64_REC_OP_AFFINE &&
                base->data.affine.scale == op->data.affine.scale) {
                if (op->data.affine.mul != base->data.affine.mul ||
                    op->data.affine.add != base->data.affine.add) {
                    lw_x64_rec_program_free(own);
                    goto done;
                }
                ++borrowed_affine;
            }
            if (op->kind == LW_X64_REC_OP_MATMUL &&
                base->data.matmul.packed_weights != NULL &&
                base->data.matmul.weights == op->data.matmul.weights) {
                if (op->data.matmul.packed_weights != base->data.matmul.packed_weights) {
                    lw_x64_rec_program_free(own);
                    goto done;
                }
                ++borrowed_matmul;
            }
        }
        if (!run_pair(own, shared[i], &error)) {
            lw_x64_rec_program_free(own);
            goto done;
        }
        lw_x64_rec_program_free(own);
    }
    if (borrowed_conv == 0u || borrowed_affine == 0u ||
        shared_total >= independent_total / 2u) goto done;
    /* Drop the original root owner before running a borrower again. */
    lw_x64_rec_program_free(root);
    root = NULL;
    if (lw_x64_rec_backend_compile(model, 640u, &independent, &error) != LW_X64_REC_COMPILE_OK ||
        !run_pair(independent, shared[3], &error)) goto done;
    printf("REC shared constants: independent=%llu shared=%llu conv=%u affine=%u matmul=%u\n",
           (unsigned long long)independent_total, (unsigned long long)shared_total,
           borrowed_conv, borrowed_affine, borrowed_matmul);
    result = 0;
done:
    if (result != 0) fprintf(stderr, "REC constant sharing failed: %s\n", error.message);
    lw_x64_rec_program_free(independent);
    for (i = 0u; i < 4u; ++i) lw_x64_rec_program_free(shared[i]);
    lw_x64_rec_program_free(root);
    lw_model_free(model);
    return result;
}
