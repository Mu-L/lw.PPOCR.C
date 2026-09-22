#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "model_internal.h"
#include "rec_internal.h"
#include "session_internal.h"
#include "executor_internal.h"
#include "lwm_read.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t node_output(const lw_model* model, uint32_t node_index) {
    const uint8_t* node = model->bytes + (size_t)model->node_offset +
                          (size_t)node_index * LWM_V0_NODE_SIZE;
    return lwm_read_u32(node + 40u);
}
static int fail(const char* message, const lw_error* error) {
    if (error != NULL && error->message[0] != '\0') {
        fprintf(stderr, "%s: %s\n", message, error->message);
    } else {
        fprintf(stderr, "%s\n", message);
    }
    return 1;
}

int main(int argc, char** argv) {
    const uint32_t width = 960u;
    const uint32_t height = 48u;
    const uint32_t source_stride = width * 3u;
    const uint64_t source_bytes = (uint64_t)source_stride * height;

    lw_model* model = NULL;
    lw_x64_rec_program* program = NULL;
    lw_x64_rec_instance* instance = NULL;
    lw_session* canonical = NULL;
    lw_session* canonical_backbone = NULL;
    lw_tensor_desc input_desc;
    lw_error error;
    lw_x64_rec_compile_result compile_result;
    uint8_t* source = NULL;
    float* canonical_input = NULL;
    uint32_t* canonical_indices = NULL;
    float* canonical_probabilities = NULL;
    uint64_t canonical_input_count = (uint64_t)3u * height * width;
    uint64_t backend_input_count = 0u;
    float* backend_input;
    uint32_t t;
    uint32_t canonical_node_cursor = 0u;
    int return_code = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: x64-rec-backend-execution-driver rec.lwm\n");
        return 2;
    }
    lw_error_init(&error);
    if (lw_model_load(argv[1], NULL, &model, &error) != LW_STATUS_OK) {
        return fail("model load failed", &error);
    }
    compile_result = lw_x64_rec_backend_compile(model, width, &program, &error);
    if (compile_result != LW_X64_REC_COMPILE_OK || program == NULL) {
        fprintf(stderr, "backend compile failed: result=%u\n", (unsigned)compile_result);
        return_code = fail("backend compile failed", &error);
        goto cleanup;
    }
    if (lw_x64_rec_instance_create(program, &instance, &error) != LW_STATUS_OK || instance == NULL) {
        return_code = fail("backend instance create failed", &error);
        goto cleanup;
    }
    backend_input = lw_x64_rec_instance_input(instance, &backend_input_count);
    if (backend_input == NULL || backend_input_count != canonical_input_count) {
        fprintf(stderr, "unexpected backend input count: %llu (expected %llu)\n",
                (unsigned long long)backend_input_count, (unsigned long long)canonical_input_count);
        goto cleanup;
    }

    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
    if (lw_session_create(model, &input_desc, 1u, NULL, &canonical_backbone, &error) != LW_STATUS_OK ||
        lw_session_create_ctc_greedy(model, &input_desc, 1u, NULL, &canonical, &error) !=
            LW_STATUS_OK) {
        return_code = fail("canonical session create failed", &error);
        goto cleanup;
    }
    source = (uint8_t*)malloc((size_t)source_bytes);
    canonical_input = (float*)malloc((size_t)canonical_input_count * sizeof(float));
    canonical_indices = (uint32_t*)malloc((size_t)program->time_steps * sizeof(uint32_t));
    canonical_probabilities = (float*)malloc((size_t)program->time_steps * sizeof(float));
    if (source == NULL || canonical_input == NULL || canonical_indices == NULL ||
        canonical_probabilities == NULL) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < source_bytes; ++i) {
        source[i] = (uint8_t)((i * 37u + i / source_stride * 13u + 17u) & 0xffu);
    }
    if (lw_rec_preprocess_bgr_u8(source, source_bytes, width, height, source_stride, width,
                                 canonical_input, canonical_input_count, NULL) != LW_STATUS_OK ||
        lw_rec_preprocess_bgr_u8_nhwc(source, source_bytes, width, height, source_stride, width,
                                      backend_input, backend_input_count, NULL) != LW_STATUS_OK) {
        return_code = fail("preprocess failed", NULL);
        goto cleanup;
    }
    for (uint32_t oi = 0u; oi < program->op_count; ++oi) {
        uint32_t semantic = program->ops[oi].semantic_begin;
        uint32_t semantic_end = semantic + program->ops[oi].semantic_count - 1u;
        uint32_t output_index = node_output(model, semantic_end);
        const float* expected;
        const float* actual;
        uint64_t count = program->values[output_index].bytes / sizeof(float);
        float max_difference = 0.0f;
        while (canonical_node_cursor <= semantic_end) {
            if (lw_executor_dispatch_node_f32(canonical_backbone, canonical_node_cursor, 0u,
                                               canonical_input, NULL) != LW_STATUS_OK) {
                fprintf(stderr, "canonical backbone node %u failed\n", canonical_node_cursor);
                goto cleanup;
            }
            ++canonical_node_cursor;
        }
        if (lw_x64_rec_instance_run_op(instance, oi, &error) != LW_STATUS_OK) {
            fprintf(stderr, "node %u execution failed\n", semantic);
            goto cleanup;
        }
        expected = lw_executor_tensor_output_data(canonical_backbone, output_index);
        actual = (const float*)(const void*)(instance->arena +
                                             (size_t)program->values[output_index].offset);
        if (expected == NULL || actual == NULL) {
            fprintf(stderr, "node %u output unavailable\n", semantic);
            goto cleanup;
        }
        if (canonical_backbone->tensors[output_index].rank == 4u) {
            uint32_t channels = (uint32_t)canonical_backbone->tensors[output_index].dimensions[1];
            uint32_t height_value = (uint32_t)canonical_backbone->tensors[output_index].dimensions[2];
            uint32_t width_value = (uint32_t)canonical_backbone->tensors[output_index].dimensions[3];
            for (uint32_t y = 0u; y < height_value; ++y) {
                for (uint32_t x = 0u; x < width_value; ++x) {
                    for (uint32_t c = 0u; c < channels; ++c) {
                        uint64_t nhwc_index = ((uint64_t)y * width_value + x) * channels + c;
                        uint64_t nchw_index = ((uint64_t)c * height_value + y) * width_value + x;
                        float difference = fabsf(expected[nchw_index] - actual[nhwc_index]);
                        if (difference > max_difference) max_difference = difference;
                    }
                }
            }
        } else {
            for (uint64_t element = 0u; element < count; ++element) {
                float difference = fabsf(expected[element] - actual[element]);
                if (difference > max_difference) max_difference = difference;
            }
        }
        if (max_difference > 1.0e-3f) {
            fprintf(stderr, "backbone mismatch at op=%u node=%u kind=%u max_abs=%.9g\n",
                    oi, semantic, (unsigned)program->ops[oi].kind, max_difference);
            goto cleanup;
        }
    }
    if (lw_execute_session_f32_ctc_greedy(canonical, canonical_input, canonical_input_count,
                                           canonical_indices, canonical_probabilities,
                                           program->time_steps, program->class_count, NULL,
                                           &error) != LW_STATUS_OK) {
        return_code = fail("canonical execution failed", &error);
        goto cleanup;
    }
    {
        const float* canonical_activation =
            lw_executor_tensor_output_data(canonical_backbone, program->ctc.activation_value);
        if (canonical_activation == NULL ||
            lw_x64_rec_ctc_execute(program, instance, canonical_activation, &error) != LW_STATUS_OK) {
            return_code = fail("CTC projection from canonical activation failed", &error);
            goto cleanup;
        }
        for (t = 0u; t < program->time_steps; ++t) {
            float difference = fabsf(canonical_probabilities[t] - instance->best_probabilities[t]);
            if (canonical_indices[t] != instance->best_indices[t] || difference > 1.0e-6f) {
                fprintf(stderr, "CTC projection mismatch at t=%u: canonical=(%u,%.9g) backend=(%u,%.9g) diff=%.9g\n",
                        t, canonical_indices[t], canonical_probabilities[t], instance->best_indices[t],
                        instance->best_probabilities[t], difference);
                goto cleanup;
            }
        }
    }
    if (lw_x64_rec_instance_run(instance, &error) != LW_STATUS_OK) {
        return_code = fail("backend execution failed", &error);
        goto cleanup;
    }
    for (t = 0u; t < program->time_steps; ++t) {
        float difference = fabsf(canonical_probabilities[t] - instance->best_probabilities[t]);
        if (canonical_indices[t] != instance->best_indices[t] || difference > 1.0e-6f) {
            fprintf(stderr, "CTC mismatch at t=%u: canonical=(%u,%.9g) backend=(%u,%.9g) diff=%.9g\n",
                    t, canonical_indices[t], canonical_probabilities[t], instance->best_indices[t],
                    instance->best_probabilities[t], difference);
            goto cleanup;
        }
    }
    printf("x64 REC backend matches canonical CTC: width=%u time_steps=%u classes=%u\n",
           width, program->time_steps, program->class_count);
    return_code = 0;

cleanup:
    free(canonical_probabilities);
    free(canonical_indices);
    free(canonical_input);
    free(source);
    lw_session_free(canonical_backbone);
    lw_session_free(canonical);
    lw_x64_rec_instance_free(instance);
    lw_x64_rec_program_free(program);
    lw_model_free(model);
    return return_code;
}
