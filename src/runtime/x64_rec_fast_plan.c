#include "x64_rec_fast_internal.h"

#include "lwm_read.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static size_t fast_align64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}



static int fast_tensor_eligible(const lw_runtime_tensor* tensor) {
    return tensor != NULL && (tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) == 0u &&
           tensor->dtype == LW_DTYPE_F32 && tensor->rank == 4u &&
           tensor->dimensions[0] > 0 && tensor->dimensions[1] > 0 &&
           tensor->dimensions[2] > 0 && tensor->dimensions[3] > 0;
}

static lw_status fast_allocate_tensor_twins(lw_x64_rec_fast_plan* plan, lw_error* error) {
    size_t total = 0u;
    if (plan == NULL || plan->session == NULL || plan->session->model == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    plan->tensor_count = plan->session->model->info.tensor_count;
    plan->tensors = (lw_x64_fast_tensor_state*)calloc(plan->tensor_count, sizeof(*plan->tensors));
    if (plan->tensors == NULL && plan->tensor_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast tensor states");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t index = 0u; index < plan->tensor_count; ++index) {
        const lw_runtime_tensor* tensor = &plan->session->tensors[index];
        lw_x64_fast_tensor_state* state = &plan->tensors[index];
        state->nhwc_offset = LW_X64_FAST_OFFSET_NONE;
        state->neutral = (uint8_t)lw_layout_tensor_is_neutral(plan->session, index);
        if (!fast_tensor_eligible(tensor) || state->neutral != 0u) {
            state->available_layouts = state->neutral != 0u
                ? (uint8_t)(LW_X64_FAST_HAVE_NCHW | LW_X64_FAST_HAVE_NHWC)
                : LW_X64_FAST_HAVE_NCHW;
            continue;
        }
        total = fast_align64(total);
        if (tensor->byte_size > SIZE_MAX - total) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "fast NHWC workspace size overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        state->nhwc_offset = (uint64_t)total;
        total += (size_t)tensor->byte_size;
        state->available_layouts = LW_X64_FAST_HAVE_NCHW;
    }
    plan->nhwc_workspace_bytes = fast_align64(total);
    if (plan->nhwc_workspace_bytes != 0u) {
        plan->nhwc_workspace = (uint8_t*)malloc(plan->nhwc_workspace_bytes);
        if (plan->nhwc_workspace == NULL) {
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast NHWC workspace");
            return LW_STATUS_OUT_OF_MEMORY;
        }
    }
    return LW_STATUS_OK;
}

lw_status lw_x64_rec_fast_plan_create(lw_session* session,
                                      lw_x64_rec_fast_plan** out_plan,
                                      lw_error* error) {
    lw_x64_rec_fast_plan* plan;
    lw_layout_planner_options options;
    lw_status status;
    if (out_plan == NULL || session == NULL || session->model == NULL) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "fast plan requires a session");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    *out_plan = NULL;
    plan = (lw_x64_rec_fast_plan*)calloc(1u, sizeof(*plan));
    if (plan == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast plan");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    plan->session = session;
    plan->node_count = session->model->info.node_count;
    plan->nodes = (lw_x64_fast_node*)calloc(plan->node_count, sizeof(*plan->nodes));
    if (plan->nodes == NULL && plan->node_count != 0u) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "unable to allocate fast nodes");
        lw_x64_rec_fast_plan_free(plan);
        return LW_STATUS_OUT_OF_MEMORY;
    }
    options.allow_direct_nhwc_graph_input = 0u;
    status = lw_layout_plan_build(session, &options, &plan->layout, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    status = fast_allocate_tensor_twins(plan, error);
    if (status != LW_STATUS_OK) {
        lw_x64_rec_fast_plan_free(plan);
        return status;
    }
    plan->graph_input_index = lwm_read_u32(session->model->bytes +
                                           (size_t)session->model->input_offset);
    plan->graph_output_index = lwm_read_u32(session->model->bytes +
                                            (size_t)session->model->output_offset);
    for (uint32_t node_index = 0u; node_index < plan->node_count; ++node_index) {
        const uint8_t* node = session->model->bytes + (size_t)session->model->node_offset +
                              (size_t)node_index * LWM_V0_NODE_SIZE;
        plan->nodes[node_index].semantic_node_index = node_index;
        plan->nodes[node_index].kind = LW_X64_FAST_NODE_GENERIC;
        plan->nodes[node_index].semantic_node_count = 1u;
        plan->nodes[node_index].output_index = lwm_read_u32(node + 40u);
    }
    plan->generic_node_count = plan->node_count;
    lw_set_error(error, LW_STATUS_OK, "");
    *out_plan = plan;
    return LW_STATUS_OK;
}

void lw_x64_rec_fast_plan_free(lw_x64_rec_fast_plan* plan) {
    if (plan == NULL) return;
    free(plan->scratch);
    free(plan->nhwc_workspace);
    free(plan->tensors);
    free(plan->nodes);
    lw_layout_plan_free(&plan->layout);
    free(plan);
}
