#ifndef LW_X64_REC_FAST_INTERNAL_H
#define LW_X64_REC_FAST_INTERNAL_H

#include "lw_infer.h"
#include "session_internal.h"
#include "layout_planner_internal.h"
#include "../kernels/nhwc_internal.h"

#include <stddef.h>
#include <stdint.h>

#define LW_X64_FAST_OFFSET_NONE UINT64_MAX

enum {
    LW_X64_FAST_HAVE_NCHW = 1u << 0,
    LW_X64_FAST_HAVE_NHWC = 1u << 1
};

typedef enum lw_x64_fast_node_kind {
    LW_X64_FAST_NODE_GENERIC = 0,
    LW_X64_FAST_NODE_POINTWISE = 1,
    LW_X64_FAST_NODE_DENSE = 2,
    LW_X64_FAST_NODE_DEPTHWISE = 3,
    LW_X64_FAST_NODE_CONTIGUOUS_UNARY = 4,
    LW_X64_FAST_NODE_CONTIGUOUS_BINARY = 5,
    LW_X64_FAST_NODE_BATCH_NORM = 6,
    LW_X64_FAST_NODE_REDUCE_MEAN = 7,
    LW_X64_FAST_NODE_POOL = 8,
    LW_X64_FAST_NODE_RESIZE = 9,
    LW_X64_FAST_NODE_CONCAT = 10,
    LW_X64_FAST_NODE_LAYER_NORM = 11
} lw_x64_fast_node_kind;

typedef struct lw_x64_fast_tensor_state {
    uint64_t nhwc_offset;
    uint8_t available_layouts;
    uint8_t neutral;
    uint16_t reserved;
} lw_x64_fast_tensor_state;

typedef struct lw_x64_fast_conv {
    uint32_t input_index;
    uint32_t output_index;
    uint32_t weight_index;
    uint32_t bias_index;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t dense_kc;
    float* packed_weights;
    uint64_t packed_weight_count;
    const float* bias;
    uint64_t scratch_bytes;
} lw_x64_fast_conv;

typedef struct lw_x64_fast_elementwise {
    uint32_t input_index;
    uint32_t rhs_index;
    uint32_t output_index;
    uint16_t operation;
    uint16_t reserved;
} lw_x64_fast_elementwise;
typedef struct lw_x64_fast_node {
    uint32_t semantic_node_index;
    uint16_t kind;
    uint16_t semantic_node_count;
    uint32_t output_index;
    union {
        lw_x64_fast_conv conv;
        lw_x64_fast_elementwise elementwise;
    } data;
} lw_x64_fast_node;

typedef struct lw_x64_rec_fast_plan {
    lw_session* session;
    lw_layout_plan layout;
    lw_x64_fast_node* nodes;
    uint32_t node_count;
    lw_x64_fast_tensor_state* tensors;
    uint32_t tensor_count;
    uint8_t* nhwc_workspace;
    size_t nhwc_workspace_bytes;
    uint8_t* scratch;
    size_t scratch_bytes;
    uint32_t graph_input_index;
    uint32_t graph_output_index;
    uint32_t fast_node_count;
    uint32_t generic_node_count;
    uint32_t pointwise_node_count;
    uint32_t dense_node_count;
    uint32_t depthwise_node_count;
    uint32_t binary_node_count;
    uint32_t relu_node_count;
    uint64_t conversion_count;
    uint64_t conversion_bytes;
} lw_x64_rec_fast_plan;

lw_status lw_x64_rec_fast_plan_create(lw_session* session,
                                      lw_x64_rec_fast_plan** out_plan,
                                      lw_error* error);
void lw_x64_rec_fast_plan_free(lw_x64_rec_fast_plan* plan);
lw_status lw_x64_rec_fast_run(lw_x64_rec_fast_plan* plan,
                              const float* input,
                              uint64_t input_element_count,
                              float* output,
                              uint64_t output_element_count,
                              lw_error* error);

void lw_x64_fast_nchw_to_nhwc(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width);
void lw_x64_fast_nhwc_to_nchw(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width);

#endif
