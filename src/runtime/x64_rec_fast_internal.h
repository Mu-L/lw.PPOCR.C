#ifndef LW_X64_REC_FAST_INTERNAL_H
#define LW_X64_REC_FAST_INTERNAL_H

#include "lw_infer.h"
#include "session_internal.h"
#include "layout_planner_internal.h"
#include "../kernels/nhwc_internal.h"

#include <stddef.h>
#include <stdint.h>

#define LW_X64_FAST_OFFSET_NONE UINT64_MAX
#define LW_X64_FAST_OPERATOR_CAPACITY 64u
#define LW_X64_FAST_PROFILE_NODE_CAPACITY 512u
#define LW_X64_FAST_PROFILE_KIND_CAPACITY 16u

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
    LW_X64_FAST_NODE_LAYER_NORM = 11,
    LW_X64_FAST_NODE_GELU = 12
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
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t dense_kc;
    float* packed_weights;
    uint64_t packed_weight_count;
    const float* bias;
    uint64_t scratch_bytes;
} lw_x64_fast_conv;

typedef enum lw_x64_fast_broadcast_kind {
    LW_X64_FAST_BROADCAST_NONE = 0,
    LW_X64_FAST_BROADCAST_RIGHT_SCALAR = 1,
    LW_X64_FAST_BROADCAST_LEFT_SCALAR = 2,
    LW_X64_FAST_BROADCAST_RIGHT_CHANNEL = 3,
    LW_X64_FAST_BROADCAST_LEFT_CHANNEL = 4
} lw_x64_fast_broadcast_kind;

typedef struct lw_x64_fast_elementwise {
    uint32_t input_index;
    uint32_t rhs_index;
    uint32_t output_index;
    uint16_t operation;
    uint8_t broadcast_kind;
    uint8_t reserved;
    uint32_t channels;
    float alpha;
    float beta;
} lw_x64_fast_elementwise;
typedef struct lw_x64_fast_spatial {
    uint32_t input_index;
    uint32_t output_index;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint8_t count_include_pad;
    uint8_t is_max;
    uint16_t reserved;
    uint32_t input_count;
    uint32_t input_indices[LWM_V0_MAX_NODE_INPUTS];
    uint32_t input_channels[LWM_V0_MAX_NODE_INPUTS];
    const float* scale;
    const float* bias;
    const float* mean;
    const float* variance;
    float epsilon;
} lw_x64_fast_spatial;
typedef struct lw_x64_fast_node {
    uint32_t semantic_node_index;
    uint16_t kind;
    uint16_t semantic_node_count;
    uint32_t output_index;
    union {
        lw_x64_fast_conv conv;
        lw_x64_fast_elementwise elementwise;
        lw_x64_fast_spatial spatial;
    } data;
} lw_x64_fast_node;

typedef uint64_t (*lw_x64_fast_profile_clock)(void* context);
typedef struct lw_x64_fast_profile {
    lw_x64_fast_profile_clock clock;
    void* clock_context;
    uint64_t node_nanoseconds[LW_X64_FAST_PROFILE_NODE_CAPACITY];
    uint64_t node_invocations[LW_X64_FAST_PROFILE_NODE_CAPACITY];
    uint64_t kind_nanoseconds[LW_X64_FAST_PROFILE_KIND_CAPACITY];
    uint64_t kind_invocations[LW_X64_FAST_PROFILE_KIND_CAPACITY];
    uint64_t conversion_nanoseconds;
    uint64_t conversion_invocations;
    uint64_t conversion_bytes;
    uint64_t total_nanoseconds;
} lw_x64_fast_profile;
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
    uint32_t unary_node_count;
    uint32_t reduce_mean_node_count;
    uint32_t pool_node_count;
    uint32_t concat_node_count;
    uint32_t batch_norm_node_count;
    uint32_t unsupported_nhwc_node_count;
    uint32_t unsupported_nhwc_by_operator[LW_X64_FAST_OPERATOR_CAPACITY];
    uint64_t conversion_count;
    uint64_t conversion_bytes;
} lw_x64_rec_fast_plan;

lw_status lw_x64_rec_fast_plan_create(lw_session* session,
                                      lw_x64_rec_fast_plan** out_plan,
                                      lw_error* error);
void lw_x64_rec_fast_plan_free(lw_x64_rec_fast_plan* plan);
lw_status lw_x64_rec_fast_run_profiled(lw_x64_rec_fast_plan* plan,
                                     const float* input, uint64_t input_element_count,
                                     float* output, uint64_t output_element_count,
                                     lw_x64_fast_profile* profile, lw_error* error);
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
