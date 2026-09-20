#ifndef LW_LAYOUT_PLANNER_INTERNAL_H
#define LW_LAYOUT_PLANNER_INTERNAL_H

#include "lw_infer.h"

#include <stdint.h>

typedef enum lw_fast_layout {
    LW_FAST_LAYOUT_NCHW = 0,
    LW_FAST_LAYOUT_NHWC = 1
} lw_fast_layout;

enum {
    LW_LAYOUT_AVAILABLE_NCHW = 1u << 0,
    LW_LAYOUT_AVAILABLE_NHWC = 1u << 1
};

typedef struct lw_layout_planner_options {
    uint8_t allow_direct_nhwc_graph_input;
} lw_layout_planner_options;

typedef struct lw_layout_plan {
    uint32_t tensor_count;
    uint32_t node_count;
    uint8_t* tensor_layout;
    uint8_t* tensor_available_layouts;
    uint8_t* node_layout;
    uint32_t nhwc_node_count;
    uint32_t nchw_node_count;
    uint32_t layout_conversion_count;
    uint32_t nhwc_island_count;
    uint8_t graph_input_direct_nhwc;
} lw_layout_plan;

lw_status lw_layout_plan_build(const lw_session* session,
                               const lw_layout_planner_options* options,
                               lw_layout_plan* plan, lw_error* error);
void lw_layout_plan_free(lw_layout_plan* plan);
int lw_layout_tensor_is_neutral(const lw_session* session, uint32_t tensor_index);
int lw_layout_node_nhwc_capable(const lw_session* session, uint32_t node_index);

#endif