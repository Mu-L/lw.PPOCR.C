#ifndef LW_OPERATOR_INTERNAL_H
#define LW_OPERATOR_INTERNAL_H

/*
 * LWM operator identifiers are an internal runtime contract. Keep the
 * numeric values in one place so execution, planning, and analysis cannot
 * silently drift apart. This header is private and is not part of the C ABI.
 */
enum {
    LW_OP_CONV = 1,
    LW_OP_ADD = 2,
    LW_OP_MUL = 3,
    LW_OP_DIV = 4,
    LW_OP_ERF = 5,
    LW_OP_HARD_SIGMOID = 6,
    LW_OP_BATCH_NORMALIZATION = 7,
    LW_OP_REDUCE_MEAN = 8,
    LW_OP_RELU = 9,
    LW_OP_AVERAGE_POOL = 10,
    LW_OP_SQUEEZE = 11,
    LW_OP_TRANSPOSE = 12,
    LW_OP_UNSQUEEZE = 13,
    LW_OP_MATMUL = 14,
    LW_OP_SOFTMAX = 15,
    LW_OP_RESHAPE = 16,
    LW_OP_CONCAT = 17,
    LW_OP_CONV_TRANSPOSE = 18,
    LW_OP_MAX_POOL = 19,
    LW_OP_RESIZE = 20,
    LW_OP_SIGMOID = 21,
    LW_OP_SUB = 22,
    LW_OP_SQRT = 23,
    LW_OP_POW = 24,
    LW_OP_SLICE = 25
};

#endif