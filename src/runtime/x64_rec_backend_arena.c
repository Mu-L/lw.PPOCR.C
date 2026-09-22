#include "x64_rec_backend_internal.h"
#include <limits.h>

uint64_t lw_x64_rec_arena_align(uint64_t value, uint64_t alignment) {
    uint64_t mask;
    if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) return UINT64_MAX;
    mask = alignment - 1u;
    if (value > UINT64_MAX - mask) return UINT64_MAX;
    return (value + mask) & ~mask;
}

lw_status lw_x64_rec_arena_alloc(uint64_t* cursor, uint64_t bytes, uint64_t alignment,
                                 uint64_t* out_offset, lw_error* error) {
    uint64_t start;
    if (cursor == NULL || out_offset == NULL || alignment == 0u) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "invalid REC arena allocation");
        return LW_STATUS_INVALID_ARGUMENT;
    }
    start = lw_x64_rec_arena_align(*cursor, alignment);
    if (start == UINT64_MAX || bytes > UINT64_MAX - start) {
        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "REC arena size overflows");
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    *out_offset = start;
    *cursor = start + bytes;
    return LW_STATUS_OK;
}
