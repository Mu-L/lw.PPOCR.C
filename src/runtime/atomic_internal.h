#ifndef LW_ATOMIC_INTERNAL_H
#define LW_ATOMIC_INTERNAL_H

/* Small internal-only acquire/release counter used by shared prepared data. */

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

typedef volatile uint32_t lw_atomic_u32;

static inline void lw_atomic_u32_init(lw_atomic_u32* value, uint32_t initial) {
    if (value != NULL) {
        *value = initial;
    }
}

static inline uint32_t lw_atomic_u32_load_acquire(const lw_atomic_u32* value) {
#if defined(_MSC_VER)
    return (uint32_t)_InterlockedCompareExchange((volatile long*)value, 0L, 0L);
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
#else
#  error "No atomic implementation for this compiler."
#endif
}

static inline int lw_atomic_u32_compare_exchange_acq_rel(lw_atomic_u32* value,
                                                           uint32_t* expected,
                                                           uint32_t desired) {
#if defined(_MSC_VER)
    const long observed = _InterlockedCompareExchange((volatile long*)value,
                                                       (long)desired, (long)*expected);
    if ((uint32_t)observed == *expected) {
        return 1;
    }
    *expected = (uint32_t)observed;
    return 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_compare_exchange_n(value, expected, desired, 0, __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
#else
#  error "No atomic implementation for this compiler."
#endif
}

#endif
