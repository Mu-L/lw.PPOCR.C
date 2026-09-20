#include "crop_buffer_internal.h"

#include <assert.h>
#include <stdint.h>

static void test_grow_and_reuse(void) {
    lw_crop_buffer buffer;
    lw_error error;
    uint8_t* first;
    uint64_t capacity;

    lw_error_init(&error);
    lw_crop_buffer_init(&buffer, UINT64_C(4096));
    assert(lw_crop_buffer_reserve(&buffer, UINT64_C(1024), &error) == LW_STATUS_OK);
    first = buffer.data;
    capacity = buffer.capacity;
    assert(first != NULL);
    assert(capacity >= UINT64_C(4096));
    assert(lw_crop_buffer_reserve(&buffer, UINT64_C(512), &error) == LW_STATUS_OK);
    assert(buffer.data == first);
    assert(buffer.capacity == capacity);
    assert(lw_crop_buffer_reserve(&buffer, UINT64_C(100000), &error) == LW_STATUS_OK);
    assert(buffer.capacity >= UINT64_C(100000));
    lw_crop_buffer_free(&buffer);
    assert(buffer.data == NULL);
    assert(buffer.capacity == 0u);
    assert(buffer.retained_limit == 0u);
}

static void test_trim(void) {
    lw_crop_buffer buffer;
    lw_error error;

    lw_error_init(&error);
    lw_crop_buffer_init(&buffer, UINT64_C(4096));
    assert(lw_crop_buffer_reserve(&buffer, UINT64_C(1) << 20, &error) == LW_STATUS_OK);
    assert(buffer.capacity >= (UINT64_C(1) << 20));
    lw_crop_buffer_trim_after_request(&buffer);
    assert(buffer.data == NULL);
    assert(buffer.capacity == 0u);
    lw_crop_buffer_free(&buffer);
}

static void test_limit_zero_keeps_buffer(void) {
    lw_crop_buffer buffer;
    lw_error error;

    lw_error_init(&error);
    lw_crop_buffer_init(&buffer, 0u);
    assert(lw_crop_buffer_reserve(&buffer, UINT64_C(4096), &error) == LW_STATUS_OK);
    assert(buffer.data != NULL);
    lw_crop_buffer_trim_after_request(&buffer);
    assert(buffer.data != NULL);
    lw_crop_buffer_free(&buffer);
}

int main(void) {
    test_grow_and_reuse();
    test_trim();
    test_limit_zero_keeps_buffer();
    return 0;
}