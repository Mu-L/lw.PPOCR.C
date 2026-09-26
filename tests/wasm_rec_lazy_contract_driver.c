#include "lw_infer.h"
#include "rec_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int recognize(lw_recognizer* recognizer, const uint8_t* source,
                     uint32_t width, uint32_t height, char* text,
                     uint64_t text_capacity) {
    lw_recognition_result result;
    lw_error error;
    lw_status status;
    lw_error_init(&error);
    lw_recognition_result_init(&result);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, source, (uint64_t)width * height * 3u, width, height,
        width * 3u, text, text_capacity, &result, &error);
    if (status != LW_STATUS_OK || result.required_text_capacity > text_capacity) {
        fprintf(stderr, "REC width %u failed: status=%d error=%s\n",
                width, (int)status, error.message);
        return 0;
    }
    return 1;
}

int main(void) {
    static const uint32_t widths[] = {192u, 320u, 480u, 640u, 960u};
    const uint32_t height = 48u;
    const uint32_t max_width = 960u * height / LW_REC_INPUT_HEIGHT;
    lw_recognizer_options options;
    lw_recognizer_info info;
    lw_recognizer* source = NULL;
    lw_recognizer* clone = NULL;
    lw_recognizer* fallback = NULL;
    lw_error error;
    uint8_t* pixels = NULL;
    char* source_text = NULL;
    char* clone_text = NULL;
    char* fallback_text = NULL;
    int success = 0;

    lw_recognizer_options_init(&options);
    options.target_width = 960u;
    lw_error_init(&error);
    if (lw_recognizer_create("/models/rec.lwm", "/models/ppocr_keys.txt",
                             &options, &source, &error) != LW_STATUS_OK) {
        fprintf(stderr, "REC create failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_recognizer_enable_adaptive_width(source, 1u, &error) != LW_STATUS_OK ||
        lw_recognizer_enable_resident_widths(source, &error) != LW_STATUS_OK ||
        lw_recognizer_clone(source, &clone, &error) != LW_STATUS_OK ||
        lw_recognizer_clone(source, &fallback, &error) != LW_STATUS_OK) {
        fprintf(stderr, "REC lazy clone failed: %s\n", error.message);
        goto cleanup;
    }
    if (lw_recognizer_test_has_canonical_session(source) ||
        lw_recognizer_test_has_canonical_session(clone) ||
        lw_recognizer_test_has_canonical_session(fallback)) {
        fprintf(stderr, "REC canonical session was retained despite full compiled coverage\n");
        goto cleanup;
    }
    /* This private hook drops all compiled slots. The next recognition must
     * construct a full canonical session from the released lazy state. */
    lw_recognizer_test_disable_x64_backend(fallback);
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(source, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "REC text capacity is invalid\n");
        goto cleanup;
    }
    pixels = (uint8_t*)malloc((size_t)max_width * height * 3u);
    source_text = (char*)malloc((size_t)info.max_text_capacity);
    clone_text = (char*)malloc((size_t)info.max_text_capacity);
    fallback_text = (char*)malloc((size_t)info.max_text_capacity);
    if (pixels == NULL || source_text == NULL || clone_text == NULL ||
        fallback_text == NULL) {
        fprintf(stderr, "REC contract allocation failed\n");
        goto cleanup;
    }
    memset(pixels, 255, (size_t)max_width * height * 3u);
    for (size_t i = 0u; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        const uint32_t width = widths[i] * height / LW_REC_INPUT_HEIGHT;
        if (lw_recognizer_target_width_for_image(source, width, height) != widths[i] ||
            !recognize(source, pixels, width, height, source_text,
                       info.max_text_capacity) ||
            !recognize(clone, pixels, width, height, clone_text,
                       info.max_text_capacity) ||
            !recognize(fallback, pixels, width, height, fallback_text,
                       info.max_text_capacity) ||
            !lw_recognizer_test_has_canonical_session(fallback) ||
            lw_recognizer_test_has_canonical_session(source) ||
            lw_recognizer_test_has_canonical_session(clone) ||
            lw_recognizer_current_target_width(fallback) != widths[i] ||
            strcmp(source_text, clone_text) != 0 ||
            strcmp(source_text, fallback_text) != 0) {
            fprintf(stderr, "REC lazy contract differs at target width %u\n", widths[i]);
            goto cleanup;
        }
    }
    puts("WASM_REC_LAZY_CONTRACT clone=pass fallback_rebuild=pass widths=5/5");
    success = 1;

cleanup:
    free(fallback_text);
    free(clone_text);
    free(source_text);
    free(pixels);
    lw_recognizer_free(fallback);
    lw_recognizer_free(clone);
    lw_recognizer_free(source);
    return success ? 0 : 1;
}
