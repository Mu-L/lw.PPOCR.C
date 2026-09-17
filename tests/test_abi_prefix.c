#include "lw_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tail_is_unchanged(const unsigned char* bytes, size_t begin, size_t end) {
    size_t index;
    for (index = begin; index < end; ++index) {
        if (bytes[index] != 0xa5u) {
            return 0;
        }
    }
    return 1;
}

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

static int check_model_prefix(const char* path) {
    lw_model* model = NULL;
    lw_model_info info;
    lw_model_info oversized;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_model_load(path, NULL, &model, &error);
    if (status != LW_STATUS_OK || model == NULL) {
        return 0;
    }

    memset(&info, 0, sizeof(info));
    info.struct_size = 0u;
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_INVALID_ARGUMENT) {
        lw_model_free(model);
        return 0;
    }

    memset(&info, 0xa5, sizeof(info));
    info.struct_size = 8u;
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK || info.struct_size != 8u || info.format_major != 0u ||
        info.format_minor != 1u) {
        lw_model_free(model);
        return 0;
    }

    memset(&oversized, 0xa5, sizeof(oversized));
    oversized.struct_size = (uint32_t)(sizeof(oversized) + 16u);
    status = lw_model_get_info(model, &oversized);
    if (status != LW_STATUS_OK || oversized.struct_size != sizeof(oversized) + 16u ||
        oversized.tensor_count == 0u) {
        lw_model_free(model);
        return 0;
    }
    lw_model_free(model);
    return 1;
}

static int check_recognizer_prefix(const char* model_path, const char* dictionary_path) {
    lw_recognizer* recognizer = NULL;
    lw_recognizer_options options;
    lw_recognizer_info info;
    lw_recognizer_info oversized;
    lw_error error;
    lw_status status;

    lw_recognizer_options_init(&options);
    options.target_width = 80u;
    lw_error_init(&error);
    status = lw_recognizer_create(model_path, dictionary_path, &options, &recognizer, &error);
    if (status != LW_STATUS_OK || recognizer == NULL) {
        return 0;
    }

    memset(&info, 0xa5, sizeof(info));
    info.struct_size = 8u;
    status = lw_recognizer_get_info(recognizer, &info);
    if (status != LW_STATUS_OK || info.struct_size != 8u || info.target_width != 80u) {
        lw_recognizer_free(recognizer);
        return 0;
    }

    memset(&oversized, 0xa5, sizeof(oversized));
    oversized.struct_size = (uint32_t)(sizeof(oversized) + 16u);
    status = lw_recognizer_get_info(recognizer, &oversized);
    if (status != LW_STATUS_OK || oversized.struct_size != sizeof(oversized) + 16u ||
        oversized.time_steps == 0u || oversized.max_text_capacity == 0u) {
        lw_recognizer_free(recognizer);
        return 0;
    }
    lw_recognizer_free(recognizer);
    return 1;
}

static int check_recognizer_short_options(const char* model_path, const char* dictionary_path) {
    lw_recognizer* recognizer = NULL;
    lw_recognizer_options options;
    lw_error error;
    lw_status status;

    memset(&options, 0, sizeof(options));
    options.struct_size = 8u;
    options.target_width = 80u;
    lw_error_init(&error);
    status = lw_recognizer_create(model_path, dictionary_path, &options, &recognizer, &error);
    if (status != LW_STATUS_OK || recognizer == NULL) {
        return 0;
    }
    lw_recognizer_free(recognizer);
    return 1;
}

static int check_recognizer_result_prefix(const char* model_path, const char* dictionary_path) {
    static const uint8_t source[5u * 24u] = {0};
    lw_recognizer* recognizer = NULL;
    lw_recognizer_info info;
    lw_recognition_result result;
    lw_recognizer_options options;
    lw_error error;
    char* text = NULL;
    lw_status status;

    lw_recognizer_options_init(&options);
    options.target_width = 80u;
    lw_error_init(&error);
    status = lw_recognizer_create(model_path, dictionary_path, &options, &recognizer, &error);
    if (status != LW_STATUS_OK || recognizer == NULL) {
        return 0;
    }
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(recognizer, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || !allocation_fits(info.max_text_capacity, 1u)) {
        lw_recognizer_free(recognizer);
        return 0;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (text == NULL) {
        lw_recognizer_free(recognizer);
        return 0;
    }
    memset(&result, 0xa5, sizeof(result));
    result.struct_size = 8u;
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(recognizer, source, sizeof(source), 7u, 5u, 24u, text,
                                            info.max_text_capacity, &result, &error);
    if (status != LW_STATUS_OK || result.struct_size != 8u ||
        !tail_is_unchanged((const unsigned char*)&result, 8u, sizeof(result))) {
        free(text);
        lw_recognizer_free(recognizer);
        return 0;
    }
    free(text);
    lw_recognizer_free(recognizer);
    return 1;
}

static int check_classifier_prefix(const char* model_path) {
    lw_classifier* classifier = NULL;
    lw_classifier_info info;
    lw_classifier_info oversized;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_classifier_create(model_path, NULL, &classifier, &error);
    if (status != LW_STATUS_OK || classifier == NULL) {
        return 0;
    }

    memset(&info, 0xa5, sizeof(info));
    info.struct_size = 8u;
    status = lw_classifier_get_info(classifier, &info);
    if (status != LW_STATUS_OK || info.struct_size != 8u || info.input_width == 0u) {
        lw_classifier_free(classifier);
        return 0;
    }

    memset(&oversized, 0xa5, sizeof(oversized));
    oversized.struct_size = (uint32_t)(sizeof(oversized) + 16u);
    status = lw_classifier_get_info(classifier, &oversized);
    if (status != LW_STATUS_OK || oversized.struct_size != sizeof(oversized) + 16u ||
        oversized.class_count == 0u || oversized.workspace_size == 0u) {
        lw_classifier_free(classifier);
        return 0;
    }
    lw_classifier_free(classifier);
    return 1;
}

static int check_classifier_short_options(const char* model_path) {
    lw_classifier* classifier = NULL;
    lw_classifier_options options;
    lw_error error;
    lw_status status;

    memset(&options, 0, sizeof(options));
    options.struct_size = 4u;
    lw_error_init(&error);
    status = lw_classifier_create(model_path, &options, &classifier, &error);
    if (status != LW_STATUS_OK || classifier == NULL) {
        return 0;
    }
    lw_classifier_free(classifier);
    return 1;
}

static int check_classifier_result_prefix(const char* model_path) {
    static const uint8_t source[5u * 24u] = {0};
    lw_classifier* classifier = NULL;
    lw_classification_result result;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_classifier_create(model_path, NULL, &classifier, &error);
    if (status != LW_STATUS_OK || classifier == NULL) {
        return 0;
    }
    memset(&result, 0xa5, sizeof(result));
    result.struct_size = 8u;
    status = lw_classifier_classify_bgr_u8(classifier, source, sizeof(source), 7u, 5u, 24u,
                                           &result, &error);
    if (status != LW_STATUS_OK || result.struct_size != 8u || result.label > 1u ||
        !tail_is_unchanged((const unsigned char*)&result, 8u, sizeof(result))) {
        lw_classifier_free(classifier);
        return 0;
    }
    lw_classifier_free(classifier);
    return 1;
}

static int check_detector_prefix(const char* model_path) {
    lw_detector* detector = NULL;
    lw_detector_info info;
    lw_detector_info oversized;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_detector_create(model_path, NULL, &detector, &error);
    if (status != LW_STATUS_OK || detector == NULL) {
        return 0;
    }

    memset(&info, 0xa5, sizeof(info));
    info.struct_size = 8u;
    status = lw_detector_get_info(detector, &info);
    if (status != LW_STATUS_OK || info.struct_size != 8u || info.limit_side_length == 0u) {
        lw_detector_free(detector);
        return 0;
    }

    memset(&oversized, 0xa5, sizeof(oversized));
    oversized.struct_size = (uint32_t)(sizeof(oversized) + 16u);
    status = lw_detector_get_info(detector, &oversized);
    if (status != LW_STATUS_OK || oversized.struct_size != sizeof(oversized) + 16u ||
        oversized.max_candidates == 0u || oversized.max_image_pixels == 0u) {
        lw_detector_free(detector);
        return 0;
    }
    lw_detector_free(detector);
    return 1;
}

static int check_detector_short_options(const char* model_path) {
    lw_detector* detector = NULL;
    lw_detector_options options;
    lw_error error;
    lw_status status;

    memset(&options, 0, sizeof(options));
    options.struct_size = 8u;
    options.limit_side_length = 320u;
    lw_error_init(&error);
    status = lw_detector_create(model_path, &options, &detector, &error);
    if (status != LW_STATUS_OK || detector == NULL) {
        return 0;
    }
    lw_detector_free(detector);
    return 1;
}

static int check_detector_result_prefix(const char* model_path) {
    static const uint8_t source[32u * 32u * 3u] = {0};
    lw_detector* detector = NULL;
    lw_detector_info info;
    lw_detection_result result;
    lw_detection_box* boxes = NULL;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_detector_create(model_path, NULL, &detector, &error);
    if (status != LW_STATUS_OK || detector == NULL) {
        return 0;
    }
    lw_detector_info_init(&info);
    if (lw_detector_get_info(detector, &info) != LW_STATUS_OK || info.max_candidates == 0u ||
        !allocation_fits(info.max_candidates, sizeof(*boxes))) {
        lw_detector_free(detector);
        return 0;
    }
    boxes = (lw_detection_box*)malloc((size_t)info.max_candidates * sizeof(*boxes));
    if (boxes == NULL) {
        lw_detector_free(detector);
        return 0;
    }
    memset(&result, 0xa5, sizeof(result));
    result.struct_size = 8u;
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(detector, source, sizeof(source), 32u, 32u, 32u * 3u,
                                       boxes, info.max_candidates, &result, &error);
    if (status != LW_STATUS_OK || result.struct_size != 8u ||
        !tail_is_unchanged((const unsigned char*)&result, 8u, sizeof(result))) {
        free(boxes);
        lw_detector_free(detector);
        return 0;
    }
    free(boxes);
    lw_detector_free(detector);
    return 1;
}

static int check_ocr_prefix(const char* det_path, const char* cls_path, const char* rec_path,
                            const char* dictionary_path) {
    lw_ocr* ocr = NULL;
    lw_ocr_info info;
    lw_ocr_info oversized;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_ocr_create(det_path, cls_path, rec_path, dictionary_path, NULL, &ocr, &error);
    if (status != LW_STATUS_OK || ocr == NULL) {
        return 0;
    }

    memset(&info, 0xa5, sizeof(info));
    info.struct_size = 8u;
    status = lw_ocr_get_info(ocr, &info);
    if (status != LW_STATUS_OK || info.struct_size != 8u || info.use_direction_classification != 1u) {
        lw_ocr_free(ocr);
        return 0;
    }

    memset(&oversized, 0xa5, sizeof(oversized));
    oversized.struct_size = (uint32_t)(sizeof(oversized) + 16u);
    status = lw_ocr_get_info(ocr, &oversized);
    if (status != LW_STATUS_OK || oversized.struct_size != sizeof(oversized) + 16u ||
        oversized.max_line_capacity == 0u || oversized.max_text_capacity == 0u) {
        lw_ocr_free(ocr);
        return 0;
    }
    lw_ocr_free(ocr);
    return 1;
}

static int check_ocr_short_options(const char* det_path, const char* cls_path,
                                   const char* rec_path, const char* dictionary_path) {
    lw_ocr* ocr = NULL;
    lw_ocr_options options;
    lw_ocr_info info;
    lw_error error;
    lw_status status;

    memset(&options, 0, sizeof(options));
    options.struct_size = 8u;
    options.use_direction_classification = 0u;
    lw_error_init(&error);
    status = lw_ocr_create(det_path, cls_path, rec_path, dictionary_path, &options, &ocr, &error);
    if (status != LW_STATUS_OK || ocr == NULL) {
        return 0;
    }
    lw_ocr_info_init(&info);
    status = lw_ocr_get_info(ocr, &info);
    if (status != LW_STATUS_OK || info.use_direction_classification != 0u) {
        lw_ocr_free(ocr);
        return 0;
    }
    lw_ocr_free(ocr);
    return 1;
}

static int check_ocr_result_prefix(const char* det_path, const char* cls_path, const char* rec_path,
                                   const char* dictionary_path) {
    static const uint8_t source[32u * 32u * 3u] = {0};
    lw_ocr* ocr = NULL;
    lw_ocr_result result;
    lw_error error;
    lw_status status;

    lw_error_init(&error);
    status = lw_ocr_create(det_path, cls_path, rec_path, dictionary_path, NULL, &ocr, &error);
    if (status != LW_STATUS_OK || ocr == NULL) {
        return 0;
    }
    memset(&result, 0xa5, sizeof(result));
    result.struct_size = 8u;
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, source, sizeof(source), 32u, 32u, 32u * 3u, NULL, 0u, NULL,
                               0u, &result, &error);
    if (status != LW_STATUS_OK || result.struct_size != 8u ||
        !tail_is_unchanged((const unsigned char*)&result, 8u, sizeof(result))) {
        lw_ocr_free(ocr);
        return 0;
    }
    lw_ocr_free(ocr);
    return 1;
}

int main(int argc, char** argv) {
    lw_model_info invalid;
    lw_status status;

    if (argc != 5) {
        fprintf(stderr, "expected det.lwm cls.lwm rec.lwm dictionary\n");
        return 2;
    }

    memset(&invalid, 0, sizeof(invalid));
    invalid.struct_size = 0u;
    status = lw_model_get_info(NULL, &invalid);
    if (status != LW_STATUS_INVALID_ARGUMENT) {
        return 1;
    }
    if (!check_model_prefix(argv[3])) {
        fprintf(stderr, "model prefix check failed\n");
        return 1;
    }
    if (!check_recognizer_prefix(argv[3], argv[4])) {
        fprintf(stderr, "recognizer prefix check failed\n");
        return 1;
    }
    if (!check_recognizer_short_options(argv[3], argv[4])) {
        fprintf(stderr, "recognizer short options check failed\n");
        return 1;
    }
    if (!check_recognizer_result_prefix(argv[3], argv[4])) {
        fprintf(stderr, "recognizer result prefix check failed\n");
        return 1;
    }
    if (!check_classifier_prefix(argv[2])) {
        fprintf(stderr, "classifier prefix check failed\n");
        return 1;
    }
    if (!check_classifier_short_options(argv[2])) {
        fprintf(stderr, "classifier short options check failed\n");
        return 1;
    }
    if (!check_classifier_result_prefix(argv[2])) {
        fprintf(stderr, "classifier result prefix check failed\n");
        return 1;
    }
    if (!check_detector_prefix(argv[1])) {
        fprintf(stderr, "detector prefix check failed\n");
        return 1;
    }
    if (!check_detector_short_options(argv[1])) {
        fprintf(stderr, "detector short options check failed\n");
        return 1;
    }
    if (!check_detector_result_prefix(argv[1])) {
        fprintf(stderr, "detector result prefix check failed\n");
        return 1;
    }
    if (!check_ocr_prefix(argv[1], argv[2], argv[3], argv[4])) {
        fprintf(stderr, "ABI output prefix compatibility check failed\n");
        return 1;
    }
    if (!check_ocr_short_options(argv[1], argv[2], argv[3], argv[4])) {
        fprintf(stderr, "OCR short options check failed\n");
        return 1;
    }
    if (!check_ocr_result_prefix(argv[1], argv[2], argv[3], argv[4])) {
        fprintf(stderr, "OCR result prefix check failed\n");
        return 1;
    }
    return 0;
}
