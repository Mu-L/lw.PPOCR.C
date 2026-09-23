#include "rec_internal.h"
#include "lw_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <time.h>
#endif

#define DEFAULT_REPEATS 10u
#define WARMUP_ROUNDS 2u
#define MAX_REPEATS 100u

static uint64_t clock_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency) ||
        frequency.QuadPart <= 0) {
        return 0u;
    }
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 /
                      (double)frequency.QuadPart);
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static int parse_repeats(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL) return 0;
    parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0u || parsed > MAX_REPEATS) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static double median(double* values, uint32_t count) {
    uint32_t i;
    uint32_t j;
    if (count == 0u) return 0.0;
    for (i = 1u; i < count; ++i) {
        double current = values[i];
        j = i;
        while (j > 0u && values[j - 1u] > current) {
            values[j] = values[j - 1u];
            --j;
        }
        values[j] = current;
    }
    if ((count & 1u) != 0u) return values[count / 2u];
    return (values[count / 2u - 1u] + values[count / 2u]) * 0.5;
}

static void fill_source(uint8_t* source, uint32_t width, uint32_t height) {
    uint64_t count = (uint64_t)width * height * 3u;
    uint64_t i;
    for (i = 0u; i < count; ++i) {
        source[i] = (uint8_t)((i * 37u + i / (uint64_t)(width * 3u) * 13u + 17u) & 0xffu);
    }
}

static int run_recognizer(lw_recognizer* recognizer, const uint8_t* source,
                          uint64_t source_bytes, uint32_t width, uint32_t height,
                          char* text, uint64_t text_capacity, double* elapsed_ms) {
    lw_recognition_result result;
    lw_error error;
    uint64_t start = clock_ns();
    lw_status status;
    lw_error_init(&error);
    lw_recognition_result_init(&result);
    status = lw_recognizer_recognize_bgr_u8(recognizer, source, source_bytes, width, height,
                                            width * 3u, text, text_capacity, &result, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "recognizer run failed: %s\n", error.message);
        return 0;
    }
    if (elapsed_ms != NULL) *elapsed_ms = (double)(clock_ns() - start) / 1000000.0;
    return 1;
}

int main(int argc, char** argv) {
    uint32_t width = 960u;
    const uint32_t height = 48u;
    uint32_t repeats = DEFAULT_REPEATS;
    uint64_t source_bytes;
    lw_recognizer_options options;
    lw_recognizer* backend_recognizer = NULL;
    lw_recognizer* canonical_recognizer = NULL;
    lw_error error;
    uint8_t* source = NULL;
    char* backend_text = NULL;
    char* canonical_text = NULL;
    double* canonical_samples = NULL;
    double* backend_samples = NULL;
    uint32_t sample_count = 0u;
    uint32_t warmup;
    uint32_t repeat;
    int result = 1;

    if (argc < 3 || argc > 5 || (argc >= 4 && !parse_repeats(argv[3], &repeats))) {
        fprintf(stderr, "usage: x64-rec-recognizer-benchmark-driver rec.lwm dict.txt [repeats] [width]\n");
        return 2;
    }
    if (argc == 5) {
        width = (uint32_t)strtoul(argv[4], NULL, 10);
        if (width != 192u && width != 320u && width != 480u &&
            width != 640u && width != 960u) return 2;
    }
    source_bytes = (uint64_t)width * height * 3u;
    lw_error_init(&error);
    lw_recognizer_options_init(&options);
    options.target_width = width;
    if (lw_recognizer_create(argv[1], argv[2], &options, &backend_recognizer, &error) !=
            LW_STATUS_OK ||
        lw_recognizer_create(argv[1], argv[2], &options, &canonical_recognizer, &error) !=
            LW_STATUS_OK) {
        fprintf(stderr, "recognizer create failed: %s\n", error.message);
        goto cleanup;
    }
#if defined(LW_EXPERIMENTAL_AVX2_FAST_PATH)
    lw_recognizer_test_disable_x64_backend(canonical_recognizer);
#endif
    source = (uint8_t*)malloc((size_t)source_bytes);
    backend_text = (char*)malloc(1024u);
    canonical_text = (char*)malloc(1024u);
    canonical_samples = (double*)malloc((size_t)repeats * sizeof(double));
    backend_samples = (double*)malloc((size_t)repeats * sizeof(double));
    if (source == NULL || backend_text == NULL || canonical_text == NULL ||
        canonical_samples == NULL || backend_samples == NULL) {
        fprintf(stderr, "benchmark allocation failed\n");
        goto cleanup;
    }
    fill_source(source, width, height);
    for (warmup = 0u; warmup < WARMUP_ROUNDS; ++warmup) {
        double ignored;
        if (!run_recognizer(backend_recognizer, source, source_bytes, width, height,
                            backend_text, 1024u, &ignored) ||
            !run_recognizer(canonical_recognizer, source, source_bytes, width, height,
                            canonical_text, 1024u, &ignored) ||
            strcmp(backend_text, canonical_text) != 0) {
            fprintf(stderr, "warmup text mismatch: backend=\"%s\" canonical=\"%s\"\n",
                    backend_text, canonical_text);
            goto cleanup;
        }
    }
    for (repeat = 0u; repeat < repeats; ++repeat) {
        double canonical_ms;
        double backend_ms;
        int backend_first = (repeat & 1u) == 0u;
        if (backend_first) {
            if (!run_recognizer(backend_recognizer, source, source_bytes, width, height,
                                backend_text, 1024u, &backend_ms) ||
                !run_recognizer(canonical_recognizer, source, source_bytes, width, height,
                                canonical_text, 1024u, &canonical_ms)) {
                goto cleanup;
            }
        } else {
            if (!run_recognizer(canonical_recognizer, source, source_bytes, width, height,
                                canonical_text, 1024u, &canonical_ms) ||
                !run_recognizer(backend_recognizer, source, source_bytes, width, height,
                                backend_text, 1024u, &backend_ms)) {
                goto cleanup;
            }
        }
        if (strcmp(backend_text, canonical_text) != 0) {
            fprintf(stderr, "text mismatch: backend=\"%s\" canonical=\"%s\"\n",
                    backend_text, canonical_text);
            goto cleanup;
        }
        backend_samples[sample_count] = backend_ms;
        canonical_samples[sample_count] = canonical_ms;
        ++sample_count;
    }
    {
        double backend_median = median(backend_samples, sample_count);
        double canonical_median = median(canonical_samples, sample_count);
        printf("{\"schema_version\":1,\"width\":%u,\"height\":%u,\"repeats\":%u,\"warmup\":%u,"
               "\"canonical_ms\":%.3f,\"backend_ms\":%.3f,\"speedup\":%.6f,\"text_match\":true}\n",
               width, height, sample_count, WARMUP_ROUNDS, canonical_median, backend_median,
               canonical_median / backend_median);
    }
    result = 0;

cleanup:
    free(backend_samples);
    free(canonical_samples);
    free(canonical_text);
    free(backend_text);
    free(source);
    lw_recognizer_free(canonical_recognizer);
    lw_recognizer_free(backend_recognizer);
    return result;
}
