#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

/* Same-process full-OCR benchmark for a project-owned PPM image list. */
#include "lw_infer.h"
#include "cpu_features.h"
#include "ppm_image.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <psapi.h>
#  include <shellapi.h>
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <mach/mach_time.h>
#  include <sys/resource.h>
#else
#  include <sys/resource.h>
#  include <time.h>
#  include <unistd.h>
#endif

#define LW_DATASET_BENCHMARK_MAX_ITERATIONS 100u
#define LW_DATASET_BENCHMARK_MAX_IMAGES 100000u
#define LW_DATASET_BENCHMARK_MAX_PATH 32768u

typedef struct lw_process_memory {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
} lw_process_memory;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) ||
        frequency.QuadPart <= 0) {
        return 0.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#elif defined(__APPLE__)
    mach_timebase_info_data_t timebase;
    uint64_t ticks = mach_absolute_time();
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0u) {
        return 0.0;
    }
    return (double)ticks * (double)timebase.numer / (double)timebase.denom / 1000000000.0;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
#endif
}

static lw_process_memory process_memory(void) {
    lw_process_memory result;
    memset(&result, 0, sizeof(result));
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    memset(&counters, 0, sizeof(counters));
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        result.current_rss_bytes = (uint64_t)counters.WorkingSetSize;
        result.peak_rss_bytes = (uint64_t)counters.PeakWorkingSetSize;
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0) {
#  if defined(__APPLE__)
        result.peak_rss_bytes = (uint64_t)usage.ru_maxrss;
        {
            mach_task_basic_info_data_t task_info_data;
            mach_msg_type_number_t task_info_count = MACH_TASK_BASIC_INFO_COUNT;
            kern_return_t task_status =
                task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                          (task_info_t)&task_info_data, &task_info_count);
            if (task_status == KERN_SUCCESS) {
                result.current_rss_bytes = (uint64_t)task_info_data.resident_size;
            }
        }
#  else
        result.peak_rss_bytes = (uint64_t)usage.ru_maxrss * 1024u;
#  endif
    }
#  if defined(__linux__)
    {
        FILE* statm = fopen("/proc/self/statm", "r");
        unsigned long long total_pages = 0u;
        unsigned long long resident_pages = 0u;
        long page_size = sysconf(_SC_PAGESIZE);
        if (statm != NULL && page_size > 0 &&
            fscanf(statm, "%llu %llu", &total_pages, &resident_pages) == 2) {
            (void)total_pages;
            result.current_rss_bytes = (uint64_t)resident_pages * (uint64_t)page_size;
        }
        if (statm != NULL) {
            fclose(statm);
        }
    }
#  endif
#endif
    return result;
}

static void observe_current_rss(lw_process_memory sample, uint64_t* sample_count,
                                uint64_t* min_current_rss, uint64_t* max_current_rss) {
    if (sample_count == NULL || min_current_rss == NULL || max_current_rss == NULL ||
        sample.current_rss_bytes == 0u) {
        return;
    }
    if (*sample_count == 0u || sample.current_rss_bytes < *min_current_rss) {
        *min_current_rss = sample.current_rss_bytes;
    }
    if (*sample_count == 0u || sample.current_rss_bytes > *max_current_rss) {
        *max_current_rss = sample.current_rss_bytes;
    }
    if (*sample_count < UINT64_MAX) {
        ++*sample_count;
    }
}

static int parse_positive_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0u || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int compare_double(const void* left, const void* right) {
    const double a = *(const double*)left;
    const double b = *(const double*)right;
    return (a > b) - (a < b);
}

static double percentile(double* values, uint64_t count, uint32_t percentage) {
    uint64_t index;
    qsort(values, (size_t)count, sizeof(*values), compare_double);
    index = ((uint64_t)percentage * count + 99u) / 100u - 1u;
    return values[index];
}

static uint64_t output_checksum(const char* text, uint64_t byte_count) {
    uint64_t checksum = UINT64_C(14695981039346656037);
    uint64_t index;
    for (index = 0u; index < byte_count; ++index) {
        checksum ^= (uint8_t)text[index];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)SIZE_MAX / (uint64_t)element_size;
}

static char* trim_line(char* line) {
    char* start = line;
    char* end;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') {
        ++start;
    }
    end = start + strlen(start);
    while (end > start && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' ||
                           end[-1] == '\n')) {
        --end;
        *end = '\0';
    }
    return start;
}

static void free_image_paths(char** paths, uint32_t count) {
    uint32_t index;
    if (paths == NULL) {
        return;
    }
    for (index = 0u; index < count; ++index) {
        free(paths[index]);
    }
    free(paths);
}

static int read_image_list(const char* list_path, char*** paths_out, uint32_t* count_out) {
    FILE* stream = NULL;
    char line[LW_DATASET_BENCHMARK_MAX_PATH];
    char** paths = NULL;
    uint32_t count = 0u;
    uint32_t capacity = 0u;
    int ok = 0;
    stream = lw_example_open_read_utf8(list_path);
    if (stream == NULL) {
        fprintf(stderr, "unable to open image list: %s\n", list_path);
        goto cleanup;
    }
    while (fgets(line, sizeof(line), stream) != NULL) {
        char* value = trim_line(line);
        size_t length;
        char* copy;
        if (*value == '\0' || *value == '#') {
            continue;
        }
        length = strlen(value);
        if (length >= sizeof(line) || count == LW_DATASET_BENCHMARK_MAX_IMAGES) {
            fprintf(stderr, "image list is too large or contains an overlong path\n");
            goto cleanup;
        }
        if (count == capacity) {
            uint32_t next = capacity == 0u ? 16u : capacity * 2u;
            char** resized;
            if (next < capacity || next > LW_DATASET_BENCHMARK_MAX_IMAGES) {
                next = LW_DATASET_BENCHMARK_MAX_IMAGES;
            }
            resized = (char**)realloc(paths, (size_t)next * sizeof(*paths));
            if (resized == NULL) {
                fprintf(stderr, "image list allocation failed\n");
                goto cleanup;
            }
            paths = resized;
            capacity = next;
        }
        copy = (char*)malloc(length + 1u);
        if (copy == NULL) {
            fprintf(stderr, "image path allocation failed\n");
            goto cleanup;
        }
        memcpy(copy, value, length + 1u);
        paths[count++] = copy;
    }
    if (ferror(stream) || count == 0u) {
        fprintf(stderr, "image list is empty or could not be read\n");
        goto cleanup;
    }
    *paths_out = paths;
    *count_out = count;
    paths = NULL;
    ok = 1;
cleanup:
    if (stream != NULL) {
        fclose(stream);
    }
    free_image_paths(paths, count);
    return ok;
}

static int run_ocr(lw_ocr* ocr, const lw_example_ppm_image* image, lw_ocr_line* lines,
                   uint32_t line_capacity, char* text, uint64_t text_capacity,
                   lw_ocr_result* result) {
    lw_error error;
    lw_status status;
    lw_ocr_result_init(result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, image->pixels, image->byte_count, image->width, image->height,
                               image->width * 3u, lines, line_capacity, text, text_capacity, result,
                               &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR failed: %s: %s\n", lw_status_string(status), error.message);
        return 0;
    }
    return 1;
}

static int benchmark_main(int argc, char** argv) {
    char** image_paths = NULL;
    uint32_t image_count = 0u;
    uint32_t warmup_count = 1u;
    uint32_t iteration_count = 1u;
    uint32_t worker_count = 0u;
    uint32_t rec_target_width = 960u;
    lw_ocr* ocr = NULL;
    lw_ocr_info ocr_info;
    lw_ocr_line* lines = NULL;
    char* text = NULL;
    uint32_t* reference_lines = NULL;
    uint64_t* reference_bytes = NULL;
    uint64_t* reference_checksums = NULL;
    double* timings = NULL;
    uint64_t timing_count;
    uint64_t timing_index = 0u;
    uint64_t total_lines = 0u;
    uint64_t total_bytes = 0u;
    uint64_t aggregate_checksum = UINT64_C(14695981039346656037);
    double total_ms = 0.0;
    lw_process_memory memory_after_warmup;
    lw_process_memory memory_final;
    uint64_t observed_peak_rss;
    uint64_t observed_rss_samples = 0u;
    uint64_t observed_min_current_rss = 0u;
    uint64_t observed_max_current_rss = 0u;
    lw_error error;
    lw_status status;
    uint32_t image_index;
    uint32_t iteration;
    int exit_code = 1;

    memset(&memory_after_warmup, 0, sizeof(memory_after_warmup));
    memset(&memory_final, 0, sizeof(memory_final));
    lw_ocr_info_init(&ocr_info);
    if (argc < 6 || argc > 10 ||
        (argc >= 7 && !parse_positive_u32(argv[6], &warmup_count)) ||
        (argc >= 8 && !parse_positive_u32(argv[7], &iteration_count)) ||
        (argc >= 9 && !parse_positive_u32(argv[8], &worker_count)) ||
        (argc >= 10 && !parse_positive_u32(argv[9], &rec_target_width)) ||
        warmup_count > LW_DATASET_BENCHMARK_MAX_ITERATIONS ||
        iteration_count > LW_DATASET_BENCHMARK_MAX_ITERATIONS) {
        fprintf(stderr, "usage: full-ocr-dataset-benchmark <det.lwm> <cls.lwm> <rec.lwm> "
                        "<dictionary.txt> <image-list.txt> [warmup=1] [iterations=1] "
                        "[workers=platform-default] [rec-target-width=960]\n");
        return 2;
    }
    if (!read_image_list(argv[5], &image_paths, &image_count)) {
        goto cleanup;
    }
    lw_error_init(&error);
    {
        lw_ocr_options options;
        lw_ocr_options_init(&options);
        if (worker_count != 0u) {
            options.worker_count = worker_count;
        }
        options.recognizer.target_width = rec_target_width;
        status = lw_ocr_create(argv[1], argv[2], argv[3], argv[4], &options, &ocr, &error);
    }
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    if (lw_ocr_get_info(ocr, &ocr_info) != LW_STATUS_OK || ocr_info.max_line_capacity == 0u ||
        !allocation_fits(ocr_info.max_line_capacity, sizeof(*lines)) ||
        ocr_info.max_text_capacity == 0u || ocr_info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query OCR capacity\n");
        goto cleanup;
    }
    if (!allocation_fits(image_count, sizeof(*reference_lines)) ||
        !allocation_fits(image_count, sizeof(*reference_bytes)) ||
        !allocation_fits(image_count, sizeof(*reference_checksums))) {
        fprintf(stderr, "image list metadata is too large\n");
        goto cleanup;
    }
    timing_count = (uint64_t)image_count * iteration_count;
    if (!allocation_fits(timing_count, sizeof(*timings))) {
        fprintf(stderr, "timing array is too large\n");
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(ocr_info.max_line_capacity, sizeof(*lines));
    text = (char*)malloc((size_t)ocr_info.max_text_capacity);
    reference_lines = (uint32_t*)calloc(image_count, sizeof(*reference_lines));
    reference_bytes = (uint64_t*)calloc(image_count, sizeof(*reference_bytes));
    reference_checksums = (uint64_t*)calloc(image_count, sizeof(*reference_checksums));
    timings = (double*)malloc((size_t)timing_count * sizeof(*timings));
    if (lines == NULL || text == NULL || reference_lines == NULL || reference_bytes == NULL ||
        reference_checksums == NULL || timings == NULL) {
        fprintf(stderr, "dataset benchmark allocation failed\n");
        goto cleanup;
    }

    for (iteration = 0u; iteration < warmup_count; ++iteration) {
        for (image_index = 0u; image_index < image_count; ++image_index) {
            lw_example_ppm_image image;
            lw_ocr_result result;
            memset(&image, 0, sizeof(image));
            if (!lw_example_ppm_image_load_bgr(image_paths[image_index], &image) ||
                image.width > UINT32_MAX / 3u ||
                !run_ocr(ocr, &image, lines, ocr_info.max_line_capacity, text,
                         ocr_info.max_text_capacity, &result)) {
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            if (iteration == 0u) {
                reference_lines[image_index] = result.line_count;
                reference_bytes[image_index] = result.required_text_capacity;
                reference_checksums[image_index] =
                    output_checksum(text, result.required_text_capacity);
                total_lines += result.line_count;
                total_bytes += result.required_text_capacity;
            } else if (reference_lines[image_index] != result.line_count ||
                       reference_bytes[image_index] != result.required_text_capacity ||
                       reference_checksums[image_index] !=
                           output_checksum(text, result.required_text_capacity)) {
                fprintf(stderr, "warm-up OCR result is not deterministic for image %u\n",
                        image_index + 1u);
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            lw_example_ppm_image_free(&image);
            observe_current_rss(process_memory(), &observed_rss_samples,
                                &observed_min_current_rss, &observed_max_current_rss);
        }
    }
    memory_after_warmup = process_memory();
    for (iteration = 0u; iteration < iteration_count; ++iteration) {
        for (image_index = 0u; image_index < image_count; ++image_index) {
            lw_example_ppm_image image;
            lw_ocr_result result;
            double started;
            double finished;
            uint64_t checksum;
            memset(&image, 0, sizeof(image));
            if (!lw_example_ppm_image_load_bgr(image_paths[image_index], &image) ||
                image.width > UINT32_MAX / 3u) {
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            started = monotonic_seconds();
            if (started <= 0.0 ||
                !run_ocr(ocr, &image, lines, ocr_info.max_line_capacity, text,
                         ocr_info.max_text_capacity, &result)) {
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            finished = monotonic_seconds();
            checksum = output_checksum(text, result.required_text_capacity);
            if (reference_lines[image_index] != result.line_count ||
                reference_bytes[image_index] != result.required_text_capacity ||
                reference_checksums[image_index] != checksum) {
                fprintf(stderr, "timed OCR result is not deterministic for image %u\n",
                        image_index + 1u);
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            if (finished < started || timing_index >= timing_count) {
                fprintf(stderr, "invalid OCR timing for image %u\n", image_index + 1u);
                lw_example_ppm_image_free(&image);
                goto cleanup;
            }
            timings[timing_index++] = (finished - started) * 1000.0;
            total_ms += timings[timing_index - 1u];
            aggregate_checksum ^= checksum;
            aggregate_checksum *= UINT64_C(1099511628211);
            lw_example_ppm_image_free(&image);
            observe_current_rss(process_memory(), &observed_rss_samples,
                                &observed_min_current_rss, &observed_max_current_rss);
        }
    }
    memory_final = process_memory();
    observed_peak_rss = memory_final.peak_rss_bytes;
    if (observed_peak_rss < memory_after_warmup.current_rss_bytes) {
        observed_peak_rss = memory_after_warmup.current_rss_bytes;
    }
    if (observed_peak_rss < memory_final.current_rss_bytes) {
        observed_peak_rss = memory_final.current_rss_bytes;
    }
    printf("{\"schema_version\":1,\"backend\":\"%s\",",
           lw_simd_level_name(lw_detect_simd_level()));
    printf("\"resident_widths\":%s,\"images\":%u,\"lines\":%llu,\"workers\":%u,\"rec_target_width\":%u,",
#if defined(LW_REC_RESIDENT_WIDTHS) && LW_REC_RESIDENT_WIDTHS
           "true",
#else
           "false",
#endif
           image_count, (unsigned long long)total_lines, ocr_info.worker_count, rec_target_width);
    printf("\"warmup\":%u,\"iterations\":%u,\"ocr_ms\":{\"mean\":%.6f,\"p95\":%.6f,\"total\":%.6f},",
           warmup_count, iteration_count, total_ms / (double)timing_count,
           percentile(timings, timing_count, 95u), total_ms);
    printf("\"output_checksum\":\"%016llx\",\"rss_after_warmup_bytes\":%llu,\"rss_final_bytes\":%llu,\"peak_rss_bytes\":%llu,\"rss_sample_count\":%llu,\"min_sampled_rss_bytes\":%llu,\"max_sampled_rss_bytes\":%llu}\n",
           (unsigned long long)aggregate_checksum,
           (unsigned long long)memory_after_warmup.current_rss_bytes,
           (unsigned long long)memory_final.current_rss_bytes,
           (unsigned long long)observed_peak_rss,
           (unsigned long long)observed_rss_samples,
           (unsigned long long)observed_min_current_rss,
           (unsigned long long)observed_max_current_rss);
    exit_code = 0;

cleanup:
    free(timings);
    free(reference_checksums);
    free(reference_bytes);
    free(reference_lines);
    free(text);
    free(lines);
    lw_ocr_free(ocr);
    free_image_paths(image_paths, image_count);
    return exit_code;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
    SetConsoleOutputCP(CP_UTF8);
    wide_argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wide_argv == NULL || argc <= 0) {
        return 2;
    }
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1, NULL,
                                        0, NULL, NULL);
        if (bytes <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], bytes,
                                utf8_argv[index], bytes, NULL, NULL) <= 0) {
            goto cleanup;
        }
    }
    result = benchmark_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index) {
        free(utf8_argv[index]);
    }
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return benchmark_main(argc, argv);
}
#endif
