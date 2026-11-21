#include "memory_leak_detector.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define LEAK_MAX_ALLOCATIONS 65536
#define LEAK_BACKTRACE_DEPTH 16
#define LEAK_MAX_REPORTS 1024

/*
 * SNEPPX - Memory Leak Detector
 *
 * WHAT
 *   Memory Leak Detector.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    void *ptr;
    size_t size;
    const char *file;
    int line;
    const char *func;
    uint64_t timestamp;
    uint8_t freed;
    uint8_t backtrace[LEAK_BACKTRACE_DEPTH][sizeof(void *)];
    int bt_depth;
} leak_allocation_t;

static leak_allocation_t allocations[LEAK_MAX_ALLOCATIONS];
static int alloc_count = 0;
static uint64_t total_allocated = 0;
static uint64_t total_freed = 0;
static uint64_t peak_usage = 0;
static uint64_t current_usage = 0;
static int tracking_enabled = 1;
static int leak_threshold = 1024;

/**
 * @brief Initialize Leak.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_init(void) {
    memset(allocations, 0, sizeof(allocations));
    alloc_count = 0;
    total_allocated = 0;
    total_freed = 0;
    peak_usage = 0;
    current_usage = 0;
    tracking_enabled = 1;
    return 0;
}

/**
 * @brief Perform Leak Track Alloc.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_track_alloc(void *ptr, size_t size, const char *file, int line, const char *func) {
    if (!ptr || !tracking_enabled) return -1;
    if (alloc_count >= LEAK_MAX_ALLOCATIONS) return -1;
    allocations[alloc_count].ptr = ptr;
    allocations[alloc_count].size = size;
    allocations[alloc_count].file = file;
    allocations[alloc_count].line = line;
    allocations[alloc_count].func = func;
    allocations[alloc_count].timestamp = (uint64_t)clock();
    allocations[alloc_count].freed = 0;
    allocations[alloc_count].bt_depth = 0;
    alloc_count++;
    total_allocated += size;
    current_usage += size;
    if (current_usage > peak_usage) peak_usage = current_usage;
    return 0;
}

/**
 * @brief Free Leak Track.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_track_free(void *ptr) {
    if (!ptr || !tracking_enabled) return -1;
    for (int i = 0; i < alloc_count; i++) {
        if (allocations[i].ptr == ptr && !allocations[i].freed) {
            allocations[i].freed = 1;
            current_usage -= allocations[i].size;
            total_freed += allocations[i].size;
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Perform Leak Check.
 *
 * @param reports [out] Reports value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_check(leak_report_t *reports, int max_reports) {
    if (!reports) return -1;
    int found = 0;
    for (int i = 0; i < alloc_count && found < max_reports; i++) {
        if (!allocations[i].freed && allocations[i].size >= (size_t)leak_threshold) {
            reports[found].ptr = allocations[i].ptr;
            reports[found].size = allocations[i].size;
            reports[found].file = allocations[i].file;
            reports[found].line = allocations[i].line;
            reports[found].func = allocations[i].func;
            reports[found].timestamp = allocations[i].timestamp;
            found++;
        }
    }
    return found;
}

/**
 * @brief Perform Leak Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_get_stats(leak_stats_t *stats) {
    if (!stats) return -1;
    stats->total_allocations = alloc_count;
    stats->active_allocations = 0;
    for (int i = 0; i < alloc_count; i++)
        if (!allocations[i].freed) stats->active_allocations++;
    stats->total_allocated = total_allocated;
    stats->total_freed = total_freed;
    stats->current_usage = current_usage;
    stats->peak_usage = peak_usage;
    stats->leak_threshold = leak_threshold;
    stats->tracking_enabled = tracking_enabled;
    return 0;
}

/**
 * @brief Perform Leak Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_set_threshold(int bytes) {
    leak_threshold = bytes > 0 ? bytes : 1024;
    return 0;
}

/**
 * @brief Perform Leak Enable Tracking.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_enable_tracking(void) {
    tracking_enabled = 1;
    return 0;
}

/**
 * @brief Perform Leak Disable Tracking.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_disable_tracking(void) {
    tracking_enabled = 0;
    return 0;
}

/**
 * @brief Reset Leak.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_reset(void) {
    alloc_count = 0;
    total_allocated = 0;
    total_freed = 0;
    peak_usage = 0;
    current_usage = 0;
    memset(allocations, 0, sizeof(allocations));
    return 0;
}

/**
 * @brief Perform Leak Malloc.
 *
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_malloc(size_t size, const char *file, int line, const char *func) {
    void *ptr = malloc(size);
    if (ptr) SNEPPX_leak_track_alloc(ptr, size, file, line, func);
    return ptr;
}

/**
 * @brief Free Leak.
 */
void SNEPPX_leak_free(void *ptr) {
    if (ptr) {
        SNEPPX_leak_track_free(ptr);
        free(ptr);
    }
}

/**
 * @brief Perform Leak Calloc.
 *
 * @param nmemb [in] Nmemb value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_calloc(size_t nmemb, size_t size, const char *file, int line, const char *func) {
    void *ptr = calloc(nmemb, size);
    if (ptr) SNEPPX_leak_track_alloc(ptr, nmemb * size, file, line, func);
    return ptr;
}

/**
 * @brief Perform Leak Realloc.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_realloc(void *ptr, size_t size, const char *file, int line, const char *func) {
    if (ptr) SNEPPX_leak_track_free(ptr);
    void *new_ptr = realloc(ptr, size);
    if (new_ptr) SNEPPX_leak_track_alloc(new_ptr, size, file, line, func);
    return new_ptr;
}
