#ifndef SNEPPX_MEMORY_LEAK_DETECTOR_H
#define SNEPPX_MEMORY_LEAK_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

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
} leak_report_t;

typedef struct {
    int total_allocations;
    int active_allocations;
    uint64_t total_allocated;
    uint64_t total_freed;
    uint64_t current_usage;
    uint64_t peak_usage;
    int leak_threshold;
    int tracking_enabled;
} leak_stats_t;

/**
 * @brief Initialize Leak.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_init(void);
/**
 * @brief Perform Leak Track Alloc.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 * @param func [in] Func value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_track_alloc(void *ptr, size_t size, const char *file, int line, const char *func);
/**
 * @brief Free Leak Track.
 *
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_track_free(void *ptr);
/**
 * @brief Perform Leak Check.
 *
 * @param reports [out] Reports value.
 * @param max_reports [in] Max Reports value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_check(leak_report_t *reports, int max_reports);
/**
 * @brief Perform Leak Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_get_stats(leak_stats_t *stats);
/**
 * @brief Perform Leak Set Threshold.
 *
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_set_threshold(int bytes);
/**
 * @brief Perform Leak Enable Tracking.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_enable_tracking(void);
/**
 * @brief Perform Leak Disable Tracking.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_disable_tracking(void);
/**
 * @brief Reset Leak.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_leak_reset(void);

/**
 * @brief Perform Leak Malloc.
 *
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 * @param func [in] Func value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_malloc(size_t size, const char *file, int line, const char *func);
/**
 * @brief Free Leak.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_leak_free(void *ptr);
/**
 * @brief Perform Leak Calloc.
 *
 * @param nmemb [in] Nmemb value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 * @param func [in] Func value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_calloc(size_t nmemb, size_t size, const char *file, int line, const char *func);
/**
 * @brief Perform Leak Realloc.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 * @param file [in] File value.
 * @param line [in] Line value.
 * @param func [in] Func value.
 *
 * @return Pointer on success, NULL on error.
 */
void *SNEPPX_leak_realloc(void *ptr, size_t size, const char *file, int line, const char *func);


#ifdef __cplusplus
}
#endif
#endif
