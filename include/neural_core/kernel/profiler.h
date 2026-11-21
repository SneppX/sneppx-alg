#ifndef SNEPPX_PROFILER_H
#define SNEPPX_PROFILER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Profiler
 *
 * WHAT
 *   Profiler.
 *
 * CONCEPT
 *   Provides performance profiling.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Maximum number of named profiler entries */
#define SNEPPX_PROFILER_MAX_ENTRIES 256
#define SNEPPX_PROFILER_NAME_MAX 64

/* Profiler entry: aggregates timing for a named operation */
typedef struct {
    char name[SNEPPX_PROFILER_NAME_MAX];
    int num_calls;
    float total_time_ms;
    float min_time_ms;
    float max_time_ms;
    float avg_time_ms;
} SNEPPX_ProfilerEntry;

/* Global profiler state */
typedef struct {
    SNEPPX_ProfilerEntry entries[SNEPPX_PROFILER_MAX_ENTRIES];
    int num_entries;
    int enabled;
} SNEPPX_Profiler;

/* Initialize/finalize */
/**
 * @brief Initialize Profiler.
 *
 * @param prof [out] Prof value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_profiler_init(SNEPPX_Profiler* prof);
/**
 * @brief Destroy Profiler.
 *
 * @param prof [out] Prof value.
 */
void SNEPPX_profiler_destroy(SNEPPX_Profiler* prof);
/**
 * @brief Perform Profiler Enable.
 *
 * @param prof [out] Prof value.
 * @param enabled [in] Enabled value.
 */
void SNEPPX_profiler_enable(SNEPPX_Profiler* prof, int enabled);

/* Record a timing sample for a named operation */
/**
 * @brief Perform Profiler Record.
 *
 * @param prof [out] Prof value.
 * @param name [in] Name value.
 * @param elapsed_ms [in] Elapsed Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_profiler_record(SNEPPX_Profiler* prof, const char* name, float elapsed_ms);

/* Get entry by name */
/**
 * @brief Get Profiler.
 *
 * @param prof [out] Prof value.
 * @param name [in] Name value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_ProfilerEntry* SNEPPX_profiler_get(SNEPPX_Profiler* prof, const char* name);

/* Reset all entries */
/**
 * @brief Reset Profiler.
 *
 * @param prof [out] Prof value.
 */
void SNEPPX_profiler_reset(SNEPPX_Profiler* prof);

/* Print summary to stdout */
/**
 * @brief Perform Profiler Print.
 *
 * @param prof [in] Prof value.
 */
void SNEPPX_profiler_print(const SNEPPX_Profiler* prof);

/* Export to JSON string (caller must free) */
/**
 * @brief Perform Profiler To Json.
 *
 * @param prof [in] Prof value.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_profiler_to_json(const SNEPPX_Profiler* prof);

/* CUDA kernel duration helper: wraps a kernel launch with timing */
#ifdef SNEPPX_HAS_CUDA
typedef struct {
    cudaEvent_t start;
    cudaEvent_t end;
} SNEPPX_KernelTimer;

/**
 * @brief Initialize Kernel Timer.
 *
 * @param kt [out] Kt value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_kernel_timer_init(SNEPPX_KernelTimer* kt);
/**
 * @brief Start Kernel Timer.
 *
 * @param kt [out] Kt value.
 * @param stream [in] Stream value.
 */
void SNEPPX_kernel_timer_start(SNEPPX_KernelTimer* kt, cudaStream_t stream);
/**
 * @brief Stop Kernel Timer.
 *
 * @param kt [out] Kt value.
 * @param stream [in] Stream value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_kernel_timer_stop(SNEPPX_KernelTimer* kt, cudaStream_t stream);
/**
 * @brief Destroy Kernel Timer.
 *
 * @param kt [out] Kt value.
 */
void SNEPPX_kernel_timer_destroy(SNEPPX_KernelTimer* kt);

/* Convenience macro: profiles a kernel launch */
#define SNEPPX_PROFILE_KERNEL(prof, name, stream, kernel_call) \
    do { \
        cudaEvent_t _p_start, _p_end; \
        cudaEventCreate(&_p_start); \
        cudaEventCreate(&_p_end); \
        cudaEventRecord(_p_start, stream); \
        kernel_call; \
        cudaEventRecord(_p_end, stream); \
        cudaEventSynchronize(_p_end); \
        float _p_ms = 0; \
        cudaEventElapsedTime(&_p_ms, _p_start, _p_end); \
        SNEPPX_profiler_record(prof, name, _p_ms); \
        cudaEventDestroy(_p_start); \
        cudaEventDestroy(_p_end); \
    } while(0)
#else
typedef struct { int _unused; } SNEPPX_KernelTimer;
#define SNEPPX_kernel_timer_init(kt) (-1)
#define SNEPPX_kernel_timer_start(kt, stream) ((void)0)
#define SNEPPX_kernel_timer_stop(kt, stream) (0.0f)
#define SNEPPX_kernel_timer_destroy(kt) ((void)0)
#define SNEPPX_PROFILE_KERNEL(prof, name, stream, kernel_call) do { kernel_call; } while(0)
#endif

/* Stack-based NVTX-style range markers (no NVTX dependency) */
/**
 * @brief Perform Range Push.
 *
 * @param name [in] Name value.
 */
void SNEPPX_range_push(const char* name);
/**
 * @brief Perform Range Pop.
 */
void SNEPPX_range_pop(void);
/**
 * @brief Perform Range Get Depth.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_range_get_depth(void);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_PROFILER_H */
