#ifndef SNEPPX_THREAD_H
#define SNEPPX_THREAD_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Concurrent Workload Dispatch
 *
 * WHAT
 *   Concurrent Workload Dispatch.
 *
 * CONCEPT
 *   Provides the Concurrent Workload Dispatch.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* ============================================================
 * Task
 * ============================================================ */

typedef struct {
    void (*func)(void* arg);
    void* arg;
} SNEPPXTask;

/* ============================================================
 * Future  –  synchronise on a single task
 * ============================================================ */

typedef struct SNEPPXFuture SNEPPXFuture;

/**
 * @brief Create Future.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFuture* SNEPPX_future_create(void);
/**
 * @brief Destroy Future.
 *
 * @param fut [out] Fut value.
 */
void        SNEPPX_future_destroy(SNEPPXFuture* fut);
/**
 * @brief Perform Future Wait.
 *
 * @param fut [out] Fut value.
 */
void        SNEPPX_future_wait(SNEPPXFuture* fut);
/**
 * @brief Perform Future Is Ready.
 *
 * @param fut [out] Fut value.
 *
 * @return 0 on success, -1 on error.
 */
int         SNEPPX_future_is_ready(SNEPPXFuture* fut);
/**
 * @brief Perform Future Set Result.
 *
 * @param fut [out] Fut value.
 * @param result [out] Result value.
 */
void        SNEPPX_future_set_result(SNEPPXFuture* fut, void* result);
/**
 * @brief Perform Future Get Result.
 *
 * @param fut [out] Fut value.
 *
 * @return Pointer on success, NULL on error.
 */
void*       SNEPPX_future_get_result(SNEPPXFuture* fut);

/* ============================================================
 * Thread Pool  –  work-stealing, N workers
 * ============================================================ */

typedef struct SNEPPXThreadPool SNEPPXThreadPool;

/**
 * @brief Create Threadpool.
 *
 * @param num_threads [in] Num Threads value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXThreadPool* SNEPPX_threadpool_create(size_t num_threads);
/**
 * @brief Destroy Threadpool.
 *
 * @param pool [out] Pool value.
 */
void            SNEPPX_threadpool_destroy(SNEPPXThreadPool* pool);

/* Submit a task (copied).  Returns 0 on success. */
/**
 * @brief Perform Threadpool Submit.
 *
 * @param pool [out] Pool value.
 * @param task [in] Task value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threadpool_submit(SNEPPXThreadPool* pool, SNEPPXTask task);

/* Submit a task with a future to track completion. */
/**
 * @brief Perform Threadpool Submit Future.
 *
 * @param pool [out] Pool value.
 * @param task [in] Task value.
 * @param fut [out] Fut value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threadpool_submit_future(SNEPPXThreadPool* pool, SNEPPXTask task, SNEPPXFuture* fut);

/* Wait until all submitted tasks complete. */
/**
 * @brief Perform Threadpool Wait.
 *
 * @param pool [out] Pool value.
 */
void SNEPPX_threadpool_wait(SNEPPXThreadPool* pool);

/* Default number of threads (hardware concurrency, min 2). */
/**
 * @brief Perform Threadpool Default Count.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_threadpool_default_count(void);

/* ============================================================
 * Parallel For / Reduce
 * ============================================================ */

/* Callback for parallel_for – processes chunk [start, end). */
typedef void (*SNEPPXRangeFunc)(size_t start, size_t end, void* arg);

/**
 * @brief Perform Parallel For.
 *
 * @param pool [out] Pool value.
 * @param start [in] Start value.
 * @param end [in] End value.
 * @param func [in] Func value.
 * @param arg [out] Arg value.
 */
void SNEPPX_parallel_for(SNEPPXThreadPool* pool,
                       size_t start, size_t end,
                       SNEPPXRangeFunc func, void* arg);

/* Reduce callback – processes chunk [start, end) into result. */
typedef void (*SNEPPXReduceFunc)(size_t start, size_t end,
                               void* arg, void* result);

/* Combine callback – merges src into dst. */
typedef void (*SNEPPXCombineFunc)(void* dst, const void* src);

/**
 * @brief Perform Parallel Reduce.
 *
 * @param pool [out] Pool value.
 * @param start [in] Start value.
 * @param end [in] End value.
 * @param init [out] Init value.
 * @param elem_size [in] Elem Size value.
 * @param reduce_func [in] Reduce Func value.
 * @param combine_func [in] Combine Func value.
 * @param result [out] Result value.
 * @param user_arg [out] User Arg value.
 */
void SNEPPX_parallel_reduce(SNEPPXThreadPool* pool,
                          size_t start, size_t end,
                          void* init, size_t elem_size,
                          SNEPPXReduceFunc reduce_func,
                          SNEPPXCombineFunc combine_func,
                          void* result,
                          void* user_arg);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_THREAD_H */
