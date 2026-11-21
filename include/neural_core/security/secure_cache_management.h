#ifndef SNEPPX_CACHE_H
#define SNEPPX_CACHE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * SNEPPX - Secure Cache Management
 *
 * WHAT
 *   Secure Cache Management.
 *
 * CONCEPT
 *   Provides cache timing protection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Cache Flush.
 *
 * @param ptr [in] Ptr value.
 * @param len [in] Len value.
 */
void SNEPPX_cache_flush(const void* ptr, size_t len);
/**
 * @brief Perform Cache Prefetch.
 *
 * @param ptr [in] Ptr value.
 */
void SNEPPX_cache_prefetch(const void* ptr);
/**
 * @brief Perform Cache Barrier.
 */
void SNEPPX_cache_barrier(void);


#ifdef __cplusplus
}
#endif
#endif
