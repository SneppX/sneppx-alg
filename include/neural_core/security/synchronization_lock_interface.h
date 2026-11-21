#ifndef SNEPPX_LOCK_H
#define SNEPPX_LOCK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * SNEPPX - Synchronization Lock Interface
 *
 * WHAT
 *   Synchronization Lock Interface.
 *
 * CONCEPT
 *   Provides the Synchronization Lock Interface.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct SNEPPXLock SNEPPXLock;

/**
 * @brief Initialize Lock.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLock* SNEPPX_lock_init(void);
/**
 * @brief Destroy Lock.
 *
 * @param lock [out] Lock value.
 */
void SNEPPX_lock_destroy(SNEPPXLock* lock);
/**
 * @brief Perform Lock Acquire.
 *
 * @param lock [out] Lock value.
 */
void SNEPPX_lock_acquire(SNEPPXLock* lock);
/**
 * @brief Perform Lock Release.
 *
 * @param lock [out] Lock value.
 */
void SNEPPX_lock_release(SNEPPXLock* lock);
/**
 * @brief Perform Lock Try Acquire.
 *
 * @param lock [out] Lock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lock_try_acquire(SNEPPXLock* lock);
/**
 * @brief Perform Lock Is Held.
 *
 * @param lock [in] Lock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lock_is_held(const SNEPPXLock* lock);

/**
 * @brief Perform Mlock.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mlock(void* ptr, size_t len);
/**
 * @brief Perform Munlock.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_munlock(void* ptr, size_t len);
/**
 * @brief Perform Mlockall Possible.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mlockall_possible(void);


#ifdef __cplusplus
}
#endif
#endif
