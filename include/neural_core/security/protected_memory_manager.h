#ifndef SNEPPX_SECURE_MEM_H
#define SNEPPX_SECURE_MEM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Protected Memory Manager
 *
 * WHAT
 *   Protected Memory Manager.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct SNEPPXSecurePool SNEPPXSecurePool;

typedef struct {
    int guard_pages;
    int canaries;
    int lock_memory;
    int randomize_layout;
} SNEPPXSecureAllocConfig;

/**
 * @brief Create Secure Pool.
 *
 * @param size [in] Size value.
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSecurePool* SNEPPX_secure_pool_create(size_t size, const SNEPPXSecureAllocConfig* config);
/**
 * @brief Destroy Secure Pool.
 *
 * @param pool [out] Pool value.
 */
void SNEPPX_secure_pool_destroy(SNEPPXSecurePool* pool);
/**
 * @brief Perform Secure Malloc.
 *
 * @param pool [out] Pool value.
 * @param size [in] Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_malloc(SNEPPXSecurePool* pool, size_t size, size_t alignment);
/**
 * @brief Free Secure Pool.
 *
 * @param pool [out] Pool value.
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 */
void SNEPPX_secure_pool_free(SNEPPXSecurePool* pool, void* ptr, size_t size);
/**
 * @brief Perform Secure Realloc.
 *
 * @param pool [out] Pool value.
 * @param ptr [out] Ptr value.
 * @param old_size [in] Old Size value.
 * @param new_size [in] New Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_realloc(SNEPPXSecurePool* pool, void* ptr, size_t old_size, size_t new_size, size_t alignment);
/**
 * @brief Perform Secure Pool Stats.
 *
 * @param pool [out] Pool value.
 * @param total [out] Total value.
 * @param used [out] Used value.
 * @param peak [out] Peak value.
 */
void SNEPPX_secure_pool_stats(SNEPPXSecurePool* pool, size_t* total, size_t* used, size_t* peak);
/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 */
void SNEPPX_secure_zero(void* ptr, size_t len);


#ifdef __cplusplus
}
#endif
#endif
