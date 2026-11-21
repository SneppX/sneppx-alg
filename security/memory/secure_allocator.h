#ifndef SNEPPX_SECURE_ALLOCATOR_H
#define SNEPPX_SECURE_ALLOCATOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
/*
 * SNEPPX - Secure Allocator
 *
 * WHAT
 *   Secure Allocator.
 *
 * CONCEPT
 *   Provides the Secure Allocator.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

typedef struct {
    void*    addr;
    size_t   size;
    size_t   guard_front;
    size_t   guard_back;
    uint64_t canary;
    int      is_freed;
} SNEPPXSecureAllocRecord;

typedef struct SNEPPXSecureAllocator {
    size_t   total_allocated;
    size_t   peak_allocated;
    size_t   num_allocations;
    int      use_guard_pages;
    int      use_canaries;
    void*    live_allocations;
    int      (*on_overflow)(const SNEPPXSecureAllocRecord* record);
} SNEPPXSecureAllocator;

typedef struct {
    size_t total_allocated;
    size_t peak_allocated;
    size_t num_allocations;
    size_t num_frees;
    size_t num_double_free_detected;
    size_t num_canary_violations;
} SNEPPXSecureAllocStats;

/**
 * @brief Initialize Secure Allocator.
 *
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_secure_allocator_init(SNEPPXSecureAllocator* alloc);
/**
 * @brief Destroy Secure Allocator.
 *
 * @param alloc [out] Alloc value.
 */
void SNEPPX_secure_allocator_destroy(SNEPPXSecureAllocator* alloc);

/**
 * @brief Perform Secure Alloc.
 *
 * @param alloc [out] Alloc value.
 * @param bytes [in] Bytes value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_alloc(SNEPPXSecureAllocator* alloc, size_t bytes, size_t alignment);
/**
 * @brief Free Secure.
 *
 * @param alloc [out] Alloc value.
 * @param ptr [out] Ptr value.
 */
void  SNEPPX_secure_free(SNEPPXSecureAllocator* alloc, void* ptr);
/**
 * @brief Perform Secure Audit.
 *
 * @param alloc [out] Alloc value.
 */
void  SNEPPX_secure_audit(SNEPPXSecureAllocator* alloc);

/**
 * @brief Perform Secure Canary Generate.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_secure_canary_generate(void);
/**
 * @brief Perform Secure Canary Check.
 *
 * @param ptr [out] Ptr value.
 * @param canary [in] Canary value.
 *
 * @return 0 on success, -1 on error.
 */
int      SNEPPX_secure_canary_check(void* ptr, uint64_t canary);

/**
 * @brief Perform Secure Freelist Check.
 *
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int                    SNEPPX_secure_freelist_check(SNEPPXSecureAllocator* alloc);
/**
 * @brief Perform Secure Free Quarantine.
 *
 * @param alloc [out] Alloc value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int                    SNEPPX_secure_free_quarantine(SNEPPXSecureAllocator* alloc, void* ptr);
/**
 * @brief Perform Secure Free Flush Quarantine.
 *
 * @param alloc [out] Alloc value.
 */
void                   SNEPPX_secure_free_flush_quarantine(SNEPPXSecureAllocator* alloc);
/**
 * @brief Perform Secure Allocator Get Stats.
 *
 * @param alloc [out] Alloc value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSecureAllocStats   SNEPPX_secure_allocator_get_stats(SNEPPXSecureAllocator* alloc);

#ifdef __cplusplus
}
#endif
#endif
