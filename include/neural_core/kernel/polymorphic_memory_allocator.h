#ifndef SNEPPX_MEMORY_H
#define SNEPPX_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Polymorphic Memory Allocator
 *
 * WHAT
 *   Polymorphic Memory Allocator.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* ============================================================
 * Core Memory Functions
 * ============================================================ */
/**
 * @brief Perform Malloc.
 *
 * @param size [in] Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_malloc(size_t size, size_t alignment);
/**
 * @brief Free Free.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 */
void  SNEPPX_free(void* ptr, size_t size);
/**
 * @brief Perform Realloc.
 *
 * @param ptr [out] Ptr value.
 * @param old_size [in] Old Size value.
 * @param new_size [in] New Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment);
/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 */
void  SNEPPX_secure_zero(void* ptr, size_t size);
/**
 * @brief Perform Secure Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param size [in] Size value.
 */
void  SNEPPX_secure_copy(void* dst, const void* src, size_t size);

/* ============================================================
 * Pool Allocator — 18 size classes (16–8192 B)
 * Lock-free per-class stacks + thread-local caches
 * ============================================================ */
#define SNEPPX_NUM_SIZE_CLASSES  18
#define SNEPPX_POOL_MAX_SIZE     8192UL
#define SNEPPX_TLS_CACHE_MAX     32
#define SNEPPX_CHUNK_SIZE        (1024UL * 1024UL)   /* 1 MB */

/* ---------- Lock-free node & stack (Treiber) ---------- */
typedef struct SNEPPXMemNode {
    struct SNEPPXMemNode* next;   /* accessed atomically */
} SNEPPXMemNode;

typedef struct {
    SNEPPXMemNode* head;          /* accessed atomically via platform intrinsics */
} SNEPPXLockFreeStack;

/* ---------- Per-size-class pool ---------- */
typedef struct {
    size_t            block_size;        /* bytes per block          */
    size_t            blocks_per_chunk;  /* blocks carved per chunk */
    SNEPPXLockFreeStack stack;             /* free-list               */
    size_t            alloc_count;       /* lifetime allocs         */
    size_t            free_count;        /* lifetime frees          */
    size_t            chunk_count;       /* chunks allocated        */
} SNEPPXMemPool;

/* ---------- Thread-local cache entry ---------- */
typedef struct {
    void*  free_list;    /* LIFO list of blocks */
    size_t count;        /* entries in list     */
    size_t max;          /* capacity            */
} SNEPPXTlsEntry;

/* ---------- Thread-local cache ---------- */
typedef struct {
    SNEPPXTlsEntry entries[SNEPPX_NUM_SIZE_CLASSES];
    size_t       hits;       /* TLS cache hits          */
    size_t       capacity;   /* max blocks per class    */
} SNEPPXTlsCache;

/* ---------- Global pool statistics ---------- */
typedef struct {
    size_t total_pool_allocated;  /* bytes allocated lifetime       */
    size_t total_pool_freed;      /* bytes freed lifetime           */
    size_t total_chunks;          /* mmap/VirtualAlloc calls        */
    size_t total_pool_hits;       /* allocations served from pool   */
    size_t total_tls_hits;        /* allocations served from TLS    */
    size_t active_tls_caches;     /* number of active thread caches */
} SNEPPXMemStats;

/* ---------- API ---------- */

/* Initialise the global pool (idempotent, thread-safe).  Returns 0 on success. */
/**
 * @brief Initialize Mem Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pool_init(void);

/* Destroy the global pool and release all chunks.  Not thread-safe. */
/**
 * @brief Destroy Mem Pool.
 */
void SNEPPX_mem_pool_destroy(void);

/* Initialise / destroy the calling thread's TLS cache. */
/**
 * @brief Initialize Tls Cache.
 */
void SNEPPX_tls_cache_init(void);
/**
 * @brief Destroy Tls Cache.
 */
void SNEPPX_tls_cache_destroy(void);

/* Allocate a block of at least `size` bytes from the pool.
 * Returns 16-byte aligned zeroed memory, or NULL on failure.
 * Blocks larger than SNEPPX_POOL_MAX_SIZE fall back to SNEPPX_malloc. */
/**
 * @brief Perform Pool Alloc.
 *
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pool_alloc(size_t size);

/* Return a block obtained from SNEPPX_pool_alloc back to the pool.
 * Safe to call with NULL.  Blocks larger than SNEPPX_POOL_MAX_SIZE
 * are forwarded to SNEPPX_free. */
/**
 * @brief Free Pool.
 *
 * @param ptr [out] Ptr value.
 * @param size [in] Size value.
 */
void SNEPPX_pool_free(void* ptr, size_t size);

/* Copy current pool statistics into *stats. */
/**
 * @brief Perform Mem Pool Stats.
 *
 * @param stats [out] Stats value.
 */
void SNEPPX_mem_pool_stats(SNEPPXMemStats* stats);

/* Print pool statistics to stdout. */
/**
 * @brief Perform Mem Pool Print Stats.
 */
void SNEPPX_mem_pool_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_MEMORY_H */
