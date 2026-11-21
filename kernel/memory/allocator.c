#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#include <intrin.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#endif

/*
 * SNEPPX - Allocator
 *
 * WHAT
 *   Allocator.
 *
 * CONCEPT
 *   Provides the Allocator.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ==================================================================
 *  Platform-abstracted atomic operations  (macros for MSVC compat)
 * ================================================================== */

#ifdef _WIN32
#define SNEPPX_CAS_PTR(dst, old, new_val) \
    (InterlockedCompareExchangePointer((void* volatile*)(dst), (void*)(new_val), (void*)(old)) == (void*)(old))
#define SNEPPX_ATOMIC_LOAD_PTR(src) \
    InterlockedCompareExchangePointer((void* volatile*)(src), NULL, NULL)
#define SNEPPX_ATOMIC_XCHG_INT(dst, val) \
    _InterlockedExchange((volatile long*)(dst), (long)(val))
#define SNEPPX_ATOMIC_ADD_INT(dst, val) \
    _InterlockedExchangeAdd((volatile long*)(dst), (long)(val))
#if defined(_M_X64)
#define SNEPPX_ATOMIC_ADD_SIZE(dst, val) \
    ((size_t)_InterlockedExchangeAdd64((volatile __int64*)(dst), (__int64)(val)))
#else
#define SNEPPX_ATOMIC_ADD_SIZE(dst, val) \
    ((size_t)_InterlockedExchangeAdd((volatile long*)(dst), (long)(val)))
#endif
#else /* GCC / Clang */
#define SNEPPX_CAS_PTR(dst, old, new_val) \
    __sync_bool_compare_and_swap((void* volatile*)(dst), (void*)(old), (void*)(new_val))
#define SNEPPX_ATOMIC_LOAD_PTR(src) \
    __sync_fetch_and_add((void* volatile*)(src), 0)
#define SNEPPX_ATOMIC_XCHG_INT(dst, val) \
    __sync_lock_test_and_set((volatile int*)(dst), (int)(val))
#define SNEPPX_ATOMIC_ADD_INT(dst, val) \
    __sync_fetch_and_add((volatile int*)(dst), (int)(val))
#define SNEPPX_ATOMIC_ADD_SIZE(dst, val) \
    __sync_fetch_and_add((volatile size_t*)(dst), (size_t)(val))
#endif

/* ==================================================================
 *  Core Memory Functions (unchanged)
 * ================================================================== */

/**
 * @brief Perform Malloc.
 *
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#elif defined(__linux__) || defined(__APPLE__)
    size_t align = alignment;
    if (align < sizeof(void*)) align = sizeof(void*);
    if ((align & (align - 1)) != 0) {
        size_t p = 1;
        while (p < align) p <<= 1;
        align = p;
    }
    if (posix_memalign(&ptr, align, size) != 0) {
        return NULL;
    }
#else
    ptr = malloc(size);
#endif
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

/**
 * @brief Free Free.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_free(void* ptr, size_t size) {
    if (!ptr) return;
    SNEPPX_secure_zero(ptr, size);
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Perform Realloc.
 *
 * @param ptr [out] Ptr value.
 * @param old_size [in] Old Size value.
 * @param new_size [in] New Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    void* new_ptr = SNEPPX_malloc(new_size, alignment);
    if (!new_ptr) return NULL;
    size_t copy_size = old_size < new_size ? old_size : new_size;
    SNEPPX_secure_copy(new_ptr, ptr, copy_size);
    SNEPPX_free(ptr, old_size);
    return new_ptr;
}

/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_secure_zero(void* ptr, size_t size) {
    if (!ptr) return;
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    for (size_t i = 0; i < size; i++) {
        p[i] = 0;
    }
}

/**
 * @brief Perform Secure Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 */
void SNEPPX_secure_copy(void* dst, const void* src, size_t size) {
    if (!dst || !src) return;
    memcpy(dst, src, size);
}

/* ==================================================================
 *  Pool Allocator
 * ==================================================================
 *
 *  Architecture
 *  ────────────
 *  18 size classes  (16, 32, 48, 64, 96, 128, 192, 256, 384, 512,
 *                    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
 *
 *  Each pool has a lock-free Treiber stack of free blocks.  Blocks
 *  are carved from 1 MiB chunks obtained via VirtualAlloc / mmap.
 *  Every thread keeps a small TLS cache (max 32 entries per class)
 *  to avoid global contention on the fast path.
 */

/* ------------------------------------------------------------------
 *  Size class table
 * ------------------------------------------------------------------ */

static const size_t g_size_classes[SNEPPX_NUM_SIZE_CLASSES] = {
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512,
    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192
};

/* Given a size, return the pool index (0-based) or -1. */
/**
 * @brief Perform Pool Size Class.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_pool_size_class(size_t size) {
    if (size == 0 || size > SNEPPX_POOL_MAX_SIZE) return -1;
    if (size <= 16)   return 0;
    if (size <= 32)   return 1;
    if (size <= 48)   return 2;
    if (size <= 64)   return 3;
    if (size <= 96)   return 4;
    if (size <= 128)  return 5;
    if (size <= 192)  return 6;
    if (size <= 256)  return 7;
    if (size <= 384)  return 8;
    if (size <= 512)  return 9;
    if (size <= 768)  return 10;
    if (size <= 1024) return 11;
    if (size <= 1536) return 12;
    if (size <= 2048) return 13;
    if (size <= 3072) return 14;
    if (size <= 4096) return 15;
    if (size <= 6144) return 16;
    if (size <= 8192) return 17;
    return -1;
}

/* ------------------------------------------------------------------
 *  Chunk tracking (singly-linked list)
 * ------------------------------------------------------------------ */

typedef struct ChunkHeader {
    struct ChunkHeader* next;
    void*               base;
    size_t              size;
} ChunkHeader;

static ChunkHeader* g_chunk_list = NULL;

/* ------------------------------------------------------------------
 *  Platform abstraction  ─  chunk allocation
 * ------------------------------------------------------------------ */

static void* os_alloc(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
#endif
}

static void os_free(void* ptr, size_t size) {
    if (!ptr) return;
    (void)size;
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);
#endif
}

/* ------------------------------------------------------------------
 *  Lock-free stack operations (Treiber)
 * ------------------------------------------------------------------ */

static inline void lf_push(SNEPPXLockFreeStack* stack, SNEPPXMemNode* node) {
    SNEPPXMemNode* old;
    do {
        old = (SNEPPXMemNode*)SNEPPX_ATOMIC_LOAD_PTR(&stack->head);
        node->next = old;
    } while (!SNEPPX_CAS_PTR(&stack->head, old, node));
}

static inline SNEPPXMemNode* lf_pop(SNEPPXLockFreeStack* stack) {
    SNEPPXMemNode* old = (SNEPPXMemNode*)SNEPPX_ATOMIC_LOAD_PTR(&stack->head);
    while (old) {
        SNEPPXMemNode* next = old->next;
        if (SNEPPX_CAS_PTR(&stack->head, old, next))
            return old;
        old = (SNEPPXMemNode*)SNEPPX_ATOMIC_LOAD_PTR(&stack->head);
    }
    return NULL;
}

/* ------------------------------------------------------------------
 *  Global pool state
 * ------------------------------------------------------------------ */

static SNEPPXMemPool  g_pools[SNEPPX_NUM_SIZE_CLASSES];
static volatile int g_pool_initialized = 0;
static volatile size_t g_total_pool_allocated = 0;
static volatile size_t g_total_pool_freed     = 0;

/* TLS cache  (one per thread, lazily init'd) */
#ifdef _MSC_VER
static __declspec(thread) SNEPPXTlsCache* g_tls_cache = NULL;
#else
static __thread SNEPPXTlsCache* g_tls_cache = NULL;
#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



static volatile int g_active_tls_caches = 0;

/* ------------------------------------------------------------------
 *  SNEPPX_mem_pool_init  –  one-shot initialisation
 * ------------------------------------------------------------------ */

/**
 * @brief Initialize Mem Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_pool_init(void) {
    /* Idempotent  –  only the first call does any work. */
    if (SNEPPX_ATOMIC_XCHG_INT(&g_pool_initialized, 1) != 0)
        return 0;

    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        g_pools[i].block_size       = g_size_classes[i];
        g_pools[i].blocks_per_chunk = SNEPPX_CHUNK_SIZE / g_size_classes[i];
        g_pools[i].stack.head       = NULL;
        g_pools[i].alloc_count      = 0;
        g_pools[i].free_count       = 0;
        g_pools[i].chunk_count      = 0;
    }
    g_chunk_list = NULL;
    return 0;
}

/* ------------------------------------------------------------------
 *  Allocate one chunk and push its blocks onto the pool stack
 * ------------------------------------------------------------------ */

static int pool_grow(SNEPPXMemPool* pool) {
    size_t   block_size = pool->block_size;
    size_t   nblocks    = pool->blocks_per_chunk;
    size_t   chunk_size = block_size * nblocks;
    uint8_t* base       = (uint8_t*)os_alloc(chunk_size);

    if (!base) return -1;

    /* Record chunk for cleanup */
    ChunkHeader* hdr = (ChunkHeader*)os_alloc(sizeof(ChunkHeader));
    if (!hdr) { os_free(base, chunk_size); return -1; }
    hdr->base = base;
    hdr->size = chunk_size;
    hdr->next = g_chunk_list;
    g_chunk_list = hdr;

    /* Carve into blocks and push onto the lock-free stack */
    for (size_t b = 0; b < nblocks; b++) {
        SNEPPXMemNode* node = (SNEPPXMemNode*)(base + b * block_size);
        lf_push(&pool->stack, node);
    }
    pool->chunk_count++;
    return 0;
}

/* ------------------------------------------------------------------
 *  TLS cache helpers
 * ------------------------------------------------------------------ */

/**
 * @brief Initialize Tls Cache.
 */
void SNEPPX_tls_cache_init(void) {
    if (g_tls_cache) return;
    g_tls_cache = (SNEPPXTlsCache*)malloc(sizeof(SNEPPXTlsCache));
    if (!g_tls_cache) return;
    memset(g_tls_cache, 0, sizeof(SNEPPXTlsCache));
    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        g_tls_cache->entries[i].max = SNEPPX_TLS_CACHE_MAX;
    }
    g_tls_cache->capacity = SNEPPX_TLS_CACHE_MAX;
    SNEPPX_ATOMIC_ADD_INT(&g_active_tls_caches, 1);
}

/**
 * @brief Destroy Tls Cache.
 */
void SNEPPX_tls_cache_destroy(void) {
    if (!g_tls_cache) return;
    /* Flush all entries back to their respective pools */
    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        SNEPPXTlsEntry* e = &g_tls_cache->entries[i];
        while (e->free_list) {
            SNEPPXMemNode* node = (SNEPPXMemNode*)e->free_list;
            e->free_list = (void*)node->next;
            e->count--;
            lf_push(&g_pools[i].stack, node);
        }
    }
    free(g_tls_cache);
    g_tls_cache = NULL;
    SNEPPX_ATOMIC_ADD_INT(&g_active_tls_caches, -1);
}

/* ------------------------------------------------------------------
 *  SNEPPX_pool_alloc  –  fast path: TLS → global stack → grow
 * ------------------------------------------------------------------ */

/**
 * @brief Perform Pool Alloc.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pool_alloc(size_t size) {
    if (size > SNEPPX_POOL_MAX_SIZE)
        return SNEPPX_malloc(size, 16);

    int idx = SNEPPX_pool_size_class(size);
    if (idx < 0) return SNEPPX_malloc(size, 16);

    /* 1. lazily init TLS cache */
    if (!g_tls_cache) SNEPPX_tls_cache_init();

    /* 2. try TLS cache first */
    if (g_tls_cache) {
        SNEPPXTlsEntry* e = &g_tls_cache->entries[idx];
        if (e->free_list) {
            SNEPPXMemNode* node = (SNEPPXMemNode*)e->free_list;
            e->free_list = (void*)node->next;
            e->count--;
            g_tls_cache->hits++;
            memset(node, 0, g_size_classes[idx]);
            return node;
        }
    }

    /* 3. try global lock-free stack */
    SNEPPXMemNode* node = lf_pop(&g_pools[idx].stack);
    if (!node) {
        /* 4. grow the pool */
        if (pool_grow(&g_pools[idx]) != 0)
            return NULL;
        node = lf_pop(&g_pools[idx].stack);
        if (!node) return NULL;
    }
    g_pools[idx].alloc_count++;
    SNEPPX_ATOMIC_ADD_SIZE(&g_total_pool_allocated, g_size_classes[idx]);
    memset(node, 0, g_size_classes[idx]);
    return node;
}

/* ------------------------------------------------------------------
 *  SNEPPX_pool_free  –  return to TLS (or fall back to global)
 * ------------------------------------------------------------------ */

/**
 * @brief Free Pool.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_pool_free(void* ptr, size_t size) {
    if (!ptr) return;

    /* Large allocations were handled by SNEPPX_malloc */
    if (size > SNEPPX_POOL_MAX_SIZE) {
        SNEPPX_free(ptr, size);
        return;
    }

    int idx = SNEPPX_pool_size_class(size);
    if (idx < 0) {
        SNEPPX_free(ptr, size);
        return;
    }

    if (!g_tls_cache) SNEPPX_tls_cache_init();

    if (g_tls_cache) {
        SNEPPXTlsEntry* e = &g_tls_cache->entries[idx];
        if (e->count < e->max) {
            ((SNEPPXMemNode*)ptr)->next = (SNEPPXMemNode*)e->free_list;
            e->free_list = ptr;
            e->count++;
            return;
        }
    }

    /* TLS full or unavailable  →  return to global stack */
    lf_push(&g_pools[idx].stack, (SNEPPXMemNode*)ptr);
    g_pools[idx].free_count++;
    SNEPPX_ATOMIC_ADD_SIZE(&g_total_pool_freed, g_size_classes[idx]);
}

/* ------------------------------------------------------------------
 *  SNEPPX_mem_pool_destroy  –  tear down everything
 * ------------------------------------------------------------------ */

/**
 * @brief Destroy Mem Pool.
 */
void SNEPPX_mem_pool_destroy(void) {
    /* Destroy the calling thread's TLS cache first (flushes entries to pool) */
    if (g_tls_cache) SNEPPX_tls_cache_destroy();

    /* Free all chunks */
    ChunkHeader* h = g_chunk_list;
    while (h) {
        ChunkHeader* next = h->next;
        os_free(h->base, h->size);
        os_free(h, sizeof(ChunkHeader));
        h = next;
    }
    g_chunk_list = NULL;

    /* Reset pools */
    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        g_pools[i].stack.head  = NULL;
        g_pools[i].alloc_count = 0;
        g_pools[i].free_count  = 0;
        g_pools[i].chunk_count = 0;
    }

    g_pool_initialized      = 0;
    g_total_pool_allocated  = 0;
    g_total_pool_freed      = 0;
}

/* ------------------------------------------------------------------
 *  Statistics
 * ------------------------------------------------------------------ */

/**
 * @brief Perform Mem Pool Stats.
 */
void SNEPPX_mem_pool_stats(SNEPPXMemStats* stats) {
    if (!stats) return;
    stats->total_pool_allocated = g_total_pool_allocated;
    stats->total_pool_freed     = g_total_pool_freed;
    stats->total_pool_hits      = 0;
    stats->total_tls_hits       = 0;
    stats->total_chunks         = 0;
    stats->active_tls_caches    = (size_t)g_active_tls_caches;

    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        stats->total_chunks += g_pools[i].chunk_count;
        stats->total_pool_hits += g_pools[i].alloc_count;
    }
}

/**
 * @brief Perform Mem Pool Print Stats.
 */
void SNEPPX_mem_pool_print_stats(void) {
    SNEPPXMemStats s;
    SNEPPX_mem_pool_stats(&s);

    size_t tls_hits = g_tls_cache ? g_tls_cache->hits : 0;

    printf("--- SNEPPX Pool Allocator Stats ---\n");
    printf("Total pool allocated:  %zu bytes\n", s.total_pool_allocated);
    printf("Total pool freed:      %zu bytes\n", s.total_pool_freed);
    printf("Total chunks:          %zu\n", s.total_chunks);
    printf("Pool alloc hits:       %zu\n", s.total_pool_hits);
    printf("TLS cache hits:        %zu\n", tls_hits);
    printf("Active TLS caches:     %zu\n", s.active_tls_caches);

    printf("\nPer-class breakdown:\n");
    printf("  %-6s  %-8s  %-6s  %-6s  %-6s\n",
           "Class", "B/blk", "Chunks", "Allocs", "Frees");
    for (int i = 0; i < SNEPPX_NUM_SIZE_CLASSES; i++) {
        printf("  %-6zu  %-8zu  %-6zu  %-6zu  %-6zu\n",
               g_size_classes[i],
               g_pools[i].block_size,
               g_pools[i].chunk_count,
               g_pools[i].alloc_count,
               g_pools[i].free_count);
    }
}
