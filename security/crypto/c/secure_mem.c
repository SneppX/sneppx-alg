#include "protected_memory_manager.h"
#include "stack_canary_protection.h"
#include "address_space_randomization.h"
#include "synchronization_lock_interface.h"
#include "cryptographic_random_generator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

/*
 * SNEPPX - Secure Mem
 *
 * WHAT
 *   Secure Mem.
 *
 * CONCEPT
 *   Provides the Secure Mem.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXSecurePool {
    uint8_t* base;
    uint8_t* raw_base;   /* original allocation (before guard/ASLR offset) */
    size_t capacity;
    size_t raw_len;      /* full mapping length for munmap/VirtualFree */
    size_t used;
    size_t peak;
    int has_guards;
    int use_canaries;
    SNEPPXCanary pool_canary;
    uint64_t alloc_count;
};

#if defined(_WIN32)
static size_t get_page_size(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}
#else
static size_t get_page_size(void) {
    return (size_t)sysconf(_SC_PAGESIZE);
}
#endif

/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_secure_zero(void* ptr, size_t len) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    for (size_t i = 0; i < len; i++) p[i] = 0;
}

/**
 * @brief Create Secure Pool.
 *
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSecurePool* SNEPPX_secure_pool_create(size_t size, const SNEPPXSecureAllocConfig* config) {
    size_t page = get_page_size();
    size_t alloc_size = size + (config && config->guard_pages ? 2 * page : 0);
    size_t random_off = 0;
    uint8_t* base = NULL;

#if defined(_WIN32)
    base = (uint8_t*)VirtualAlloc(NULL, alloc_size + page, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return NULL;
    if (config && config->guard_pages) {
        VirtualAlloc(base + page, size, MEM_COMMIT, PAGE_READWRITE);
        VirtualAlloc(base, page, MEM_COMMIT, PAGE_NOACCESS);
        VirtualAlloc(base + page + size, page, MEM_COMMIT, PAGE_NOACCESS);
    } else {
        VirtualAlloc(base, size, MEM_COMMIT, PAGE_READWRITE);
    }
#else
    base = (uint8_t*)mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    if (config && config->guard_pages) {
        mprotect(base, page, PROT_NONE);
        mprotect(base + page + size, page, PROT_NONE);
    }
#endif

    SNEPPXSecurePool* pool = (SNEPPXSecurePool*)malloc(sizeof(SNEPPXSecurePool));
    if (!pool) {
#if defined(_WIN32)
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, alloc_size);
#endif
        return NULL;
    }

    uint8_t* usable = base + (config && config->guard_pages ? page : 0);
    size_t usable_size = size;

    pool->raw_base = base;
    pool->raw_len = alloc_size;

    if (config && config->randomize_layout) {
        size_t max_off = usable_size / 4;
        if (max_off > page) max_off = page;
        random_off = SNEPPX_aslr_random_offset(max_off);
        usable += random_off;
        usable_size -= random_off;
    }

    if (config && config->lock_memory) {
        SNEPPX_mlock(usable, usable_size);
    }

    pool->base = usable;
    pool->capacity = usable_size;
    pool->used = 0;
    pool->peak = 0;
    pool->has_guards = (config && config->guard_pages) ? 1 : 0;
    pool->use_canaries = (config && config->canaries) ? 1 : 0;
    pool->alloc_count = 0;
    SNEPPX_canary_generate(&pool->pool_canary);
    return pool;
}

/**
 * @brief Destroy Secure Pool.
 */
void SNEPPX_secure_pool_destroy(SNEPPXSecurePool* pool) {
    if (!pool) return;
    SNEPPX_secure_zero(pool->base, pool->capacity);
    SNEPPX_munlock(pool->base, pool->capacity);
#if defined(_WIN32)
    VirtualFree(pool->raw_base, 0, MEM_RELEASE);
#else
    munmap(pool->raw_base, pool->raw_len);
#endif
/*
 * SNEPPX - Secure Memory Allocation
 *
 * WHAT
 *   Secure Memory Allocation.
 *
 * CONCEPT
 *   Allocator that locks pages, prevents swapping, and wipes memory on free.
 *
 * ROLE
 *   Used by every crypto module handling secret keys and intermediate values.
 *
 * REFERENCES
 *   None (internal utility).
 */


    SNEPPX_secure_zero((void*)pool, sizeof(SNEPPXSecurePool));
    free(pool);
}

/**
 * @brief Perform Secure Malloc.
 *
 * @param pool [out] Pool value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_malloc(SNEPPXSecurePool* pool, size_t size, size_t alignment) {
    if (!pool || !size) return NULL;
    if (alignment < 16) alignment = 16;
    size_t header = pool->use_canaries ? SNEPPX_CANARY_SIZE : 0;
    size_t aligned_offset = (pool->used + alignment - 1) & ~(alignment - 1);
    size_t needed = aligned_offset + header + size + header;
    if (needed > pool->capacity) return NULL;
    uint8_t* ptr = pool->base + aligned_offset + header;
    if (pool->use_canaries) {
        SNEPPXCanary canary;
        SNEPPX_canary_generate(&canary);
        SNEPPX_canary_write(&canary, ptr - SNEPPX_CANARY_SIZE);
        SNEPPX_canary_write(&canary, ptr + size);
    }
    pool->used = needed;
    if (pool->used > pool->peak) pool->peak = pool->used;
    pool->alloc_count++;
    return ptr;
}

/**
 * @brief Free Secure Pool.
 *
 * @param pool [out] Pool value.
 * @param ptr [out] Ptr value.
 */
void SNEPPX_secure_pool_free(SNEPPXSecurePool* pool, void* ptr, size_t size) {
    if (!pool || !ptr) return;
    if (pool->use_canaries) {
        uint8_t* mem = (uint8_t*)ptr;
        SNEPPXCanary* lead = (SNEPPXCanary*)(mem - SNEPPX_CANARY_SIZE);
        SNEPPXCanary* trail = (SNEPPXCanary*)(mem + size);
        if (!SNEPPX_canary_verify(lead, lead->value)) {
            fprintf(stderr, "HEAP CORRUPTION: leading canary mismatch at %p\n", ptr);
            abort();
        }
        if (!SNEPPX_canary_verify(trail, trail->value)) {
            fprintf(stderr, "HEAP CORRUPTION: trailing canary mismatch at %p\n", ptr);
            abort();
        }
    }
    SNEPPX_secure_zero(ptr, size);
}

/**
 * @brief Perform Secure Realloc.
 *
 * @param pool [out] Pool value.
 * @param ptr [out] Ptr value.
 * @param old_size [in] Old Size value.
 * @param new_size [in] New Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_secure_realloc(SNEPPXSecurePool* pool, void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (!pool) return NULL;
    if (!ptr) return SNEPPX_secure_malloc(pool, new_size, alignment);
    void* new_ptr = SNEPPX_secure_malloc(pool, new_size, alignment);
    if (!new_ptr) return NULL;
    size_t copy = old_size < new_size ? old_size : new_size;
    memcpy(new_ptr, ptr, copy);
    SNEPPX_secure_pool_free(pool, ptr, old_size);
    return new_ptr;
}

/**
 * @brief Perform Secure Pool Stats.
 *
 * @param pool [out] Pool value.
 * @param total [out] Total value.
 * @param used [out] Used value.
 */
void SNEPPX_secure_pool_stats(SNEPPXSecurePool* pool, size_t* total, size_t* used, size_t* peak) {
    if (!pool) return;
    if (total) *total = pool->capacity;
    if (used) *used = pool->used;
    if (peak) *peak = pool->peak;
}
