#include "secure_cache_management.h"
#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <emmintrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

/*
 * SNEPPX - Cache-Line Hardening
 *
 * WHAT
 *   Provides cache-flush, prefetch, and memory-barrier primitives for
 *   cache-line hardening against side-channel attacks.
 *
 * CONCEPT
 *   Flushes specific cache lines using CLFLUSH (x86) or DC CIVAC (ARM),
 *   prefetches data into L1 to avoid timing leaks from page faults, and
 *   issues a full memory fence (MFENCE/DSB) to prevent reordering of
 *   memory operations across the barrier.
 *
 * ROLE
 *   Internal hardening module (cache-line hardening). Used by secure_mem
 *   and timing countermeasure components.
 *
 * REFERENCES
 *   Internal hardening module.
 */

/**
 * @brief Flush cache lines for a memory region.
 * @param ptr  Start of the memory region.
 * @param len  Length of the region in bytes.
 */
void SNEPPX_cache_flush(const void* ptr, size_t len) {
    if (!ptr) return;
    const uint8_t* p = (const uint8_t*)ptr;
    for (size_t i = 0; i < len; i += 64) {
#if defined(_MSC_VER)
        _mm_clflush(&p[i]);
#elif defined(__x86_64__) || defined(__amd64__)
        _mm_clflush(&p[i]);
#elif defined(__aarch64__)
        __asm__ volatile("dc civac, %0" :: "r"(&p[i]) : "memory");
#else
        (void)p;
#endif
    }
}

/**
 * @brief Prefetch data into cache.
 * @param ptr  Pointer to the data to prefetch.
 */
void SNEPPX_cache_prefetch(const void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#elif defined(__x86_64__) || defined(__amd64__)
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#elif defined(__aarch64__)
    __asm__ volatile("prfm pldl1keep, [%0]" :: "r"(ptr));
#else
    (void)ptr;
#endif
}

/**
 * @brief Issue a full memory barrier.
 */
void SNEPPX_cache_barrier(void) {
#if defined(_MSC_VER)
    _mm_mfence();
#elif defined(__x86_64__) || defined(__amd64__)
    _mm_mfence();
#elif defined(__aarch64__)
    __asm__ volatile("dsb ish" : : : "memory");
#endif
}
