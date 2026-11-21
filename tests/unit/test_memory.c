#include "polymorphic_memory_allocator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Memory
 *
 * WHAT
 *   Test Memory.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





/* ── Pool init / destroy ───────────────────────────────────── */

static void test_pool_init_destroy(void) {
    int ret = SNEPPX_mem_pool_init();
    SX_ASSERT(ret == 0, "pool init returns 0");
    /* idempotent: calling again should succeed */
    ret = SNEPPX_mem_pool_init();
    SX_ASSERT(ret == 0, "pool init idempotent");
    SNEPPX_mem_pool_destroy();
}

/* ── TLS cache init / destroy ──────────────────────────────── */

static void test_tls_cache_init_destroy(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();
    /* second call is a no-op */
    SNEPPX_tls_cache_init();
    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Basic alloc / free for every size class ──────────────── */

static void test_alloc_free_each_class(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();

    static const size_t sizes[] = {
        1, 16, 17, 32, 48, 64, 96, 128, 192, 256,
        384, 512, 768, 1024, 1536, 2048, 3072, 4096,
        6144, 8192
    };
    size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < nsizes; i++) {
        void* p = SNEPPX_pool_alloc(sizes[i]);
        SX_ASSERT(p != NULL, "alloc returns non-NULL");
        memset(p, 0xAB, sizes[i]);
        SNEPPX_pool_free(p, sizes[i]);
    }

    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Allocated memory is zeroed ────────────────────────────── */

static void test_alloc_zeroed(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();

    void* p = SNEPPX_pool_alloc(128);
    SX_ASSERT(p != NULL, "alloc succeeds");
    unsigned char* bytes = (unsigned char*)p;
    for (int i = 0; i < 128; i++) {
        SX_ASSERT(bytes[i] == 0, "all bytes zeroed");
    }
    SNEPPX_pool_free(p, 128);

    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Many iterations (stress) ──────────────────────────────── */

static void test_stress_alloc_free(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();

    enum { N = 1000 };
    void** ptrs = (void**)malloc(N * sizeof(void*));
    SX_ASSERT(ptrs != NULL, "stress malloc");

    for (int i = 0; i < N; i++) {
        size_t sz = (size_t)(1 + (rand() % 4096));
        ptrs[i] = SNEPPX_pool_alloc(sz);
        SX_ASSERT(ptrs[i] != NULL, "stress alloc");
        memset(ptrs[i], (unsigned char)(i & 0xFF), sz);
    }
    for (int i = 0; i < N; i++) {
        SNEPPX_pool_free(ptrs[i], (size_t)(1 + (rand() % 4096)));
    }
    free(ptrs);

    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Size class behavior tested through public API ─────────── */

static void test_size_class_behavior(void) {
    /* Allocate at each class boundary and verify success */
    static const size_t sizes[] = {1, 16, 32, 48, 64, 128, 256, 512,
                                   1024, 2048, 4096, 8192, 8193};
    for (size_t i = 0; i < 13; i++) {
        void* p = SNEPPX_pool_alloc(sizes[i]);
        SX_ASSERT(p != NULL, "alloc at class boundary");
        SNEPPX_pool_free(p, sizes[i]);
    }
}

/* ── Large alloc falls back to SNEPPX_malloc ─────────────────── */

static void test_large_alloc_fallback(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();

    /* 1 MiB > pool max → should use SNEPPX_malloc under the hood */
    void* p = SNEPPX_pool_alloc(1024 * 1024);
    SX_ASSERT(p != NULL, "large alloc succeeds");
    memset(p, 0xFF, 1024 * 1024);
    SNEPPX_pool_free(p, 1024 * 1024);

    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Null safety for pool_free ─────────────────────────────── */

static void test_null_free(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_pool_free(NULL, 0);   /* must not crash */
    SNEPPX_pool_free(NULL, 128); /* must not crash */
    SNEPPX_mem_pool_destroy();
}

/* ── Stats sanity ──────────────────────────────────────────── */

static void test_stats_sanity(void) {
    SNEPPX_mem_pool_init();
    SNEPPX_tls_cache_init();

    SNEPPXMemStats s;
    SNEPPX_mem_pool_stats(&s);
    SX_ASSERT(s.total_chunks >= 0, "chunks >= 0");
    SX_ASSERT(s.active_tls_caches >= 1, "at least one TLS cache");

    void* p = SNEPPX_pool_alloc(256);
    SX_ASSERT(p != NULL, "alloc for stats");
    SNEPPX_pool_free(p, 256);

    SNEPPX_mem_pool_stats(&s);
    SX_ASSERT(s.total_pool_allocated >= 256, "allocated bytes tracked");

    SNEPPX_mem_pool_print_stats();

    SNEPPX_tls_cache_destroy();
    SNEPPX_mem_pool_destroy();
}

/* ── Multiple pool allocs without TLS ──────────────────────── */

static void test_alloc_without_tls(void) {
    SNEPPX_mem_pool_init();
    /* Do NOT init TLS cache — should still work via global stack */

    void* p1 = SNEPPX_pool_alloc(128);
    SX_ASSERT(p1 != NULL, "alloc without TLS works");
    void* p2 = SNEPPX_pool_alloc(128);
    SX_ASSERT(p2 != NULL, "second alloc works");
    SX_ASSERT(p1 != p2, "different pointers");

    SNEPPX_pool_free(p1, 128);
    SNEPPX_pool_free(p2, 128);

    SNEPPX_mem_pool_destroy();
}

/* ── Sequential destroy and re-init ────────────────────────── */

static void test_reinit(void) {
    for (int round = 0; round < 5; round++) {
        int ret = SNEPPX_mem_pool_init();
        SX_ASSERT(ret == 0, "re-init succeeds");
        SNEPPX_tls_cache_init();
        void* p = SNEPPX_pool_alloc(512);
        SX_ASSERT(p != NULL, "re-init alloc");
        SNEPPX_pool_free(p, 512);
        SNEPPX_tls_cache_destroy();
        SNEPPX_mem_pool_destroy();
    }
}

/* ── Mix of core and pool allocators ───────────────────────── */

static void test_mixed_allocators(void) {
    SNEPPX_mem_pool_init();

    void* a = SNEPPX_malloc(256, 64);
    SX_ASSERT(a != NULL, "core malloc");
    void* b = SNEPPX_pool_alloc(256);
    SX_ASSERT(b != NULL, "pool alloc");
    void* c = SNEPPX_malloc(4096, 16);
    SX_ASSERT(c != NULL, "core malloc large");

    SNEPPX_free(a, 256);
    SNEPPX_pool_free(b, 256);
    SNEPPX_free(c, 4096);

    SNEPPX_mem_pool_destroy();
}

/* ───────────────────────────────────────────────────────────── */


TEST(test_memory, pool_init_destroy) { test_pool_init_destroy(); }
TEST(test_memory, tls_cache_init_destroy) { test_tls_cache_init_destroy(); }
TEST(test_memory, alloc_free_each_class) { test_alloc_free_each_class(); }
TEST(test_memory, alloc_zeroed) { test_alloc_zeroed(); }
TEST(test_memory, stress_alloc_free) { test_stress_alloc_free(); }
TEST(test_memory, size_class_behavior) { test_size_class_behavior(); }
TEST(test_memory, large_alloc_fallback) { test_large_alloc_fallback(); }
TEST(test_memory, null_free) { test_null_free(); }
TEST(test_memory, stats_sanity) { test_stats_sanity(); }
TEST(test_memory, alloc_without_tls) { test_alloc_without_tls(); }
TEST(test_memory, reinit) { test_reinit(); }
TEST(test_memory, mixed_allocators) { test_mixed_allocators(); }
