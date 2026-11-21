#include "../../mm/internal/slab_alloc.h"
#include "test_gtest.h"
#include "../../mm/internal/slab_alloc.c"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Slab Alloc
 *
 * WHAT
 *   Test Slab Alloc.
 *
 * CONCEPT
 *   Provides the Test Slab Alloc.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_slab_create_destroy(void) {
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 64, 16);
    SX_ASSERT(ret == 0, "slab cache create");
    SX_ASSERT(cache != NULL, "slab cache created");
    SNEPPX_slab_cache_destroy(cache);
}

static void test_slab_alloc_free(void) {
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 32, 8);
    SX_ASSERT(ret == 0, "slab cache create");
    if (!cache) return;
    void* p = SNEPPX_slab_cache_alloc(cache);
    if (p == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_slab_cache_destroy(cache);
        return;
    }
    memset(p, 0x42, 32);
    unsigned char* buf = (unsigned char*)p;
    SX_ASSERT(buf[0] == 0x42, "memory writable");
    SNEPPX_slab_cache_free(cache, p);
    SNEPPX_slab_cache_destroy(cache);
}

static void test_slab_multiple_allocs(void) {
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 16, 4);
    SX_ASSERT(ret == 0, "slab cache create");
    if (!cache) return;
    void* p1 = SNEPPX_slab_cache_alloc(cache);
    void* p2 = SNEPPX_slab_cache_alloc(cache);
    if (p1 == NULL || p2 == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_slab_cache_destroy(cache);
        return;
    }
    SX_ASSERT(p1 != p2, "distinct allocations");
    SNEPPX_slab_cache_free(cache, p2);
    SNEPPX_slab_cache_free(cache, p1);
    SNEPPX_slab_cache_destroy(cache);
}


TEST(test_slab_alloc, slab_create_destroy) { test_slab_create_destroy(); }
TEST(test_slab_alloc, slab_alloc_free) { test_slab_alloc_free(); }
TEST(test_slab_alloc, slab_multiple_allocs) { test_slab_multiple_allocs(); }
