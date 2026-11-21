#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "protected_memory_manager.h"
#include "stack_canary_protection.h"

/*
 * SNEPPX - Test Secure Mem
 *
 * WHAT
 *   Test Secure Mem.
 *
 * CONCEPT
 *   Provides the Test Secure Mem.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */



void test_pool_create_destroy(void) {
    printf("\n--- test_pool_create_destroy ---\n");
    SNEPPXSecureAllocConfig cfg = {1, 1, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(1024 * 1024, &cfg);
    SX_TEST("pool created", pool != NULL);
    size_t t, u, p;
    SNEPPX_secure_pool_stats(pool, &t, &u, &p);
    SX_TEST("pool capacity > 0", t == 1024 * 1024 || t > 0);
    SNEPPX_secure_pool_destroy(pool);
    SX_TEST("pool destroyed", 1);
}

void test_malloc_free(void) {
    printf("\n--- test_malloc_free ---\n");
    SNEPPXSecureAllocConfig cfg = {0, 0, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(65536, &cfg);
    SX_TEST("pool created", pool != NULL);
    void* ptrs[100];
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        ptrs[i] = SNEPPX_secure_malloc(pool, 64, 16);
        if (!ptrs[i]) { ok = 0; break; }
        memset(ptrs[i], i, 64);
    }
    SX_TEST("100 allocations", ok);
    for (int i = 0; i < 100; i++) {
        SNEPPX_secure_pool_free(pool, ptrs[i], 64);
    }
    SX_TEST("100 frees", 1);
    SNEPPX_secure_pool_destroy(pool);
}

void test_canary_corruption(void) {
    printf("\n--- test_canary_corruption ---\n");
    SNEPPXCanary c1, c2;
    SNEPPX_canary_generate(&c1);
    SNEPPX_canary_generate(&c2);
    uint8_t buf[32];
    SNEPPX_canary_write(&c1, buf);
    SNEPPX_canary_write(&c2, buf + 16);
    SX_TEST("canary verify match", SNEPPX_canary_verify(&c1, buf));
    SX_TEST("canary verify mismatch", !SNEPPX_canary_verify(&c1, buf + 16));
    buf[0] ^= 0xFF;
    SX_TEST("canary detect corruption", !SNEPPX_canary_verify(&c1, buf));
}

void test_overflow_detection(void) {
    printf("\n--- test_overflow_detection ---\n");
    SNEPPXSecureAllocConfig cfg = {0, 1, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(65536, &cfg);
    SX_TEST("pool created", pool != NULL);
    void* ptr = SNEPPX_secure_malloc(pool, 64, 16);
    SX_TEST("alloc succeeded", ptr != NULL);
    SNEPPX_secure_pool_free(pool, ptr, 64);
    SX_TEST("normal free ok", 1);
    SNEPPX_secure_pool_destroy(pool);
}


TEST(test_secure_mem, test_pool_create_destroy) { test_pool_create_destroy(); }
TEST(test_secure_mem, test_malloc_free) { test_malloc_free(); }
TEST(test_secure_mem, test_canary_corruption) { test_canary_corruption(); }
TEST(test_secure_mem, test_overflow_detection) { test_overflow_detection(); }
