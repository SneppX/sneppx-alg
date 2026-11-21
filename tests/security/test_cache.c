#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdint.h>
#include "secure_cache_management.h"

/*
 * SNEPPX - Test Cache
 *
 * WHAT
 *   Test Cache.
 *
 * CONCEPT
 *   Provides cache timing protection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */



void test_flush(void) {
    printf("\n--- test_flush_prefetch ---\n");
    uint8_t buf[256];
    memset(buf, 0x42, 256);
    SNEPPX_cache_flush(buf, 256);
    SX_TEST("cache flush no crash", 1);
}

void test_prefetch(void) {
    uint8_t val = 42;
    SNEPPX_cache_prefetch(&val);
    SX_TEST("cache prefetch no crash", 1);
}

void test_barrier(void) {
    SNEPPX_cache_barrier();
    SX_TEST("cache barrier no crash", 1);
}


TEST(test_cache, test_flush) { test_flush(); }
TEST(test_cache, test_prefetch) { test_prefetch(); }
TEST(test_cache, test_barrier) { test_barrier(); }
