#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include "timing_attack_countermeasure.h"

/*
 * SNEPPX - Test Timing
 *
 * WHAT
 *   Test Timing.
 *
 * CONCEPT
 *   Provides timing attack countermeasures.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */



void test_timer_resolution(void) {
    printf("\n--- test_timer_resolution ---\n");
    uint64_t t0 = SNEPPX_timing_start();
    uint64_t t1 = SNEPPX_timing_end();
    SX_TEST("timer produces value", t1 != t0 || t1 != 0);
    SX_TEST("timer monotonic", t1 >= t0);
}

void test_random_delay(void) {
    printf("\n--- test_random_delay ---\n");
    uint64_t t0 = SNEPPX_timing_start();
    SNEPPX_timing_random_delay(100, 5000);
    uint64_t t1 = SNEPPX_timing_end();
    SX_TEST("delay completed", t1 >= t0);
}

void test_safe_equal(void) {
    printf("\n--- test_safe_equal ---\n");
    uint8_t a[32] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                     17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    uint8_t b[32], c[32];
    memcpy(b, a, 32);
    memset(c, 0xFF, 32);
    uint64_t timing;
    int r1 = SNEPPX_timing_safe_equal(a, b, 32, &timing);
    SX_TEST("safe equal same", r1);
    int r2 = SNEPPX_timing_safe_equal(a, c, 32, &timing);
    SX_TEST("safe equal diff", !r2);
}


TEST(test_timing, test_timer_resolution) { test_timer_resolution(); }
TEST(test_timing, test_random_delay) { test_random_delay(); }
TEST(test_timing, test_safe_equal) { test_safe_equal(); }
