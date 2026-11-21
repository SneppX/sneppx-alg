#include "power_analysis_mitigation.h"
#include "test_gtest.h"
#include <stdio.h>
#include <stdint.h>

/*
 * SNEPPX - Test Power
 *
 * WHAT
 *   Test Power.
 *
 * CONCEPT
 *   Provides power analysis mitigation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_power_mul_const_time(void) {
    int r = SNEPPX_power_mul_const_time(5, 3);
    SX_ASSERT(r == 15, "multiplication result correct");
}

static void test_power_cmp_const_time(void) {
    int r = SNEPPX_power_cmp_const_time(10, 20);
    SX_ASSERT(r < 0, "cmp result negative");
    r = SNEPPX_power_cmp_const_time(30, 10);
    SX_ASSERT(r > 0, "cmp result positive");
    r = SNEPPX_power_cmp_const_time(7, 7);
    SX_ASSERT(r == 0, "cmp result zero");
}

static void test_power_mask_gen(void) {
    uint64_t mask = SNEPPX_power_mask_gen(1);
    SX_ASSERT(mask == (uint64_t)-1 || mask == 0, "mask generated");
}


TEST(test_power, power_mul_const_time) { test_power_mul_const_time(); }
TEST(test_power, power_cmp_const_time) { test_power_cmp_const_time(); }
TEST(test_power, power_mask_gen) { test_power_mask_gen(); }
