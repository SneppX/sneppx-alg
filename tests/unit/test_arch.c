#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include <stdio.h>

/*
 * SNEPPX - Test Arch
 *
 * WHAT
 *   Test Arch.
 *
 * CONCEPT
 *   Provides neural architecture.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_arch_has_avx(void) {
    int has = SNEPPX_arch_has_avx();
    SX_ASSERT(has == 0 || has == 1, "has_avx returns boolean");
}

static void test_arch_has_avx2(void) {
    int has = SNEPPX_arch_has_avx2();
    SX_ASSERT(has == 0 || has == 1, "has_avx2 returns boolean");
}

static void test_arch_has_neon(void) {
    int has = SNEPPX_arch_has_neon();
    SX_ASSERT(has == 0 || has == 1, "has_neon returns boolean");
}

static void test_arch_num_cores(void) {
    int cores = SNEPPX_arch_num_cores();
    SX_ASSERT(cores > 0, "num_cores > 0");
}

static void test_arch_cache_line_size(void) {
    int sz = SNEPPX_arch_cache_line_size();
    SX_ASSERT(sz > 0, "cache line size > 0");
}


TEST(test_arch, arch_has_avx) { test_arch_has_avx(); }
TEST(test_arch, arch_has_avx2) { test_arch_has_avx2(); }
TEST(test_arch, arch_has_neon) { test_arch_has_neon(); }
TEST(test_arch, arch_num_cores) { test_arch_num_cores(); }
TEST(test_arch, arch_cache_line_size) { test_arch_cache_line_size(); }
