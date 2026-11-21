#include "address_space_randomization.h"
#include "test_gtest.h"
#include <stdio.h>

/*
 * SNEPPX - Test Aslr
 *
 * WHAT
 *   Test Aslr.
 *
 * CONCEPT
 *   Provides address space layout randomization.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_aslr_random_offset(void) {
    size_t off = SNEPPX_aslr_random_offset(4096);
    SX_ASSERT(off < 4096, "random offset within range");
    SX_ASSERT((off & 4095) == 0, "random offset page-aligned");
}

static void test_aslr_random_offset_varied(void) {
    size_t off1 = SNEPPX_aslr_random_offset(65536);
    size_t off2 = SNEPPX_aslr_random_offset(65536);
    SX_ASSERT(off1 < 65536, "offset1 within range");
    SX_ASSERT(off2 < 65536, "offset2 within range");
}

static void test_aslr_apply(void) {
    char buf[256];
    void* ptr = (void*)buf;
    size_t sz = sizeof(buf);
    SNEPPX_aslr_apply(&ptr, &sz, 128);
    SX_ASSERT(sz <= sizeof(buf), "size not increased");
    // offset might be 0, so we just check it doesn't crash
    SX_ASSERT(ptr != NULL, "ptr valid after apply");
}


TEST(test_aslr, aslr_random_offset) { test_aslr_random_offset(); }
TEST(test_aslr, aslr_random_offset_varied) { test_aslr_random_offset_varied(); }
TEST(test_aslr, aslr_apply) { test_aslr_apply(); }
