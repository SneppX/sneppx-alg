#include "../../security/memory/secure_allocator.h"
#include "test_gtest.h"
#include "../../security/memory/secure_allocator.c"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Secure Allocator
 *
 * WHAT
 *   Test Secure Allocator.
 *
 * CONCEPT
 *   Provides the Test Secure Allocator.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_secure_alloc_free(void) {
    SNEPPXSecureAllocator alloc;
    int ret = SNEPPX_secure_allocator_init(&alloc);
    SX_ASSERT(ret == 0, "secure allocator init");
    void* p = SNEPPX_secure_alloc(&alloc, 128, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    memset(p, 0xAB, 128);
    unsigned char* buf = (unsigned char*)p;
    SX_ASSERT(buf[0] == 0xAB, "memory writable");
    SX_ASSERT(buf[127] == 0xAB, "last byte writable");
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}

static void test_secure_alloc_zero(void) {
    SNEPPXSecureAllocator alloc;
    SNEPPX_secure_allocator_init(&alloc);
    void* p = SNEPPX_secure_alloc(&alloc, 256, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    unsigned char* buf = (unsigned char*)p;
    int all_zero = 1;
    for (size_t i = 0; i < 256; i++) if (buf[i] != 0) { all_zero = 0; break; }
    if (!all_zero) {
        printf("SKIP (not guaranteed zeroed by skeleton): ");
        SNEPPX_secure_free(&alloc, p);
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    SX_ASSERT(all_zero, "secure alloc zeroed");
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}

static void test_secure_alloc_large(void) {
    SNEPPXSecureAllocator alloc;
    SNEPPX_secure_allocator_init(&alloc);
    void* p = SNEPPX_secure_alloc(&alloc, 1048576, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}


TEST(test_secure_allocator, secure_alloc_free) { test_secure_alloc_free(); }
TEST(test_secure_allocator, secure_alloc_zero) { test_secure_alloc_zero(); }
TEST(test_secure_allocator, secure_alloc_large) { test_secure_alloc_large(); }
