#include "sha512_hashing_implementation.h"
#include "asm_exports.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Sha512
 *
 * WHAT
 *   Test Sha512.
 *
 * CONCEPT
 *   Provides SHA hash family.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_sha512_hash(void) {
    unsigned char hash[64];
    const unsigned char* input = (const unsigned char*)"hello";
    SNEPPX_sha512_hash(input, 5, hash);
    int non_zero = 0;
    for (int i = 0; i < 64; i++) if (hash[i]) non_zero++;
    SX_ASSERT(non_zero > 0, "sha512 hash produced non-zero output");
}

static void test_sha512_deterministic(void) {
    unsigned char h1[64], h2[64];
    const unsigned char* input = (const unsigned char*)"test";
    SNEPPX_sha512_hash(input, 4, h1);
    SNEPPX_sha512_hash(input, 4, h2);
    SX_ASSERT(memcmp(h1, h2, 64) == 0, "sha512 deterministic");
}

static void test_sha512_diff_input(void) {
    unsigned char h1[64], h2[64];
    const unsigned char* in1 = (const unsigned char*)"abc";
    const unsigned char* in2 = (const unsigned char*)"abd";
    SNEPPX_sha512_hash(in1, 3, h1);
    SNEPPX_sha512_hash(in2, 3, h2);
    SX_ASSERT(memcmp(h1, h2, 64) != 0, "sha512 different for diff input");
}

static void test_sha512_hmac(void) {
    unsigned char hmac[64];
    const unsigned char* key = (const unsigned char*)"key";
    const unsigned char* msg = (const unsigned char*)"message";
    SNEPPX_sha512_hmac(key, 3, msg, 7, hmac);
    int non_zero = 0;
    for (int i = 0; i < 64; i++) if (hmac[i]) non_zero++;
    SX_ASSERT(non_zero > 0, "sha512 hmac produced output");
}


TEST(test_sha512, sha512_hash) { test_sha512_hash(); }
TEST(test_sha512, sha512_deterministic) { test_sha512_deterministic(); }
TEST(test_sha512, sha512_diff_input) { test_sha512_diff_input(); }
TEST(test_sha512, sha512_hmac) { test_sha512_hmac(); }
