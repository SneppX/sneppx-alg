#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdlib.h>
#include "keccak_sha3_hashing.h"

/*
 * SNEPPX - Test Sha3
 *
 * WHAT
 *   Test Sha3.
 *
 * CONCEPT
 *   Provides SHA hash family.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_vectors_256(void) {
    printf("\n--- test_vectors_256 ---\n");
    uint8_t input[] = "abc";
    uint8_t expected[32] = {
        0x3a,0x98,0x5d,0xa7,0x4f,0xe2,0x25,0xb2,
        0x04,0x5c,0x17,0x2d,0x6b,0xd3,0x90,0xbd,
        0x85,0x5f,0x08,0x6e,0x3e,0x9d,0x52,0x5b,
        0x46,0xbf,0xe2,0x45,0x11,0x43,0x15,0x32
    };
    SNEPPXSHA3State state;
    SNEPPX_sha3_256_init(&state);
    SNEPPX_sha3_update(&state, input, 3);
    uint8_t hash[32];
    SNEPPX_sha3_finish(&state, hash);
    int match256 = memcmp(hash, expected, 32) == 0;
    if (!match256) { printf("  Got: "); for (int i = 0; i < 32; i++) printf("%02x", hash[i]); printf("\n  Exp: "); for (int i = 0; i < 32; i++) printf("%02x", expected[i]); printf("\n"); }
    SX_TEST("SHA3-256('abc')", match256);
}

void test_vectors_512(void) {
    printf("\n--- test_vectors_512 ---\n");
    uint8_t input[] = "abc";
    uint8_t expected[64] = {
        0xb7,0x51,0x85,0x0b,0x1a,0x57,0x16,0x8a,
        0x56,0x93,0xcd,0x92,0x4b,0x6b,0x09,0x6e,
        0x08,0xf6,0x21,0x82,0x74,0x44,0xf7,0x0d,
        0x88,0x4f,0x5d,0x02,0x40,0xd2,0x71,0x2e,
        0x10,0xe1,0x16,0xe9,0x19,0x2a,0xf3,0xc9,
        0x1a,0x7e,0xc5,0x76,0x47,0xe3,0x93,0x40,
        0x57,0x34,0x0b,0x4c,0xf4,0x08,0xd5,0xa5,
        0x65,0x92,0xf8,0x27,0x4e,0xec,0x53,0xf0
    };
    SNEPPXSHA3State state;
    SNEPPX_sha3_512_init(&state);
    SNEPPX_sha3_update(&state, input, 3);
    uint8_t hash[64];
    SNEPPX_sha3_finish(&state, hash);
    int match512 = memcmp(hash, expected, 64) == 0;
    if (!match512) { printf("  Got: "); for (int i = 0; i < 64; i++) printf("%02x", hash[i]); printf("\n  Exp: "); for (int i = 0; i < 64; i++) printf("%02x", expected[i]); printf("\n"); }
    SX_TEST("SHA3-512('abc')", match512);
}

void test_empty_message(void) {
    printf("\n--- test_empty_message ---\n");
    uint8_t hash[32];
    SNEPPXSHA3State state;
    SNEPPX_sha3_256_init(&state);
    SNEPPX_sha3_finish(&state, hash);
    uint8_t non_zero = 0;
    for (int i = 0; i < 32; i++) non_zero |= hash[i];
    SX_TEST("empty hash non-zero", non_zero != 0);
}

void test_large_message(void) {
    printf("\n--- test_large_message ---\n");
    size_t len = 1024 * 1024;
    uint8_t* data = (uint8_t*)malloc(len);
    if (!data) { printf("SKIP: large_message (malloc failed)\n"); return; }
    memset(data, 0, len);
    SNEPPXSHA3State state;
    SNEPPX_sha3_256_init(&state);
    SNEPPX_sha3_update(&state, data, len);
    uint8_t hash[32];
    SNEPPX_sha3_finish(&state, hash);
    uint8_t non_zero = 0;
    for (int i = 0; i < 32; i++) non_zero |= hash[i];
    SX_TEST("1MB hash non-zero", non_zero != 0);
    free(data);
}


TEST(test_sha3, test_vectors_256) { test_vectors_256(); }
TEST(test_sha3, test_vectors_512) { test_vectors_512(); }
TEST(test_sha3, test_empty_message) { test_empty_message(); }
TEST(test_sha3, test_large_message) { test_large_message(); }
