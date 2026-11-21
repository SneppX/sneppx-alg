#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdint.h>
#include "chacha20_stream_cipher.h"

/*
 * SNEPPX - Test Chacha20
 *
 * WHAT
 *   Test Chacha20.
 *
 * CONCEPT
 *   Provides ChaCha20 stream cipher (RFC 8439).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_vectors(void) {
    printf("\n--- test_vectors (RFC 8439 section 2.3.2) ---\n");
    uint8_t key[32] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
        0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
        0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f
    };
    uint8_t nonce[12] = {
        0x00,0x00,0x00,0x09,0x00,0x00,0x00,0x4a,
        0x00,0x00,0x00,0x00
    };
    uint8_t expected[64] = {
        0x10,0xf1,0xe7,0xe4,0xd1,0x3b,0x59,0x15,0x50,0x0f,0xdd,0x1f,0xa3,0x20,0x71,0xc4,
        0xc7,0xd1,0xf4,0xc7,0x33,0xc0,0x68,0x03,0x04,0x22,0xaa,0x9a,0xc3,0xd4,0x6c,0x4e,
        0xd2,0x82,0x64,0x46,0x07,0x9f,0xaa,0x09,0x14,0xc2,0xd7,0x05,0xd9,0x8b,0x02,0xa2,
        0xb5,0x12,0x9c,0xd1,0xde,0x16,0x4e,0xb9,0xcb,0xd0,0x83,0xe8,0xa2,0x50,0x3c,0x4e
    };

    SNEPPXChaCha20State state;
    SNEPPX_chacha20_init(&state, key, nonce, 1);
    uint8_t output[64];
    SNEPPX_chacha20_block(&state, output);
    int match = memcmp(output, expected, 64) == 0;
    if (!match) {
        printf("  Got:      "); for (int i = 0; i < 32; i++) printf("%02x", output[i]); printf("\n");
        printf("  Expected: "); for (int i = 0; i < 32; i++) printf("%02x", expected[i]); printf("\n");
    }
    SX_TEST("RFC 8439 test vector block 0", match);
}

void test_counter_overflow(void) {
    printf("\n--- test_counter_overflow ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);

    SNEPPXChaCha20State s1, s2;
    SNEPPX_chacha20_init(&s1, key, nonce, 0xFFFFFFFF);
    SNEPPX_chacha20_init(&s2, key, nonce, 0xFFFFFFFF);

    uint8_t b1[64], b2[64];
    SNEPPX_chacha20_block(&s1, b1);
    SNEPPX_chacha20_block(&s2, b2);
    SNEPPX_chacha20_block(&s2, b2);

    SX_TEST("counter wraps from 0xFFFFFFFF", s1.state[12] == 0);
    SX_TEST("two counter increments differ", memcmp(b1, b2, 64) != 0);
}

void test_keystream_distinct(void) {
    printf("\n--- test_keystream_distinct ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);

    SNEPPXChaCha20State s1, s2;
    SNEPPX_chacha20_init(&s1, key, nonce, 0);
    SNEPPX_chacha20_init(&s2, key, nonce, 1);

    uint8_t b1[64], b2[64];
    SNEPPX_chacha20_block(&s1, b1);
    SNEPPX_chacha20_block(&s2, b2);
    SX_TEST("different counter = different keystream", memcmp(b1, b2, 64) != 0);
}


TEST(test_chacha20, test_vectors) { test_vectors(); }
TEST(test_chacha20, test_counter_overflow) { test_counter_overflow(); }
TEST(test_chacha20, test_keystream_distinct) { test_keystream_distinct(); }
