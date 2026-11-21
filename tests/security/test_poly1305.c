#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include "polynomial_authentication_mac.h"

/*
 * SNEPPX - Test Poly1305
 *
 * WHAT
 *   Test Poly1305.
 *
 * CONCEPT
 *   Provides the Test Poly1305.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_vectors(void) {
    printf("\n--- test_vectors (RFC 8439) ---\n");
    uint8_t key[32] = {
        0x85,0xd6,0xbe,0x78,0x57,0x55,0x6d,0x33,
        0x7f,0x44,0x52,0xfe,0x42,0xd5,0x06,0xa8,
        0x01,0x03,0x80,0x8a,0xfb,0x0d,0xb2,0xfd,
        0x4a,0xbf,0xf6,0xaf,0x41,0x49,0xf5,0x1b
    };
    uint8_t msg[] = "Cryptographic Forum Research Group";
    uint8_t expected[] = {
        0xa8,0x06,0x1d,0xc1,0x30,0x51,0x36,0xc6,
        0xc2,0x2b,0x8b,0xaf,0x0c,0x01,0x27,0xa9
    };

    SNEPPXPoly1305State state;
    SNEPPX_poly1305_init(&state, key);
    SNEPPX_poly1305_update(&state, msg, sizeof(msg)-1);
    uint8_t mac[16];
    SNEPPX_poly1305_finish(&state, mac);
    int match = memcmp(mac, expected, 16) == 0;
    if (!match) {
        printf("  Got:      "); for (int i = 0; i < 16; i++) printf("%02x", mac[i]); printf("\n");
        printf("  Expected: "); for (int i = 0; i < 16; i++) printf("%02x", expected[i]); printf("\n");
    }
    SX_TEST("RFC 8439 test vector", match);
}

void test_key_reuse(void) {
    printf("\n--- test_key_reuse ---\n");
    uint8_t key[32];
    memset(key, 0x42, 32);

    uint8_t mac1[16], mac2[16];
    SNEPPXPoly1305State s;
    SNEPPX_poly1305_init(&s, key);
    SNEPPX_poly1305_update(&s, (uint8_t*)"Message one", 11);
    SNEPPX_poly1305_finish(&s, mac1);

    SNEPPX_poly1305_init(&s, key);
    SNEPPX_poly1305_update(&s, (uint8_t*)"Message two", 11);
    SNEPPX_poly1305_finish(&s, mac2);

    SX_TEST("different messages = different tags", memcmp(mac1, mac2, 16) != 0);
}


TEST(test_poly1305, test_vectors) { test_vectors(); }
TEST(test_poly1305, test_key_reuse) { test_key_reuse(); }
