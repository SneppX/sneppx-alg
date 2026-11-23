#include "hmac.h"
#include "test_gtest.h"
#include <string.h>

/*
 * SNEPPX - Test HMAC
 *
 * WHAT
 *   Known-answer tests for HMAC-SHA256 and an output-buffer overrun guard.
 *
 * CONCEPT
 *   RFC 4231 vectors verify the keyed hash; the trailing canary bytes detect
 *   any implementation that writes a 64-byte SHA-512 digest into the
 *   32-byte HMAC-SHA256 output buffer.
 *
 * ROLE
 *   SNEPPX-Algo core security component.
 */

static void test_hmac_sha256_vectors(void)
{
    static const uint8_t expected1[32] = {0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf,
                                          0xce, 0xaf, 0x0b, 0xf1, 0x2b, 0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83,
                                          0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7};
    static const uint8_t expected2[32] = {0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24,
                                          0x26, 0x08, 0x95, 0x75, 0xc7, 0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27,
                                          0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec, 0x38, 0x43};

    uint8_t key1[20];
    memset(key1, 0x0b, sizeof(key1));
    const char* data1 = "Hi There";
    uint8_t out[48];
    memset(out, 0xCC, sizeof(out));
    SX_ASSERT(SNEPPX_hmac_sha256(key1, sizeof(key1), (const uint8_t*) data1, strlen(data1), out) == 0,
              "hmac sha256 vector 1");
    SX_ASSERT(memcmp(out, expected1, sizeof(expected1)) == 0, "hmac sha256 vector 1 digest");
    for (size_t i = sizeof(expected1); i < sizeof(out); i++)
        SX_ASSERT(out[i] == 0xCC, "hmac sha256 output buffer overrun");

    const char* key2 = "Jefe";
    const char* data2 = "what do ya want for nothing?";
    memset(out, 0xCC, sizeof(out));
    SX_ASSERT(SNEPPX_hmac_sha256((const uint8_t*) key2, strlen(key2), (const uint8_t*) data2, strlen(data2), out) == 0,
              "hmac sha256 vector 2");
    SX_ASSERT(memcmp(out, expected2, sizeof(expected2)) == 0, "hmac sha256 vector 2 digest");
    for (size_t i = sizeof(expected2); i < sizeof(out); i++)
        SX_ASSERT(out[i] == 0xCC, "hmac sha256 output buffer overrun");
}

static void test_hmac_sha256_long_key(void)
{
    uint8_t key[128];
    memset(key, 0xAA, sizeof(key));
    const char* data = "long key";
    uint8_t out[48];
    memset(out, 0xCC, sizeof(out));
    SX_ASSERT(SNEPPX_hmac_sha256(key, sizeof(key), (const uint8_t*) data, strlen(data), out) == 0,
              "hmac sha256 long key");
    for (size_t i = 32; i < sizeof(out); i++)
        SX_ASSERT(out[i] == 0xCC, "hmac sha256 long key output buffer overrun");
}

TEST(test_hmac, sha256_rfc4231_vectors)
{
    test_hmac_sha256_vectors();
}

TEST(test_hmac, sha256_long_key)
{
    test_hmac_sha256_long_key();
}
