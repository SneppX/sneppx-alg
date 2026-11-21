#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdlib.h>
#include "authenticated_encryption_module.h"

/*
 * SNEPPX - Test Aead
 *
 * WHAT
 *   Test Aead.
 *
 * CONCEPT
 *   Provides the Test Aead.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_encrypt_decrypt(void) {
    printf("\n--- test_encrypt_decrypt ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);
    uint8_t plaintext[] = "Hello, SNEPPX-Algo AEAD!";
    size_t len = sizeof(plaintext);
    uint8_t ciphertext[256], decrypted[256];
    uint8_t tag[16];

    int ret = SNEPPX_aead_encrypt(ciphertext, tag, plaintext, len, NULL, 0, key, nonce);
    SX_TEST("encrypt success", ret == 0);

    ret = SNEPPX_aead_decrypt(decrypted, ciphertext, len, tag, NULL, 0, key, nonce);
    SX_TEST("decrypt success", ret == 0);
    SX_TEST("plaintext matches", memcmp(plaintext, decrypted, len) == 0);
}

void test_aad_integrity(void) {
    printf("\n--- test_aad_integrity ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);
    uint8_t pt[] = "Test with AAD";
    uint8_t aad[] = "authenticated but unencrypted";
    uint8_t ct[256], dt[256], tag[16];

    SNEPPX_aead_encrypt(ct, tag, pt, sizeof(pt), aad, sizeof(aad), key, nonce);

    uint8_t wrong_aad[] = "tampered AAD";
    int ret = SNEPPX_aead_decrypt(dt, ct, sizeof(pt), tag, wrong_aad, sizeof(wrong_aad), key, nonce);
    SX_TEST("modified AAD fails", ret != 0);
}

void test_ciphertext_integrity(void) {
    printf("\n--- test_ciphertext_integrity ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);
    uint8_t pt[] = "Test ciphertext integrity";
    uint8_t ct[256], dt[256], tag[16];

    SNEPPX_aead_encrypt(ct, tag, pt, sizeof(pt), NULL, 0, key, nonce);
    ct[5] ^= 0x42;
    int ret = SNEPPX_aead_decrypt(dt, ct, sizeof(pt), tag, NULL, 0, key, nonce);
    SX_TEST("modified ciphertext fails", ret != 0);
}

void test_tag_integrity(void) {
    printf("\n--- test_tag_integrity ---\n");
    uint8_t key[32], nonce[12];
    memset(key, 0x42, 32); memset(nonce, 0, 12);
    uint8_t pt[] = "Test tag integrity";
    uint8_t ct[256], dt[256], tag[16];

    SNEPPX_aead_encrypt(ct, tag, pt, sizeof(pt), NULL, 0, key, nonce);
    tag[0] ^= 0xFF;
    int ret = SNEPPX_aead_decrypt(dt, ct, sizeof(pt), tag, NULL, 0, key, nonce);
    SX_TEST("modified tag fails", ret != 0);
}


TEST(test_aead, test_encrypt_decrypt) { test_encrypt_decrypt(); }
TEST(test_aead, test_aad_integrity) { test_aad_integrity(); }
TEST(test_aead, test_ciphertext_integrity) { test_ciphertext_integrity(); }
TEST(test_aead, test_tag_integrity) { test_tag_integrity(); }
