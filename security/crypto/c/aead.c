#include "authenticated_encryption_module.h"
#include "chacha20_stream_cipher.h"
#include "polynomial_authentication_mac.h"
#include "constant_time_operations.h"
#include <string.h>
#include <stdlib.h>
/*
 * SNEPPX - AEAD (Authenticated Encryption with Associated Data)
 *
 * WHAT
 *   AEAD (Authenticated Encryption with Associated Data).
 *
 * CONCEPT
 *   Combines symmetric encryption with a MAC covering both ciphertext and associated data.
 *
 * ROLE
 *   Layer S2 authenticated encryption for transport and key-vault components.
 *
 * REFERENCES
 *   NIST SP 800-38D (GCM), RFC 8439 (ChaCha20-Poly1305).
 */



static void poly1305_key_gen(const uint8_t key[32], const uint8_t nonce[12], uint8_t poly_key[32]) {
    SNEPPXChaCha20State state;
    uint8_t block[64];
    SNEPPX_chacha20_init(&state, key, nonce, 0);
    SNEPPX_chacha20_block(&state, block);
    memcpy(poly_key, block, 32);
}

/**
 * @brief Encrypt Aead.
 *
 * @param ciphertext [out] Ciphertext value.
 * @param plaintext [in] Plaintext value.
 * @param len [in] Len value.
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_aead_encrypt(uint8_t* ciphertext, uint8_t tag[16], const uint8_t* plaintext, size_t len,
                      const uint8_t* aad, size_t aad_len, const uint8_t key[32], const uint8_t nonce[12]) {
    if (!ciphertext || !tag || !plaintext || !key || !nonce) return -1;
    uint8_t poly_key[32];
    poly1305_key_gen(key, nonce, poly_key);
    SNEPPXChaCha20State state;
    SNEPPX_chacha20_init(&state, key, nonce, 1);
    if (ciphertext != plaintext) memcpy(ciphertext, plaintext, len);
    SNEPPX_chacha20_encrypt(&state, ciphertext, len);
    SNEPPXPoly1305State mac;
    SNEPPX_poly1305_init(&mac, poly_key);
    if (aad && aad_len) SNEPPX_poly1305_update(&mac, aad, aad_len);
    size_t pad_aad = aad_len ? (16 - aad_len % 16) % 16 : 0;
    uint8_t zeros[16] = {0};
    if (pad_aad) SNEPPX_poly1305_update(&mac, zeros, pad_aad);
    if (len) SNEPPX_poly1305_update(&mac, ciphertext, len);
    size_t pad_ct = (16 - len % 16) % 16;
    if (pad_ct) SNEPPX_poly1305_update(&mac, zeros, pad_ct);
    uint8_t len_block[16];
    for (int i = 0; i < 8; i++) {
        len_block[i] = (uint8_t)(aad_len >> (i * 8));
        len_block[i + 8] = (uint8_t)(len >> (i * 8));
    }
    SNEPPX_poly1305_update(&mac, len_block, 16);
    SNEPPX_poly1305_finish(&mac, tag);
    return 0;
}

/**
 * @brief Decrypt Aead.
 *
 * @param plaintext [out] Plaintext value.
 * @param ciphertext [in] Ciphertext value.
 * @param len [in] Len value.
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_aead_decrypt(uint8_t* plaintext, const uint8_t* ciphertext, size_t len,
                      const uint8_t tag[16], const uint8_t* aad, size_t aad_len,
                      const uint8_t key[32], const uint8_t nonce[12]) {
    if (!plaintext || !ciphertext || !tag || !key || !nonce) return -1;
    uint8_t poly_key[32];
    uint8_t block[64];
    {   SNEPPXChaCha20State ks;
        SNEPPX_chacha20_init(&ks, key, nonce, 0);
        SNEPPX_chacha20_block(&ks, block);
    }
    memcpy(poly_key, block, 32);
    SNEPPXPoly1305State mac;
    SNEPPX_poly1305_init(&mac, poly_key);
    if (aad && aad_len) SNEPPX_poly1305_update(&mac, aad, aad_len);
    size_t pad_aad = aad_len ? (16 - aad_len % 16) % 16 : 0;
    uint8_t zeros[16] = {0};
    if (pad_aad) SNEPPX_poly1305_update(&mac, zeros, pad_aad);
    if (len) SNEPPX_poly1305_update(&mac, ciphertext, len);
    size_t pad_ct = (16 - len % 16) % 16;
    if (pad_ct) SNEPPX_poly1305_update(&mac, zeros, pad_ct);
    uint8_t len_block[16];
    for (int i = 0; i < 8; i++) {
        len_block[i] = (uint8_t)(aad_len >> (i * 8));
        len_block[i + 8] = (uint8_t)(len >> (i * 8));
    }
    SNEPPX_poly1305_update(&mac, len_block, 16);
    uint8_t expected_tag[16];
    SNEPPX_poly1305_finish(&mac, expected_tag);
    int ok = SNEPPX_ct_equal(tag, expected_tag, 16);
    if (ok) {
        SNEPPXChaCha20State state;
        SNEPPX_chacha20_init(&state, key, nonce, 1);
        if (plaintext != ciphertext) memcpy(plaintext, ciphertext, len);
        SNEPPX_chacha20_encrypt(&state, plaintext, len);
    }
    return ok ? 0 : -1;
}
