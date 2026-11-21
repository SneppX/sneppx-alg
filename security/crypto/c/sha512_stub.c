#include "sha512_hashing_implementation.h"
#include <string.h>
/*
 * SNEPPX - SHA-512 Stub (Platform Fallback)
 *
 * WHAT
 *   SHA-512 Stub (Platform Fallback).
 *
 * CONCEPT
 *   Software fallback SHA-512 for platforms without hardware SHA support.
 *
 * ROLE
 *   Fallback for SHA-512 on platforms without hardware support.
 *
 * REFERENCES
 *   FIPS 180-4 (SHA-512).
 */



/**
 * @brief Hash Sha512.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_sha512_hash(const uint8_t *data, size_t len, uint8_t hash[64]) {
    SNEPPX_sha512(data, len, hash);
}

/**
 * @brief Perform Sha512 Hmac.
 *
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 */
void SNEPPX_sha512_hmac(const uint8_t *key, size_t key_len, const uint8_t *data, size_t data_len, uint8_t mac[64]) {
    uint8_t k_ipad[128], k_opad[128];
    uint8_t inner_key[128];
    uint8_t inner_hash[64];
    size_t i;

    if (key_len > 128) {
        SNEPPX_sha512(key, key_len, inner_key);
        for (i = 0; i < 64; i++) {
            k_ipad[i] = inner_key[i] ^ 0x36;
            k_opad[i] = inner_key[i] ^ 0x5c;
        }
        memset(k_ipad + 64, 0x36, 64);
        memset(k_opad + 64, 0x5c, 64);
    } else {
        for (i = 0; i < key_len; i++) {
            k_ipad[i] = key[i] ^ 0x36;
            k_opad[i] = key[i] ^ 0x5c;
        }
        for (; i < 128; i++) {
            k_ipad[i] = 0x36;
            k_opad[i] = 0x5c;
        }
    }

    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, k_ipad, 128);
    SNEPPX_sha512_update(&ctx, data, data_len);
    SNEPPX_sha512_finish(&ctx, inner_hash);

    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, k_opad, 128);
    SNEPPX_sha512_update(&ctx, inner_hash, 64);
    SNEPPX_sha512_finish(&ctx, mac);
}
