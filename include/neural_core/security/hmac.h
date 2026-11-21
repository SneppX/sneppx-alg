#ifndef SNEPPX_HMAC_H
#define SNEPPX_HMAC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_HMAC_MAX_OUTPUT 64
#define SNEPPX_HMAC_MAX_KEY 128
/*
 * SNEPPX - HMAC (Hash-based Message Authentication Code)
 *
 * WHAT
 *   HMAC (Hash-based Message Authentication Code).
 *
 * CONCEPT
 *   RFC 2104 HMAC: a keyed hash providing integrity and authenticity.
 *
 * ROLE
 *   Used by HKDF (extract phase), DRBG reseeding, and keyed integrity checks.
 *
 * REFERENCES
 *   RFC 2104 (HMAC), FIPS 198-1.
 */



typedef struct {
    uint8_t key[SNEPPX_HMAC_MAX_KEY];
    size_t key_len;
    int hash_type;
} SNEPPXHMAC;

/**
 * @brief Compute HMAC-SHA256 of input with key.
 * @param uint8_t *out
 * @param const uint8_t *key
 * @param size_t keylen
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hmac_init(SNEPPXHMAC* ctx, const uint8_t* key, size_t key_len, int hash_type);
/**
 * @brief Compute HMAC-SHA256 of input with key.
 * @param uint8_t *out
 * @param const uint8_t *key
 * @param size_t keylen
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hmac_compute(SNEPPXHMAC* ctx, const uint8_t* data, size_t data_len, uint8_t* out, size_t* out_len);
/**
 * @brief Compute HMAC-SHA256 of input with key.
 * @param uint8_t *out
 * @param const uint8_t *key
 * @param size_t keylen
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[32]);
/**
 * @brief Compute HMAC-SHA256 of input with key.
 * @param uint8_t *out
 * @param const uint8_t *key
 * @param size_t keylen
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[64]);

#ifdef __cplusplus
}
#endif
#endif
