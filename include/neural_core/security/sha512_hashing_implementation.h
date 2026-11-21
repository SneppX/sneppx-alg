#ifndef SNEPPX_SHA512_H
#define SNEPPX_SHA512_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SHA512_BLOCK_SIZE 128
#define SNEPPX_SHA512_DIGEST_SIZE 64
/*
 * SNEPPX - SHA-512 Cryptographic Hash
 *
 * WHAT
 *   SHA-512 Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 180-4 SHA-512: 512-bit digest from arbitrary input.
 *
 * ROLE
 *   Used by HMAC, HKDF, DRBG, and anywhere a 512-bit hash is needed.
 *
 * REFERENCES
 *   FIPS 180-4 (SHA-512).
 */



typedef struct {
    uint64_t state[8];
    uint64_t count[2];
    uint8_t buffer[SNEPPX_SHA512_BLOCK_SIZE];
    unsigned int buflen;
} SNEPPXSHA512Context;

/**
 * @brief Initialize Sha512.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_sha512_init(SNEPPXSHA512Context* ctx);
/**
 * @brief Compute SHA-512 hash of input.
 * @param uint8_t *out
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
void SNEPPX_sha512_update(SNEPPXSHA512Context* ctx, const uint8_t* data, size_t len);
/**
 * @brief Compute SHA-512 hash of input.
 * @param uint8_t *out
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
void SNEPPX_sha512_finish(SNEPPXSHA512Context* ctx, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);
/**
 * @brief Compute SHA-512 hash of input.
 * @param uint8_t *out
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
void SNEPPX_sha512(const uint8_t* data, size_t len, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);
/**
 * @brief Legacy SHA-512 finish (pre-RFC-8032 length encoding, 8x too large).
 *
 * Maintained for backward-compatible verification of manifests signed
 * before the length-encoding fix.  Use SNEPPX_sha512_legacy() or
 * SNEPPX_sha512_init/update + SNEPPX_sha512_legacy_finish().
 */
void SNEPPX_sha512_legacy_finish(SNEPPXSHA512Context* ctx, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);
/**
 * @brief Legacy SHA-512 (pre-RFC-8032 length encoding).
 */
void SNEPPX_sha512_legacy(const uint8_t* data, size_t len, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);


#ifdef __cplusplus
}
#endif
#endif
