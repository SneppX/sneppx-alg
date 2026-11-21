#ifndef SNEPPX_SHA256_H
#define SNEPPX_SHA256_H

#include <stdint.h>
#include <stddef.h>
/*
 * SNEPPX - SHA-256 Cryptographic Hash
 *
 * WHAT
 *   SHA-256 Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 180-4 SHA-256: 256-bit digest from arbitrary input.
 *
 * ROLE
 *   Used by HMAC, HKDF, DRBG, and as a general-purpose integrity hash.
 *
 * REFERENCES
 *   FIPS 180-4 (SHA-256).
 */



/**
 * @brief Compute SHA-256 hash of input.
 * @param uint8_t *out
 * @param const uint8_t *in
 * @param size_t inlen
 * @return 0 on success, -1 on error.
 */
void SNEPPX_sha256(uint8_t out[32], const uint8_t *in, size_t inlen);

#endif
