#ifndef SNEPPX_SIPHASH_H
#define SNEPPX_SIPHASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_SIPHASH_KEY_SIZE 16
#define SNEPPX_SIPHASH_OUT_SIZE 8
/*
 * SNEPPX - SipHash 2-4 Pseudorandom Function
 *
 * WHAT
 *   SipHash 2-4 Pseudorandom Function.
 *
 * CONCEPT
 *   Fast, cryptographically secure PRF designed by Aumasson and Bernstein.
 *
 * ROLE
 *   Used for hash-table key hashing and as a building block in key derivation.
 *
 * REFERENCES
 *   Aumasson & Bernstein 2012 (SipHash).
 */



typedef struct {
    uint64_t v0, v1, v2, v3;
    uint64_t k0, k1;
    int c_rounds;
    int d_rounds;
} SNEPPXSipHash;

/**
 * @brief Initialize Siphash.
 *
 * @param sh [out] Sh value.
 */
void SNEPPX_siphash_init(SNEPPXSipHash* sh, const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE]);
/**
 * @brief Update Siphash.
 *
 * @param sh [out] Sh value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_siphash_update(SNEPPXSipHash* sh, const uint8_t* data, size_t len);
/**
 * @brief Compute SipHash 2-4 of input.
 * @param const uint8_t *key
 * @param const uint8_t *in
 * @param size_t inlen
 * @return hash value.
 */
uint64_t SNEPPX_siphash_finalize(SNEPPXSipHash* sh);
/**
 * @brief Compute SipHash 2-4 of input.
 * @param const uint8_t *key
 * @param const uint8_t *in
 * @param size_t inlen
 * @return hash value.
 */
uint64_t SNEPPX_siphash(const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE], const uint8_t* data, size_t len);

/**
 * @brief Initialize Siphash 24.
 *
 * @param sh [out] Sh value.
 */
void SNEPPX_siphash_24_init(SNEPPXSipHash* sh, const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE]);
/**
 * @brief Compute SipHash 2-4 of input.
 * @param const uint8_t *key
 * @param const uint8_t *in
 * @param size_t inlen
 * @return hash value.
 */
uint64_t SNEPPX_siphash_24(const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE], const uint8_t* data, size_t len);

#ifdef __cplusplus
}
#endif
#endif
