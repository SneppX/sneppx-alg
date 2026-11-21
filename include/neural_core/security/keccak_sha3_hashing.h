#ifndef SNEPPX_SHA3_H
#define SNEPPX_SHA3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SHA3_256_DIGEST_SIZE 32
#define SNEPPX_SHA3_512_DIGEST_SIZE 64
#define SNEPPX_SHA3_STATE_SIZE 200
/*
 * SNEPPX - SHA-3 / Keccak Cryptographic Hash
 *
 * WHAT
 *   SHA-3 / Keccak Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 202 SHA-3 using the Keccak sponge construction.
 *
 * ROLE
 *   Used by DRBG (Hash-DRBG), key-vault hashing, and general-purpose hashing.
 *
 * REFERENCES
 *   FIPS 202 (SHA-3 / Keccak).
 */



typedef struct {
    uint64_t state[25];
    unsigned int rate;
    unsigned int capacity;
    unsigned int buflen;
    uint8_t buffer[SNEPPX_SHA3_STATE_SIZE];
    unsigned int digest_size;
} SNEPPXSHA3State;

/**
 * @brief Initialize Sha3 256.
 *
 * @param state [out] State value.
 */
void SNEPPX_sha3_256_init(SNEPPXSHA3State* state);
/**
 * @brief Initialize Sha3 512.
 *
 * @param state [out] State value.
 */
void SNEPPX_sha3_512_init(SNEPPXSHA3State* state);
/**
 * @brief Update Sha3.
 *
 * @param state [out] State value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_sha3_update(SNEPPXSHA3State* state, const uint8_t* data, size_t len);
/**
 * @brief Perform Sha3 Finish.
 *
 * @param state [out] State value.
 * @param hash [out] Hash value.
 */
void SNEPPX_sha3_finish(SNEPPXSHA3State* state, uint8_t* hash);


#ifdef __cplusplus
}
#endif
#endif
