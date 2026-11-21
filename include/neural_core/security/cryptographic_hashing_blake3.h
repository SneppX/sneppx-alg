#ifndef SNEPPX_BLAKE3_H
#define SNEPPX_BLAKE3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_BLAKE3_OUT_LEN 32
#define SNEPPX_BLAKE3_BLOCK_LEN 64
#define SNEPPX_BLAKE3_CHUNK_LEN 1024

/*
 * SNEPPX - Cryptographic Hashing Blake3
 *
 * WHAT
 *   Cryptographic Hashing Blake3.
 *
 * CONCEPT
 *   Provides the Cryptographic Hashing Blake3.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint32_t key[8];
    uint64_t counter;
    uint8_t buf[SNEPPX_BLAKE3_CHUNK_LEN];
    size_t buflen;
    uint8_t flags;
} SNEPPXBlake3State;

/**
 * @brief Initialize Blake3.
 *
 * @param state [out] State value.
 */
void SNEPPX_blake3_init(SNEPPXBlake3State* state);
/**
 * @brief Update Blake3.
 *
 * @param state [out] State value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_blake3_update(SNEPPXBlake3State* state, const uint8_t* data, size_t len);
/**
 * @brief Perform Blake3 Finish.
 *
 * @param state [out] State value.
 * @param hash [out] Hash value.
 */
void SNEPPX_blake3_finish(SNEPPXBlake3State* state, uint8_t* hash);


#ifdef __cplusplus
}
#endif
#endif
