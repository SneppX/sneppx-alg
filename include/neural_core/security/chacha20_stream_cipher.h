#ifndef SNEPPX_CHACHA20_H
#define SNEPPX_CHACHA20_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - ChaCha20 Stream Cipher
 *
 * WHAT
 *   ChaCha20 Stream Cipher.
 *
 * CONCEPT
 *   ChaCha20 256-bit stream cipher with 96-bit nonce and 32-bit counter.
 *
 * ROLE
 *   Layer S2 authenticated encryption (ChaCha20-Poly1305).
 *
 * REFERENCES
 *   RFC 8439 (ChaCha20-Poly1305), RFC 7539.
 */



typedef struct {
    uint32_t state[16];
} SNEPPXChaCha20State;

/**
 * @brief Initialize Chacha20.
 *
 * @param state [out] State value.
 * @param counter [in] Counter value.
 */
void SNEPPX_chacha20_init(SNEPPXChaCha20State* state, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter);
/**
 * @brief Perform Chacha20 Block.
 *
 * @param state [out] State value.
 */
void SNEPPX_chacha20_block(SNEPPXChaCha20State* state, uint8_t output[64]);
/**
 * @brief Encrypt/decrypt with ChaCha20 (XOR stream).
 * @param uint8_t *ciphertext
 * @param const uint8_t *plaintext
 * @param size_t len
 * @param const uint8_t *key
 * @param const uint8_t *nonce
 * @return 0 on success, -1 on error.
 */
void SNEPPX_chacha20_encrypt(SNEPPXChaCha20State* state, uint8_t* data, size_t len);


#ifdef __cplusplus
}
#endif
#endif
