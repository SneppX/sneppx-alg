#ifndef SNEPPX_AEAD_H
#define SNEPPX_AEAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Authenticated Encryption Module
 *
 * WHAT
 *   Authenticated Encryption Module.
 *
 * CONCEPT
 *   Provides the Authenticated Encryption Module.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


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
                      const uint8_t* aad, size_t aad_len, const uint8_t key[32], const uint8_t nonce[12]);

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
                      const uint8_t key[32], const uint8_t nonce[12]);


#ifdef __cplusplus
}
#endif
#endif
