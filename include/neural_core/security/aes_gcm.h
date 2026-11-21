#ifndef SNEPPX_AES_GCM_H
#define SNEPPX_AES_GCM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AES_BLOCK_SIZE 16
#define SNEPPX_AES256_KEY_SIZE 32
#define SNEPPX_GCM_IV_SIZE 12
#define SNEPPX_GCM_TAG_SIZE 16
#define SNEPPX_GCM_MAX_AAD 65536
#define SNEPPX_GCM_MAX_PLAINTEXT (1ULL << 36)
/*
 * SNEPPX - AES-GCM (Galois/Counter Mode)
 *
 * WHAT
 *   AES-GCM (Galois/Counter Mode).
 *
 * CONCEPT
 *   AES-128/192/256 encryption in CTR mode with GHASH authentication tag.
 *
 * ROLE
 *   Layer S2 authenticated encryption for TLS record protection and key wrapping.
 *
 * REFERENCES
 *   NIST SP 800-38D (GCM), FIPS 197 (AES).
 */



typedef struct {
    uint32_t rk[60];
    int rounds;
    uint8_t h[SNEPPX_AES_BLOCK_SIZE];
    uint8_t j0[SNEPPX_AES_BLOCK_SIZE];
    uint8_t y[SNEPPX_AES_BLOCK_SIZE];
    uint8_t tag[SNEPPX_GCM_TAG_SIZE];
    uint64_t aad_len;
    uint64_t crypt_len;
    int mode;
} SNEPPXAESGCM;

/**
 * @brief Expand Aes256.
 */
void SNEPPX_aes256_key_expansion(const uint8_t key[SNEPPX_AES256_KEY_SIZE], uint32_t rk[60]);
/**
 * @brief Encrypt a single block with Aes256.
 */
void SNEPPX_aes256_encrypt_block(const uint32_t rk[60], const uint8_t in[SNEPPX_AES_BLOCK_SIZE], uint8_t out[SNEPPX_AES_BLOCK_SIZE]);
/**
 * @brief Decrypt a single block with Aes256.
 */
void SNEPPX_aes256_decrypt_block(const uint32_t rk[60], const uint8_t in[SNEPPX_AES_BLOCK_SIZE], uint8_t out[SNEPPX_AES_BLOCK_SIZE]);

/**
 * @brief Initialize Aes Gcm.
 *
 * @param ctx [out] Ctx value.
 * @param encrypt [in] Encrypt value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_aes_gcm_init(SNEPPXAESGCM* ctx, const uint8_t key[SNEPPX_AES256_KEY_SIZE], const uint8_t iv[SNEPPX_GCM_IV_SIZE], int encrypt);
/**
 * @brief Perform Aes Gcm Update Aad.
 *
 * @param ctx [out] Ctx value.
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 */
void SNEPPX_aes_gcm_update_aad(SNEPPXAESGCM* ctx, const uint8_t* aad, size_t aad_len);
/**
 * @brief Encrypt plaintext with AES-GCM.
 * @param uint8_t *ciphertext
 * @param size_t *ciphertext_len
 * @param const uint8_t *plaintext
 * @param size_t plaintext_len
 * @param const uint8_t *key
 * @param const uint8_t *nonce
 * @param size_t nonce_len
 * @param const uint8_t *aad
 * @param size_t aad_len
 * @return 0 on success, -1 on error.
 */
void SNEPPX_aes_gcm_encrypt(SNEPPXAESGCM* ctx, const uint8_t* plaintext, uint8_t* ciphertext, size_t len);
/**
 * @brief Decrypt ciphertext with AES-GCM.
 * @param uint8_t *plaintext
 * @param size_t *plaintext_len
 * @param const uint8_t *ciphertext
 * @param size_t ciphertext_len
 * @param const uint8_t *key
 * @param const uint8_t *nonce
 * @param size_t nonce_len
 * @param const uint8_t *aad
 * @param size_t aad_len
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_aes_gcm_decrypt(SNEPPXAESGCM* ctx, const uint8_t* ciphertext, uint8_t* plaintext, size_t len);
/**
 * @brief Perform Aes Gcm Finalize.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_aes_gcm_finalize(SNEPPXAESGCM* ctx, uint8_t tag[SNEPPX_GCM_TAG_SIZE]);
/**
 * @brief Perform Aes Gcm Verify Tag.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_aes_gcm_verify_tag(SNEPPXAESGCM* ctx, const uint8_t expected_tag[SNEPPX_GCM_TAG_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
