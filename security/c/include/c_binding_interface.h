#ifndef SNEPPX_C_SECURITY_WRAPPER_H
#define SNEPPX_C_SECURITY_WRAPPER_H
/*
 * SNEPPX - C Binding Interface
 *
 * WHAT
 *   C Binding Interface.
 *
 * CONCEPT
 *   Provides the C Binding Interface.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * C Language Security Bindings — v0.5
 *
 * PURPOSE: Thin C wrapper over the internal security/crypto/c library
 * for integration with systems that require a strict C89/C99 interface.
 * Functions here delegate to the implementations in security/crypto/c/
 * (aead.c, blake3.c, chacha20.c, etc.) with additional input validation.
 *
 * DEPENDENCIES: authenticated_encryption_module.h, chacha20_stream_cipher.h, cryptographic_hashing_blake3.h, keccak_sha3_hashing.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Hashing ---------- */
/**
 * @brief Perform C Hash Blake3.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_blake3(const uint8_t* data, size_t len, uint8_t* out, size_t out_len);
/**
 * @brief Perform C Hash Sha3 256.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_sha3_256(const uint8_t* data, size_t len, uint8_t out[32]);
/**
 * @brief Perform C Hash Sha3 512.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_sha3_512(const uint8_t* data, size_t len, uint8_t out[64]);

/* ---------- Symmetric encryption ---------- */
/**
 * @brief Encrypt C Chacha20.
 *
 * @param plaintext [in] Plaintext value.
 * @param len [in] Len value.
 * @param ciphertext [out] Ciphertext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* plaintext, size_t len, uint8_t* ciphertext);
/**
 * @brief Decrypt C Chacha20.
 *
 * @param ciphertext [in] Ciphertext value.
 * @param len [in] Len value.
 * @param plaintext [out] Plaintext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_chacha20_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* ciphertext, size_t len, uint8_t* plaintext);
/**
 * @brief Encrypt C Aead.
 *
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 * @param plaintext [in] Plaintext value.
 * @param plaintext_len [in] Plaintext Len value.
 * @param ciphertext [out] Ciphertext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_aead_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* plaintext, size_t plaintext_len,
                         uint8_t* ciphertext, uint8_t tag[16]);
/**
 * @brief Decrypt C Aead.
 *
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 * @param ciphertext [in] Ciphertext value.
 * @param ciphertext_len [in] Ciphertext Len value.
 * @param plaintext [out] Plaintext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* ciphertext, size_t ciphertext_len,
                         const uint8_t tag[16], uint8_t* plaintext);

/* ---------- Key derivation ---------- */
/**
 * @brief Hash C Argon2.
 *
 * @param password [in] Password value.
 * @param pwd_len [in] Pwd Len value.
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 * @param t_cost [in] T Cost value.
 * @param m_cost [in] M Cost value.
 * @param parallelism [in] Parallelism value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_argon2_hash(const char* password, size_t pwd_len,
                        const uint8_t* salt, size_t salt_len,
                        uint8_t* out, size_t out_len,
                        uint32_t t_cost, uint32_t m_cost, uint32_t parallelism);

/* ---------- Constant-time utilities ---------- */
/**
 * @brief Perform C Ct Memcmp.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_c_ct_memcmp(const void* a, const void* b, size_t len);
/**
 * @brief Perform C Ct Memzero.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 */
void SNEPPX_c_ct_memzero(void* ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_C_SECURITY_WRAPPER_H */
