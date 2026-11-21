#include "../include/c_binding_interface.h"
#include "authenticated_encryption_module.h"
#include "chacha20_stream_cipher.h"
#include "cryptographic_hashing_blake3.h"
#include "keccak_sha3_hashing.h"
#include "memory_hard_key_derivation.h"
#include "constant_time_operations.h"
#include <string.h>

/*
 * SNEPPX - C Binding Implementation
 *
 * WHAT
 *   C Binding Implementation.
 *
 * CONCEPT
 *   Provides the C Binding Implementation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform C Hash Blake3.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_blake3(const uint8_t* data, size_t len, uint8_t* out, size_t out_len) {
    if (!data || !out || out_len < 32) return -1;
    SNEPPXBlake3State ctx;
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, data, len);
    SNEPPX_blake3_finish(&ctx, out);
    return 0;
}

/**
 * @brief Perform C Hash Sha3 256.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_sha3_256(const uint8_t* data, size_t len, uint8_t out[32]) {
    if (!data || !out) return -1;
    SNEPPXSHA3State ctx;
    SNEPPX_sha3_256_init(&ctx);
    SNEPPX_sha3_update(&ctx, data, len);
    SNEPPX_sha3_finish(&ctx, out);
    return 0;
}

/**
 * @brief Perform C Hash Sha3 512.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_hash_sha3_512(const uint8_t* data, size_t len, uint8_t out[64]) {
    if (!data || !out) return -1;
    SNEPPXSHA3State ctx;
    SNEPPX_sha3_512_init(&ctx);
    SNEPPX_sha3_update(&ctx, data, len);
    SNEPPX_sha3_finish(&ctx, out);
    return 0;
}

/**
 * @brief Encrypt C Chacha20.
 *
 * @param plaintext [in] Plaintext value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* plaintext, size_t len, uint8_t* ciphertext) {
    if (!key || !nonce || !plaintext || !ciphertext) return -1;
    SNEPPXChaCha20State state;
    SNEPPX_chacha20_init(&state, key, nonce, 0);
    memcpy(ciphertext, plaintext, len);
    SNEPPX_chacha20_encrypt(&state, ciphertext, len);
    return 0;
}

/**
 * @brief Decrypt C Chacha20.
 *
 * @param ciphertext [in] Ciphertext value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_chacha20_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                             const uint8_t* ciphertext, size_t len, uint8_t* plaintext) {
    if (!key || !nonce || !ciphertext || !plaintext) return -1;
    SNEPPXChaCha20State state;
    SNEPPX_chacha20_init(&state, key, nonce, 0);
    memcpy(plaintext, ciphertext, len);
    SNEPPX_chacha20_encrypt(&state, plaintext, len);
    return 0;
}

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
                         uint8_t* ciphertext, uint8_t tag[16]) {
    if (!key || !nonce || !plaintext || !ciphertext || !tag) return -1;
    return SNEPPX_aead_encrypt(ciphertext, tag, plaintext, plaintext_len, aad, aad_len, key, nonce);
}

/**
 * @brief Decrypt C Aead.
 *
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 * @param ciphertext [in] Ciphertext value.
 * @param ciphertext_len [in] Ciphertext Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_aead_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                         const uint8_t* aad, size_t aad_len,
                         const uint8_t* ciphertext, size_t ciphertext_len,
                         const uint8_t tag[16], uint8_t* plaintext) {
    if (!key || !nonce || !ciphertext || !plaintext) return -1;
    return SNEPPX_aead_decrypt(plaintext, ciphertext, ciphertext_len, tag, aad, aad_len, key, nonce);
}

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
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_argon2_hash(const char* password, size_t pwd_len,
                        const uint8_t* salt, size_t salt_len,
                        uint8_t* out, size_t out_len,
                        uint32_t t_cost, uint32_t m_cost, uint32_t parallelism) {
    if (!password || !salt || !out) return -1;
    SNEPPXArgon2Config cfg;
    cfg.memory_kb = m_cost;
    cfg.iterations = t_cost;
    cfg.parallelism = parallelism;
    cfg.hash_len = out_len;
    return SNEPPX_argon2id((const uint8_t*)password, pwd_len, salt, salt_len, &cfg, out);
}

/**
 * @brief Perform C Ct Memcmp.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_c_ct_memcmp(const void* a, const void* b, size_t len) {
    if (!a || !b) return -1;
    return SNEPPX_ct_equal(a, b, len) ? 0 : 1;
}

/**
 * @brief Perform C Ct Memzero.
 *
 * @param ptr [out] Ptr value.
 */
void SNEPPX_c_ct_memzero(void* ptr, size_t len) {
    if (ptr) {
        volatile unsigned char* p = (volatile unsigned char*)ptr;
        for (size_t i = 0; i < len; i++) p[i] = 0;
    }
}
