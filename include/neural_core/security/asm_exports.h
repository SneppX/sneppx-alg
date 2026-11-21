#ifndef SNEPPX_ASM_EXPORTS_H
#define SNEPPX_ASM_EXPORTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Asm Exports
 *
 * WHAT
 *   Asm Exports.
 *
 * CONCEPT
 *   Provides the Asm Exports.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ChaCha20
void sneppx_chacha20_init_state(uint32_t state[16], const uint32_t key[8], const uint32_t nonce[3], uint32_t counter);
void sneppx_chacha20_block(uint8_t keystream[64], const uint32_t state[16]);
void sneppx_chacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[3], uint32_t counter);
void sneppx_xchacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[4], uint32_t counter);
void sneppx_chacha20_poly1305_aead_encrypt(uint8_t *out, const uint8_t *in, size_t len);
void sneppx_chacha20_wipe_state(uint32_t state[16]);

// SHA-256
void sneppx_sha256_transform(uint32_t state[8], const uint8_t block[64]);
void sneppx_sha256_hash(uint8_t out[32], const uint8_t *in, size_t len);
void sneppx_sha256_hmac(uint8_t out[32], const uint8_t *key, size_t key_len, const uint8_t *msg, size_t msg_len);
void sneppx_sha256_wipe_state(uint32_t state[8]);

// AES-NI
void sneppx_aes_encrypt_block(const uint32_t rk[60], const uint8_t in[16], uint8_t out[16]);
void sneppx_aes_expand_key(const uint8_t key[32], uint32_t rk[60]);
void sneppx_aes_gfmul(uint8_t x[16], uint8_t y[16], uint8_t out[16]);
void sneppx_aes_gcm_ghash(const uint8_t h[16], const uint8_t *aad, size_t aad_len, const uint8_t *ct, size_t ct_len, uint8_t out[16]);
void sneppx_aes_ctr_encrypt(const uint32_t rk[60], uint8_t *out, const uint8_t *in, size_t len, uint8_t iv[16]);
int sneppx_aes_gcm_init(uint8_t out_tag[16], const uint8_t key[32], const uint8_t iv[12], const uint8_t *aad, size_t aad_len, const uint8_t *plaintext, const uint8_t *ciphertext, size_t len, int encrypt);
void sneppx_aes_cmac(const uint8_t key[32], const uint8_t *in, size_t in_len, uint8_t out[16]);
void sneppx_aes_wipe_key(uint32_t rk[60]);

// Poly1305

void sneppx_poly1305_mac(uint8_t mac[16], const uint8_t *msg, size_t msg_len, const uint8_t key[32]);
int sneppx_poly1305_verify(const uint8_t mac1[16], const uint8_t mac2[16]);
void sneppx_poly1305_wipe_key(uint32_t key[8]);

// SHA-512
/**
 * @brief Hash Sha512.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_sha512_hash(const uint8_t *data, size_t len, uint8_t hash[64]);
/**
 * @brief Perform Sha512 Hmac.
 *
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 */
void SNEPPX_sha512_hmac(const uint8_t *key, size_t key_len, const uint8_t *data, size_t data_len, uint8_t mac[64]);

// Keccak
void sneppx_keccak_theta(uint64_t state[25]);
void sneppx_keccak_f1600(uint64_t state[25]);
void sneppx_keccak_absorb(uint64_t state[25], const uint8_t *in, size_t len, size_t rate);
void sneppx_keccak_squeeze(uint64_t state[25], uint8_t *out, size_t out_len, size_t rate);

// Ed25519
void sneppx_ed25519_reduce(uint8_t out[32], const uint8_t scalar[32]);
void sneppx_ed25519_fe_mul(uint8_t out[32], const uint8_t a[32], const uint8_t b[32]);
void sneppx_ed25519_point_double(uint8_t out[32], const uint8_t point[32]);
void sneppx_ed25519_point_add(uint8_t out[32], const uint8_t p[32], const uint8_t q[32]);
void sneppx_ed25519_scalarmult(uint8_t out[32], const uint8_t scalar[32], const uint8_t point[32]);

// Curve25519 field ops
void sneppx_fe25519_add(uint64_t *result, const uint64_t *a, const uint64_t *b);
void sneppx_fe25519_sub(uint64_t *result, const uint64_t *a, const uint64_t *b);
void sneppx_fe25519_mul(uint64_t *result, const uint64_t *a, const uint64_t *b);
void sneppx_fe25519_sq(uint64_t *result, const uint64_t *a);
void sneppx_fe25519_inv(uint64_t *result, const uint64_t *a);

// Montgomery multiplication
void sneppx_montgomery_mul(uint64_t *result, const uint64_t *a, const uint64_t *b, const uint64_t *mod, uint64_t n0);
void sneppx_montgomery_redc(uint64_t *result, const uint64_t *t, const uint64_t *mod, uint64_t n0);
void sneppx_montgomery_to_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t n0);
void sneppx_montgomery_from_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t n0);

// Barrett reduction
uint16_t sneppx_barrett_reduce_u16(uint16_t a, uint16_t m, uint32_t mu);
uint32_t sneppx_barrett_reduce_u32(uint64_t a, uint32_t m, uint64_t mu);
uint16_t sneppx_barrett_reduce_u16_mont(uint32_t a, uint16_t m, uint32_t mu);
void sneppx_barrett_reduce_poly(uint32_t *out, const uint32_t *in, size_t n, uint32_t m, uint64_t mu);
uint32_t sneppx_barrett_reduce_u32_tight(uint64_t a, uint32_t m, uint64_t mu);

// GF(256) arithmetic
uint8_t sneppx_gf256_mul(uint8_t a, uint8_t b);
uint8_t sneppx_gf256_inv(uint8_t a);
uint8_t sneppx_gf256_sq(uint8_t a);
uint8_t sneppx_gf256_mul_scalar(const uint8_t *vec, uint8_t s, size_t n);

// Constant-time comparison
int sneppx_ct_compare_u64(const uint64_t *a, const uint64_t *b);
int sneppx_ct_compare_bytes(const uint8_t *a, const uint8_t *b, size_t len);
void sneppx_ct_conditional_swap(uint64_t *a, uint64_t *b, int condition);
uint64_t sneppx_ct_select_u64(int condition, uint64_t a, uint64_t b);
uint64_t sneppx_ct_conditional_negate(uint64_t val, int condition);
int sneppx_ct_is_zero_u64(const uint64_t *a);
int sneppx_ct_is_nonzero_u64(const uint64_t *a);
uint32_t sneppx_ct_compare_u32_accum(const uint32_t *a, const uint32_t *b, size_t len);
void sneppx_ct_conditional_copy(uint8_t *dst, const uint8_t *src, size_t len, int condition);
void sneppx_ct_conditional_memzero(uint8_t *dst, size_t len, int condition);
int sneppx_ct_compare_16(const uint8_t *a, const uint8_t *b);

// CT swap operations
void sneppx_ct_swap_u64(uint64_t *a, uint64_t *b);
void sneppx_ct_swap_bytes(uint8_t *a, uint8_t *b, size_t len);
uint64_t sneppx_ct_negate_u64(uint64_t val);
void sneppx_ct_cswap_4x64(uint64_t *a, uint64_t *b, int condition);
void sneppx_ct_cswap_8x64(uint64_t *a, uint64_t *b, int condition);
void sneppx_ct_cmask_u64(uint64_t *buf, uint64_t mask, size_t len);
uint64_t sneppx_ct_abs_u64(uint64_t val);
uint8_t sneppx_ct_mask_u8(uint8_t condition);
uint64_t sneppx_ct_mask_u64(uint64_t condition);

// SC cmov operations
uint64_t sneppx_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b);
int sneppx_sc_equal_u64(uint64_t a, uint64_t b);
void sneppx_sc_cond_swap_u64(uint64_t *a, uint64_t *b, uint64_t condition);
uint64_t sneppx_sc_cond_neg_u64(uint64_t val, uint64_t condition);
void sneppx_sc_cond_copy_u64(uint64_t *dst, const uint64_t *src, uint64_t condition, size_t len);
int sneppx_sc_memcmp_ct(const uint8_t *a, const uint8_t *b, size_t len);
uint64_t sneppx_sc_mask_load_u64(const uint64_t *ptr);
void sneppx_sc_secure_zero(uint8_t *dst, size_t len);
int sneppx_sc_ct_is_zero(const uint8_t *a, size_t len);
int sneppx_sc_ct_is_equal_or(uint64_t a, uint64_t b, uint64_t c);

// Cache-constant lookups
uint8_t sneppx_cache_const_lookup(const uint8_t *table, size_t table_size, size_t idx);
uint32_t sneppx_cache_const_lookup32(const uint32_t *table, size_t table_size, size_t idx);
void sneppx_cache_const_sbox(uint8_t out[16], const uint8_t in[16], const uint8_t sbox[256]);
uint32_t sneppx_cache_const_te0_lookup(const uint32_t *te0, size_t idx);

// Secure wipe
void sneppx_secure_wipe(void *ptr, size_t len);
void sneppx_secure_wipe_register_state(void);
void sneppx_secure_wipe_page(void *page);
void sneppx_secure_wipe_xmm(void);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_ASM_EXPORTS_H
