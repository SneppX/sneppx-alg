#ifndef SNEPPX_ED25519_H
#define SNEPPX_ED25519_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_ED25519_PUBLIC_KEY_LEN 32
#define SNEPPX_ED25519_PRIVATE_KEY_LEN 64
#define SNEPPX_ED25519_SIGNATURE_LEN 64
#define SNEPPX_ED25519_SEED_LEN 32
/*
 * SNEPPX - Ed25519 Digital Signatures
 *
 * WHAT
 *   Ed25519 Digital Signatures.
 *
 * CONCEPT
 *   Ed25519 signature generation and verification on the Edwards curve Ed25519.
 *
 * ROLE
 *   Used for code-signing, release manifests, and compact authentication.
 *
 * REFERENCES
 *   RFC 8032 (EdDSA).
 */



typedef struct {
    uint8_t public_key[SNEPPX_ED25519_PUBLIC_KEY_LEN];
    uint8_t private_key[SNEPPX_ED25519_PRIVATE_KEY_LEN];
} SNEPPXEd25519Keypair;

typedef struct {
    uint8_t data[SNEPPX_ED25519_SIGNATURE_LEN];
} SNEPPXEd25519Signature;

/**
 * @brief Perform Ed25519 Keypair Generate.
 *
 * @param kp [out] Kp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_keypair_generate(SNEPPXEd25519Keypair* kp);
/**
 * @brief Perform Ed25519 Secret Key Expand.
 *
 * @param expanded_sk [out] Expanded Sk value.
 * @param seed [in] Seed value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed);
/**
 * @brief Derive the Ed25519 public key from a 32-byte seed.
 *
 * @param seed [in] 32-byte seed (the Ed25519 secret seed).
 * @param public_key [out] 32-byte public key (little-endian Y with sign bit).
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_publickey_from_seed(const uint8_t* seed, uint8_t* public_key);
/**
 * @brief Sign a message with Ed25519.
 * @param uint8_t *sig
 * @param size_t *siglen
 * @param const uint8_t *m
 * @param size_t mlen
 * @param const uint8_t *sk
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_sign(const SNEPPXEd25519Keypair* kp, const uint8_t* message, size_t msg_len, SNEPPXEd25519Signature* sig);
/**
 * @brief Verify an Ed25519 signature.
 * @param const uint8_t *sig
 * @param size_t siglen
 * @param const uint8_t *m
 * @param size_t mlen
 * @param const uint8_t *pk
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const SNEPPXEd25519Signature* sig);
/**
 * @brief Verify Ed25519 with optional legacy-hash fallback for migration.
 * @param sig [in] 64-byte raw signature (R || S).
 * @param m [in] Message bytes.
 * @param mlen [in] Message length.
 * @param pk [in] 32-byte public key.
 * @param allow_legacy [in] If non-zero, retry with legacy (pre-fix) SHA-512 on failure.
 * @return 1 on valid, 0 on invalid, -1 on error.
 */
int SNEPPX_ed25519_verify_compat(const uint8_t* sig, const uint8_t* m, size_t mlen, const uint8_t* pk, int allow_legacy);
/**
 * @brief Perform Ed25519 Scalar Multiply.
 *
 * @param result [out] Result value.
 * @param scalar [in] Scalar value.
 * @param point [in] Point value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* point);


#ifdef __cplusplus
}
#endif
#endif
