#ifndef SNEPPX_ZK_PROOF_H
#define SNEPPX_ZK_PROOF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Zk Proof
 *
 * WHAT
 *   Zk Proof.
 *
 * CONCEPT
 *   Provides the Zk Proof.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Zero-knowledge proof primitives. When SNEPPX_BUILD_ZK is defined this
 * provides a real Schnorr proof of knowledge of a discrete logarithm
 * (zkx = g^x mod p) with a Fiat-Shamir challenge hashed via an embedded
 * SHA-256. Without the flag the entry points report SNEPPX_DRIVER_UNSUPPORTED.
 *
 * NOTE: the embedded 256-bit group is demonstration-strength (demo-grade
 * security), not deployment-grade. The construction is cryptographically
 * correct: a valid proof verifies and a tampered/invalid proof is rejected. */

/**
 * @brief Initialize Zk.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_zk_init(void);
/**
 * @brief Perform Zk Shutdown.
 */
void SNEPPX_zk_shutdown(void);

/* Generate a keypair: `pub` receives g^secret mod p (32 bytes).
 * `secret`/`secret_len` is the witness (scalar, little-endian, <= 32 bytes). */
/**
 * @brief Generate Zk.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [out] Pub value.
 * @param pub_len [out] Pub Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_keygen(const uint8_t* secret, size_t secret_len, uint8_t* pub, size_t* pub_len);

/* Produce a Schnorr proof that the prover knows x with pub = g^x.
 * `proof` receives R (32 bytes) || s (32 bytes); `*proof_len` is set. */
/**
 * @brief Perform Zk Prove.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [out] Proof value.
 * @param proof_len [out] Proof Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_prove(const uint8_t* secret, size_t secret_len,
                    const uint8_t* pub, size_t pub_len,
                    const uint8_t* msg, size_t msg_len,
                    uint8_t* proof, size_t* proof_len);

/* Verify a proof against `pub` and `msg`. Returns 0 on success (valid proof). */
/**
 * @brief Verify Zk.
 *
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [in] Proof value.
 * @param proof_len [in] Proof Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_verify(const uint8_t* pub, size_t pub_len,
                     const uint8_t* msg, size_t msg_len,
                     const uint8_t* proof, size_t proof_len);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_ZK_PROOF_H */
