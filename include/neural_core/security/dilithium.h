#ifndef SNEPPX_DILITHIUM_H
#define SNEPPX_DILITHIUM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define DILITHIUM_PUBLICKEYBYTES 1312
#define DILITHIUM_SECRETKEYBYTES 2560
#define DILITHIUM_SIGBYTES 3368
/*
 * SNEPPX - Dilithium ML-DSA (Post-Quantum Digital Signatures)
 *
 * WHAT
 *   Dilithium ML-DSA (Post-Quantum Digital Signatures).
 *
 * CONCEPT
 *   Module-lattice-based digital signature scheme for FIPS 204 quantum-safe signatures.
 *
 * ROLE
 *   Layer S0 post-quantum crypto of the S0-S9 security stack.
 *
 * REFERENCES
 *   FIPS 204 (Dilithium / ML-DSA), NIST PQC Round 3 finalist.
 */



/**
 * @brief Generate a Dilithium key pair.
 * @param uint8_t *pk
 * @param uint8_t *sk
 * @param int variant
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dilithium_keygen(uint8_t *pk, uint8_t *sk, int variant);
/**
 * @brief Sign a message with the secret key.
 * @param uint8_t *sig
 * @param size_t *siglen
 * @param const uint8_t *m
 * @param size_t mlen
 * @param const uint8_t *sk
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dilithium_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant);
/**
 * @brief Verify a signature with the public key.
 * @param const uint8_t *sig
 * @param size_t siglen
 * @param const uint8_t *m
 * @param size_t mlen
 * @param const uint8_t *pk
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dilithium_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant);


#ifdef __cplusplus
}
#endif
#endif
