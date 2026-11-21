#ifndef SNEPPX_KYBER_H
#define SNEPPX_KYBER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define KYBER_PUBLICKEYBYTES 1184
#define KYBER_SECRETKEYBYTES 1632
#define KYBER_CIPHERTEXTBYTES 1088
#define KYBER_SSBYTES 32
#define KYBER_K 3

#define KYBER_VARIANT_512 2
#define KYBER_VARIANT_768 3
#define KYBER_VARIANT_1024 4
/*
 * SNEPPX - Kyber ML-KEM (Post-Quantum Key Encapsulation)
 *
 * WHAT
 *   Kyber ML-KEM (Post-Quantum Key Encapsulation).
 *
 * CONCEPT
 *   Module-lattice (ML-KEM) key generation, encapsulation, and decapsulation for FIPS 203 Kyber-512/768/1024.
 *
 * ROLE
 *   Layer S0 post-quantum crypto of the S0-S9 security stack.
 *
 * REFERENCES
 *   FIPS 203 (Kyber), NIST PQC Round 3 finalist.
 */



/**
 * @brief Generate Kyber.
 *
 * @param pk [out] Pk value.
 * @param sk [out] Sk value.
 * @param variant [in] Variant value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_keygen(uint8_t *pk, uint8_t *sk, int variant);
/**
 * @brief Encapsulate Kyber.
 *
 * @param ct [out] Ct value.
 * @param ss [out] Ss value.
 * @param pk [in] Pk value.
 * @param variant [in] Variant value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_encaps(uint8_t *ct, uint8_t *ss, const uint8_t *pk, int variant);
/**
 * @brief Decapsulate Kyber.
 *
 * @param ss [out] Ss value.
 * @param ct [in] Ct value.
 * @param sk [in] Sk value.
 * @param variant [in] Variant value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_decaps(uint8_t *ss, const uint8_t *ct, const uint8_t *sk, int variant);


#ifdef __cplusplus
}
#endif
#endif
