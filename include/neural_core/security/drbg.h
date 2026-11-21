#ifndef SNEPPX_DRBG_H
#define SNEPPX_DRBG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_DRBG_MAX_OUTPUT 65536
#define SNEPPX_DRBG_SEED_SIZE 48
/*
 * SNEPPX - Deterministic Random Bit Generator (DRBG)
 *
 * WHAT
 *   Deterministic Random Bit Generator (DRBG).
 *
 * CONCEPT
 *   NIST SP 800-90A compliant HMAC-DRBG and Hash-DRBG.
 *
 * ROLE
 *   Foundation for all random-number generation in the crypto module.
 *
 * REFERENCES
 *   NIST SP 800-90A (DRBG).
 */



typedef struct {
    uint8_t v[SNEPPX_DRBG_SEED_SIZE];
    uint8_t c[SNEPPX_DRBG_SEED_SIZE];
    uint64_t reseed_counter;
    int security_strength;
    int initialized;
} SNEPPXHashDRBG;

typedef struct {
    SNEPPXHashDRBG hb;
    int use_hmac;
} SNEPPXDRBG;

/**
 * @brief Initialize a DRBG context with entropy.
 * @param SNEPPXDRBG *ctx
 * @param const uint8_t *entropy
 * @param size_t entropy_len
 * @param const uint8_t *nonce
 * @param size_t nonce_len
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_drbg_init(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len, const uint8_t* nonce, size_t nonce_len);
/**
 * @brief Reseed DRBG with fresh entropy.
 * @param SNEPPXDRBG *ctx
 * @param const uint8_t *entropy
 * @param size_t entropy_len
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_drbg_reseed(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len);
/**
 * @brief Generate random bytes from DRBG.
 * @param SNEPPXDRBG *ctx
 * @param uint8_t *out
 * @param size_t outlen
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_drbg_generate(SNEPPXDRBG* ctx, uint8_t* out, size_t out_len);
/**
 * @brief Destroy DRBG context and wipe sensitive data.
 * @param SNEPPXDRBG *ctx
 * @return void.
 */
void SNEPPX_drbg_destroy(SNEPPXDRBG* ctx);
/**
 * @brief Perform Drbg Self Test.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_drbg_self_test(void);

#ifdef __cplusplus
}
#endif
#endif
