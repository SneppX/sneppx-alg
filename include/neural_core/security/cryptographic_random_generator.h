#ifndef SNEPPX_RANDOM_H
#define SNEPPX_RANDOM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Cryptographic Random Generator
 *
 * WHAT
 *   Cryptographic Random Generator.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Random Bytes.
 *
 * @param buffer [out] Buffer value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_random_bytes(uint8_t* buffer, size_t len);

/**
 * @brief Seed the test-only deterministic RNG.
 *
 * When seeded, all SNEPPX_random_bytes output becomes reproducible from the
 * given 32-byte seed (SHA-512 hash-chain). Intended for known-answer tests;
 * production code must NOT call this. Call SNEPPX_random_bytes_clear_seed to
 * restore OS/CSPRNG randomness.
 *
 * @param seed [in] 32-byte seed value.
 */
void SNEPPX_random_bytes_set_seed(const uint8_t seed[32]);

/**
 * @brief Disable the test-only deterministic RNG and restore OS randomness.
 */
void SNEPPX_random_bytes_clear_seed(void);
/**
 * @brief Generate a random uint32_t with explicit error reporting.
 *
 * @param value [out] Receives the random value on success.
 * @return 0 on success, -1 on invalid arguments or entropy failure.
 */
int SNEPPX_random_uint32_checked(uint32_t* value);

/**
 * @brief Generate a bounded uniform integer with explicit error reporting.
 *
 * @param upper_bound [in] Exclusive upper bound; must be non-zero.
 * @param value [out] Receives a value in [0, upper_bound) on success.
 * @return 0 on success, -1 on invalid arguments or entropy failure.
 */
int SNEPPX_random_uniform_checked(uint32_t upper_bound, uint32_t* value);

/**
 * @brief Compatibility wrapper; returns zero if entropy retrieval fails.
 */
uint32_t SNEPPX_random_uint32(void);

/**
 * @brief Compatibility wrapper; returns zero for invalid bounds or entropy failure.
 */
uint32_t SNEPPX_random_uniform(uint32_t upper_bound);


#ifdef __cplusplus
}
#endif
#endif
