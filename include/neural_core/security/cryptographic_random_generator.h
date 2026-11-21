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

#ifdef SNEPPX_ENABLE_TEST_RNG
/**
 * @brief Test-only deterministic RNG hooks.
 *
 * These declarations exist only when the library is explicitly compiled with
 * SNEPPX_ENABLE_TEST_RNG. Never enable this option in production builds.
 */
void SNEPPX_random_bytes_set_seed(const uint8_t seed[32]);
void SNEPPX_random_bytes_clear_seed(void);
#endif
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
