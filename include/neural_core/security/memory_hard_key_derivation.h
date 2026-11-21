#ifndef SNEPPX_ARGON2_H
#define SNEPPX_ARGON2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Memory Hard Key Derivation
 *
 * WHAT
 *   Memory Hard Key Derivation.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    size_t memory_kb;
    size_t iterations;
    size_t parallelism;
    size_t hash_len;
} SNEPPXArgon2Config;

#define SNEPPX_ARGON2_DEFAULT_MEMORY 65536
#define SNEPPX_ARGON2_DEFAULT_ITERATIONS 3
#define SNEPPX_ARGON2_DEFAULT_PARALLELISM 4
#define SNEPPX_ARGON2_DEFAULT_HASH_LEN 32

/**
 * @brief Perform Argon2id.
 *
 * @param password [in] Password value.
 * @param password_len [in] Password Len value.
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param config [in] Config value.
 * @param hash [out] Hash value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_argon2id(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                  const SNEPPXArgon2Config* config, uint8_t* hash);
/**
 * @brief Verify Argon2id.
 *
 * @param password [in] Password value.
 * @param password_len [in] Password Len value.
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param config [in] Config value.
 * @param expected_hash [in] Expected Hash value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_argon2id_verify(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                         const SNEPPXArgon2Config* config, const uint8_t* expected_hash);


#ifdef __cplusplus
}
#endif
#endif
