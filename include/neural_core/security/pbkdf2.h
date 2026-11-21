#ifndef SNEPPX_PBKDF2_H
#define SNEPPX_PBKDF2_H

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - PBKDF2 (Password-Based Key Derivation Function 2)
 *
 * WHAT
 *   PBKDF2 (Password-Based Key Derivation Function 2).
 *
 * CONCEPT
 *   RFC 8018 PBKDF2: iterated HMAC to stretch a password into a cryptographic key.
 *
 * ROLE
 *   Legacy key derivation in the key-vault; Argon2 is preferred for new code.
 *
 * REFERENCES
 *   RFC 8018 (PKCS #5), NIST SP 800-132.
 */



#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief PBKDF2-HMAC-SHA256 key derivation.
 * @param uint8_t *out
 * @param size_t outlen
 * @param const uint8_t *password
 * @param size_t passlen
 * @param const uint8_t *salt
 * @param size_t saltlen
 * @param uint32_t iterations
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pbkdf2_hmac_sha256(const uint8_t* password, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iterations, uint8_t* out, size_t out_len);
/**
 * @brief PBKDF2-HMAC-SHA256 key derivation.
 * @param uint8_t *out
 * @param size_t outlen
 * @param const uint8_t *password
 * @param size_t passlen
 * @param const uint8_t *salt
 * @param size_t saltlen
 * @param uint32_t iterations
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pbkdf2_hmac_sha512(const uint8_t* password, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iterations, uint8_t* out, size_t out_len);

#ifdef __cplusplus
}
#endif
#endif
