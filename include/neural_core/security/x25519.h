#ifndef SNEPPX_X25519_H
#define SNEPPX_X25519_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_X25519_KEY_SIZE 32
#define SNEPPX_X25519_SHARED_SIZE 32
/*
 * SNEPPX - X25519 Key Exchange (ECDH over Curve25519)
 *
 * WHAT
 *   X25519 Key Exchange (ECDH over Curve25519).
 *
 * CONCEPT
 *   X25519 ECDH key exchange computing shared secrets from private/public key pairs.
 *
 * ROLE
 *   Used by the key-vault for establishing shared secrets with peers.
 *
 * REFERENCES
 *   RFC 7748 (X25519).
 */



/**
 * @brief Compute X25519 shared secret.
 * @param uint8_t *shared_secret
 * @param const uint8_t *private_key
 * @param const uint8_t *public_key
 * @return 0 on success, -1 on error.
 */
void SNEPPX_x25519_clamp(uint8_t scalar[SNEPPX_X25519_KEY_SIZE]);
/**
 * @brief Compute X25519 shared secret.
 * @param uint8_t *shared_secret
 * @param const uint8_t *private_key
 * @param const uint8_t *public_key
 * @return 0 on success, -1 on error.
 */
void SNEPPX_x25519_scalar_mult(uint8_t out[SNEPPX_X25519_KEY_SIZE], const uint8_t scalar[SNEPPX_X25519_KEY_SIZE], const uint8_t point[SNEPPX_X25519_KEY_SIZE]);
/**
 * @brief Compute X25519 shared secret.
 * @param uint8_t *shared_secret
 * @param const uint8_t *private_key
 * @param const uint8_t *public_key
 * @return 0 on success, -1 on error.
 */
void SNEPPX_x25519_keygen(uint8_t public_key[SNEPPX_X25519_KEY_SIZE], uint8_t secret_key[SNEPPX_X25519_KEY_SIZE]);
/**
 * @brief Compute X25519 shared secret.
 * @param uint8_t *shared_secret
 * @param const uint8_t *private_key
 * @param const uint8_t *public_key
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_x25519_shared_secret(uint8_t shared[SNEPPX_X25519_SHARED_SIZE], const uint8_t secret_key[SNEPPX_X25519_KEY_SIZE], const uint8_t public_key[SNEPPX_X25519_KEY_SIZE]);

/**
 * @brief Perform Curve25519 Basepoint.
 */
void SNEPPX_curve25519_basepoint(uint8_t out[SNEPPX_X25519_KEY_SIZE]);
/**
 * @brief Compute X25519 shared secret.
 * @param uint8_t *shared_secret
 * @param const uint8_t *private_key
 * @param const uint8_t *public_key
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_x25519_scalar_valid(const uint8_t scalar[SNEPPX_X25519_KEY_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
