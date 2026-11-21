#ifndef SNEPPX_HKDF_H
#define SNEPPX_HKDF_H

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - HKDF (HMAC-based Key Derivation Function)
 *
 * WHAT
 *   HKDF (HMAC-based Key Derivation Function).
 *
 * CONCEPT
 *   RFC 5869 HKDF: extract a pseudorandom key, then expand into derived keys.
 *
 * ROLE
 *   Used by the key-vault for sub-key derivation and secure-transport session key derivation.
 *
 * REFERENCES
 *   RFC 5869 (HKDF).
 */



#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief HKDF extract phase.
 * @param uint8_t *prk
 * @param const uint8_t *salt
 * @param size_t saltlen
 * @param const uint8_t *ikm
 * @param size_t ikmlen
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf_extract(const uint8_t* salt, size_t salt_len, const uint8_t* ikm, size_t ikm_len, uint8_t* prk, size_t prk_len);
/**
 * @brief HKDF expand phase.
 * @param uint8_t *okm
 * @param size_t okmlen
 * @param const uint8_t *prk
 * @param const uint8_t *info
 * @param size_t infolen
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf_expand(const uint8_t* prk, size_t prk_len, const uint8_t* info, size_t info_len, uint8_t* okm, size_t okm_len);
/**
 * @brief Perform Hkdf.
 *
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param ikm [in] Ikm value.
 * @param ikm_len [in] Ikm Len value.
 * @param info [in] Info value.
 * @param info_len [in] Info Len value.
 * @param okm [out] Okm value.
 * @param okm_len [in] Okm Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf(const uint8_t* salt, size_t salt_len, const uint8_t* ikm, size_t ikm_len, const uint8_t* info, size_t info_len, uint8_t* okm, size_t okm_len);

#ifdef __cplusplus
}
#endif
#endif
