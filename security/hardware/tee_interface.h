#ifndef SNEPPX_HARDWARE_SECURITY_INTERFACE_H
#define SNEPPX_HARDWARE_SECURITY_INTERFACE_H

#include <stdint.h>

#define SNEPPX_TEE_MAX_SLOTS 16
#define SNEPPX_TEE_KEY_SIZE 32

/*
 * SNEPPX - Tee Interface
 *
 * WHAT
 *   Tee Interface.
 *
 * CONCEPT
 *   Provides the Tee Interface.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TEE_NONE = 0,
    SNEPPX_TEE_SGX,
    SNEPPX_TEE_SEV,
    SNEPPX_TEE_TDX,
    SNEPPX_TEE_PSP
} SNEPPXTeeType;

typedef struct {
    int initialized;
    SNEPPXTeeType tee_type;
    int attestation_verified;
    int secure_boot_enabled;
    int hsm_present;
} SNEPPXHardwareSecurity;

/**
 * @brief Perform Hsm Attest.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_attest(void);
/**
 * @brief Perform Hsm Seal Key.
 *
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 * @param sealed [out] Sealed value.
 * @param sealed_len [out] Sealed Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_seal_key(const uint8_t* key, size_t key_len, uint8_t* sealed, size_t* sealed_len);
/**
 * @brief Perform Hsm Unseal Key.
 *
 * @param sealed [in] Sealed value.
 * @param sealed_len [in] Sealed Len value.
 * @param key [out] Key value.
 * @param key_len [out] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_unseal_key(const uint8_t* sealed, size_t sealed_len, uint8_t* key, size_t* key_len);
/**
 * @brief Perform Hsm Verify Measurement.
 *
 * @param expected [in] Expected value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_verify_measurement(const uint8_t* expected, size_t len);
/**
 * @brief Perform Hsm Get Random.
 *
 * @param buf [out] Buf value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_get_random(uint8_t* buf, size_t len);

#endif
