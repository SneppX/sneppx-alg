#ifndef SNEPPX_SIGNED_UPDATE_H
#define SNEPPX_SIGNED_UPDATE_H
/*
 * SNEPPX - Signed Update
 *
 * WHAT
 *   Signed Update.
 *
 * CONCEPT
 *   Provides the Signed Update.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S7 Secure Updates — Signed Update Verification
 * Cryptographic verification of signed delta updates with rollback protection.
 */

#include <stddef.h>
#include <stdint.h>
#include "ed25519_signature_verification.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_UPDATE_SIG_LEN 64
#define SNEPPX_UPDATE_HASH_LEN 32
#define SNEPPX_MAX_UPDATE_CHUNKS 256

typedef struct {
    uint32_t chunk_index;
    uint8_t chunk_hash[SNEPPX_UPDATE_HASH_LEN];
    uint32_t chunk_size;
} SNEPPXUpdateChunk;

typedef struct {
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_patch;
    uint8_t signature[SNEPPX_UPDATE_SIG_LEN];
    uint8_t update_hash[SNEPPX_UPDATE_HASH_LEN];
    SNEPPXUpdateChunk chunks[SNEPPX_MAX_UPDATE_CHUNKS];
    int chunk_count;
    uint64_t timestamp;
    int is_delta;
} SNEPPXSignedUpdate;

typedef struct {
    uint32_t current_version[3];
    uint32_t min_allowed_version[3];
    int rollback_protection_enabled;
    int verification_enabled;
    uint8_t trusted_pubkey[SNEPPX_ED25519_PUBLIC_KEY_LEN];
    int has_trusted_key;
} SNEPPXUpdateVerifier;

/**
 * @brief Initialize Update Verifier.
 *
 * @param uv [out] Uv value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_update_verifier_init(SNEPPXUpdateVerifier* uv);
/**
 * @brief Destroy Update Verifier.
 *
 * @param uv [out] Uv value.
 */
void SNEPPX_update_verifier_destroy(SNEPPXUpdateVerifier* uv);
/**
 * @brief Perform Update Verifier Set Min Version.
 *
 * @param uv [out] Uv value.
 * @param major [in] Major value.
 * @param minor [in] Minor value.
 * @param patch [in] Patch value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_update_verifier_set_min_version(SNEPPXUpdateVerifier* uv,
                                           uint32_t major, uint32_t minor, uint32_t patch);
/**
 * @brief Perform Update Verifier Check.
 *
 * @param uv [out] Uv value.
 * @param update [in] Update value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_update_verifier_check(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update);
/**
 * @brief Apply Update Verifier.
 *
 * @param uv [out] Uv value.
 * @param update [in] Update value.
 * @param update_data [in] Update Data value.
 * @param data_len [in] Data Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_update_verifier_apply(SNEPPXUpdateVerifier* uv, const SNEPPXSignedUpdate* update,
                                 const uint8_t* update_data, size_t data_len);
/**
 * @brief Perform Update Verifier Rollback Check.
 *
 * @param uv [out] Uv value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_update_verifier_rollback_check(SNEPPXUpdateVerifier* uv, uint32_t target_version[3]);

/* S7: verification API extensions */
int  SNEPPX_update_verifier_set_trusted_pubkey(SNEPPXUpdateVerifier* uv, const uint8_t* public_key);
int  SNEPPX_update_verifier_verify_signature(const SNEPPXSignedUpdate* update, const uint8_t* public_key, size_t key_len);
int  SNEPPX_update_verifier_sign_update(SNEPPXSignedUpdate* update, const uint8_t* signing_key, size_t key_len);

#ifdef __cplusplus
}
#endif
#endif
