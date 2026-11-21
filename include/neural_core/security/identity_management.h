#ifndef SNEPPX_IDENTITY_MANAGEMENT_H
#define SNEPPX_IDENTITY_MANAGEMENT_H
/*
 * SNEPPX - Identity Management
 *
 * WHAT
 *   Identity Management.
 *
 * CONCEPT
 *   Provides the Identity Management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S4 Network Security — Identity & Access Management
 * Certificate pinning, certificate validation, DDoS protection.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MAX_PINNED_CERTS 16
#define SNEPPX_CERT_FINGERPRINT_LEN 32

typedef struct {
    uint8_t fingerprint[SNEPPX_CERT_FINGERPRINT_LEN];
    char subject[256];
    uint64_t expiry;
    int is_active;
} SNEPPXPinnedCert;

typedef struct {
    SNEPPXPinnedCert certs[SNEPPX_MAX_PINNED_CERTS];
    int cert_count;
    int ddos_protection_enabled;
    uint64_t ddos_request_limit;
    uint64_t ddos_window_ms;
    uint64_t ddos_current_count;
    uint64_t ddos_window_start;
} SNEPPXIdentityManager;

/**
 * @brief Initialize Identity.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_init(SNEPPXIdentityManager* mgr);
/**
 * @brief Perform Identity Shutdown.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_shutdown(SNEPPXIdentityManager* mgr);
/**
 * @brief Perform Identity Pin Cert.
 *
 * @param mgr [out] Mgr value.
 * @param fingerprint [in] Fingerprint value.
 * @param subject [in] Subject value.
 * @param expiry [in] Expiry value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_pin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint,
                             const char* subject, uint64_t expiry);
/**
 * @brief Perform Identity Verify Cert.
 *
 * @param mgr [out] Mgr value.
 * @param fingerprint [in] Fingerprint value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_verify_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint);
/**
 * @brief Perform Identity Unpin Cert.
 *
 * @param mgr [out] Mgr value.
 * @param fingerprint [in] Fingerprint value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_unpin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint);
/**
 * @brief Perform Identity Ddos Check.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_ddos_check(SNEPPXIdentityManager* mgr);
/**
 * @brief Reset Identity Ddos.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_ddos_reset(SNEPPXIdentityManager* mgr);
/**
 * @brief Verify Identity Tls.
 *
 * @param hostname [in] Hostname value.
 * @param cert_der [in] Cert Der value.
 * @param cert_len [in] Cert Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_identity_tls_verify(const char* hostname, const uint8_t* cert_der, size_t cert_len);

#ifdef __cplusplus
}
#endif
#endif
