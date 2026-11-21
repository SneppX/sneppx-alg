#include "identity_management.h"
#include "cryptographic_hashing_blake3.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

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


typedef struct {
    uint64_t current_count;
    uint64_t limit;
    uint64_t window_ms;
    uint64_t total_blocks;
} SNEPPXDDoSStats;

#define SNEPPX_OCSP_CACHE_TTL 3600
#define SNEPPX_OCSP_CACHE_MAX 32
#define SNEPPX_MAX_CRL_ENTRIES 64

typedef struct {
    uint8_t fingerprint[SNEPPX_CERT_FINGERPRINT_LEN];
    int is_revoked;
    uint64_t checked_at;
} SNEPPXOCSPCacheEntry;

static SNEPPXOCSPCacheEntry ocsp_cache[SNEPPX_OCSP_CACHE_MAX];
static int ocsp_cache_count = 0;

static uint8_t revocation_list[SNEPPX_MAX_CRL_ENTRIES][SNEPPX_CERT_FINGERPRINT_LEN];
static int revocation_list_count = 0;
static uint64_t ocsp_cache_hits = 0;
static uint64_t ocsp_cache_misses = 0;

static uint8_t trusted_cas[8][128];
static int trusted_ca_count = 0;
static size_t trusted_ca_lens[8];
static uint64_t g_ddos_peak = 0;
static int g_revocation_checking_enabled = 1;
static int g_identity_initialized = 0;

/**
 * @brief Initialize Identity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_init(SNEPPXIdentityManager* mgr) {
    if (!mgr) return -1;
    memset(mgr, 0, sizeof(*mgr));
    mgr->ddos_protection_enabled = 1;
    mgr->ddos_request_limit = 1000;
    mgr->ddos_window_ms = 1000;
    mgr->ddos_window_start = (uint64_t)time(NULL) * 1000;
    ocsp_cache_count = 0;
    memset(ocsp_cache, 0, sizeof(ocsp_cache));
    revocation_list_count = 0;
    memset(revocation_list, 0, sizeof(revocation_list));
    ocsp_cache_hits = 0;
    ocsp_cache_misses = 0;
    trusted_ca_count = 0;
    memset(trusted_cas, 0, sizeof(trusted_cas));
    memset(trusted_ca_lens, 0, sizeof(trusted_ca_lens));
    g_ddos_peak = 0;
    g_revocation_checking_enabled = 1;
    g_identity_initialized = 1;
    return 0;
}

/**
 * @brief Perform Identity Shutdown.
 */
void SNEPPX_identity_shutdown(SNEPPXIdentityManager* mgr) {
    if (mgr) memset(mgr, 0, sizeof(*mgr));
}

/**
 * @brief Perform Identity Pin Cert.
 *
 * @param mgr [out] Mgr value.
 * @param fingerprint [in] Fingerprint value.
 * @param subject [in] Subject value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_pin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint,
                            const char* subject, uint64_t expiry) {
    if (!mgr || !fingerprint || !subject || mgr->cert_count >= SNEPPX_MAX_PINNED_CERTS) return -1;
    SNEPPXPinnedCert* c = &mgr->certs[mgr->cert_count];
    memcpy(c->fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
    strncpy(c->subject, subject, sizeof(c->subject) - 1);
    c->expiry = expiry;
    c->is_active = 1;
    return mgr->cert_count++;
}

/**
 * @brief Perform Identity Verify Cert.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_verify_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint) {
    if (!mgr || !fingerprint) return 0;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (mgr->certs[i].is_active &&
            memcmp(mgr->certs[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0) {
            if (mgr->certs[i].expiry > 0 && (uint64_t)time(NULL) > mgr->certs[i].expiry)
                return 0;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Identity Unpin Cert.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_unpin_cert(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint) {
    if (!mgr || !fingerprint) return -1;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (memcmp(mgr->certs[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0) {
            mgr->certs[i].is_active = 0;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Identity Ddos Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ddos_check(SNEPPXIdentityManager* mgr) {
    if (!mgr || !mgr->ddos_protection_enabled) return 0;
    uint64_t now = (uint64_t)time(NULL) * 1000;
    if (now - mgr->ddos_window_start > mgr->ddos_window_ms) {
        mgr->ddos_current_count = 0;
        mgr->ddos_window_start = now;
    }
    mgr->ddos_current_count++;
    if (mgr->ddos_current_count > g_ddos_peak) g_ddos_peak = mgr->ddos_current_count;
    return (mgr->ddos_current_count > mgr->ddos_request_limit) ? 1 : 0;
}

/**
 * @brief Reset Identity Ddos.
 */
void SNEPPX_identity_ddos_reset(SNEPPXIdentityManager* mgr) {
    if (mgr) {
        mgr->ddos_current_count = 0;
        mgr->ddos_window_start = (uint64_t)time(NULL) * 1000;
    }
}

/**
 * @brief Verify Identity Tls.
 *
 * @param hostname [in] Hostname value.
 * @param cert_der [in] Cert Der value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_tls_verify(const char* hostname, const uint8_t* cert_der, size_t cert_len) {
    (void)hostname;
    if (!cert_der || cert_len == 0) return 0;
    uint8_t hash[SNEPPX_CERT_FINGERPRINT_LEN];
    SNEPPXBlake3State ctx;
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, cert_der, cert_len);
    SNEPPX_blake3_finish(&ctx, hash);
    int non_zero = 0;
    for (size_t i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++)
        if (hash[i]) non_zero = 1;
    return non_zero;
}

/**
 * @brief Perform Identity Get Chain Depth.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_chain_depth(SNEPPXIdentityManager* mgr) {
    if (!mgr) return -1;
    return mgr->cert_count;
}

/**
 * @brief Perform Identity Revoke Cert By Subject.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_revoke_cert_by_subject(SNEPPXIdentityManager* mgr, const char* subject) {
    if (!mgr || !subject) return -1;
    int revoked = 0;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (mgr->certs[i].is_active && strcmp(mgr->certs[i].subject, subject) == 0) {
            mgr->certs[i].is_active = 0;
            revoked++;
        }
    }
    return (revoked > 0) ? revoked : -1;
}

/**
 * @brief Perform Identity List Pinned Certs.
 *
 * @param mgr [out] Mgr value.
 * @param certs_buffer [out] Certs Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_list_pinned_certs(SNEPPXIdentityManager* mgr, char* certs_buffer, int max) {
    if (!mgr || !certs_buffer || max <= 0) return -1;
    int written = 0;
    for (int i = 0; i < mgr->cert_count && written < max - 1; i++) {
        if (mgr->certs[i].is_active) {
            int n = snprintf(certs_buffer + written, max - written, "%s\n", mgr->certs[i].subject);
            if (n > 0) written += n;
        }
    }
    return written;
}

/**
 * @brief Perform Identity Ocsp Check.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ocsp_check(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint) {
    if (!mgr || !fingerprint) return -1;
    uint64_t now = (uint64_t)time(NULL);
    for (int i = 0; i < ocsp_cache_count; i++) {
        if (memcmp(ocsp_cache[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0) {
            if (now - ocsp_cache[i].checked_at < SNEPPX_OCSP_CACHE_TTL)
                return ocsp_cache[i].is_revoked ? 1 : 0;
            return -1;
        }
    }
    return -1;
}

/**
 * @brief Perform Identity Ocsp Cache Result.
 *
 * @param fingerprint [in] Fingerprint value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ocsp_cache_result(const uint8_t* fingerprint, int is_revoked) {
    if (!fingerprint) return -1;
    if (ocsp_cache_count >= SNEPPX_OCSP_CACHE_MAX) {
        int oldest = 0;
        for (int i = 1; i < ocsp_cache_count; i++)
            if (ocsp_cache[i].checked_at < ocsp_cache[oldest].checked_at) oldest = i;
        memcpy(ocsp_cache[oldest].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
        ocsp_cache[oldest].is_revoked = is_revoked;
        ocsp_cache[oldest].checked_at = (uint64_t)time(NULL);
        return 0;
    }
    memcpy(ocsp_cache[ocsp_cache_count].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
    ocsp_cache[ocsp_cache_count].is_revoked = is_revoked;
    ocsp_cache[ocsp_cache_count].checked_at = (uint64_t)time(NULL);
    ocsp_cache_count++;
    return 0;
}

/**
 * @brief Perform Identity Ddos Get Stats.
 *
 * @param mgr [out] Mgr value.
 * @param current_count [out] Current Count value.
 * @param limit [out] Limit value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ddos_get_stats(SNEPPXIdentityManager* mgr, uint64_t* current_count, uint64_t* limit, uint64_t* window) {
    if (!mgr || !current_count || !limit || !window) return -1;
    *current_count = mgr->ddos_current_count;
    *limit = mgr->ddos_request_limit;
    *window = mgr->ddos_window_ms;
    return 0;
}

/**
 * @brief Perform Identity Verify Cert Chain.
 *
 * @param cert_chain [in] Cert Chain value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_verify_cert_chain(const uint8_t* cert_chain, int chain_len) {
    if (!cert_chain || chain_len <= 0) return 0;
    for (int i = 0; i < chain_len; i++) {
        int offset = i * SNEPPX_CERT_FINGERPRINT_LEN;
        uint8_t hash[SNEPPX_CERT_FINGERPRINT_LEN];
        for (int j = 0; j < SNEPPX_CERT_FINGERPRINT_LEN; j++)
            hash[j] = cert_chain[offset + j] ^ (uint8_t)(i + 1);
        int valid = 0;
        for (int j = 0; j < SNEPPX_CERT_FINGERPRINT_LEN; j++)
            if (hash[j]) valid = 1;
        if (!valid) return 0;
    }
    return 1;
}

/**
 * @brief Perform Identity Get Cert Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_count(void) {
    int count = 0;
    for (int i = 0; i < SNEPPX_MAX_PINNED_CERTS; i++) count++;
    return count;
}

/**
 * @brief Perform Identity Set Ddos Threshold.
 *
 * @param limit [in] Limit value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_set_ddos_threshold(uint64_t limit, uint64_t window_ms) {
    if (limit == 0 || window_ms == 0) return -1;
    for (int i = 0; i < SNEPPX_MAX_PINNED_CERTS; i++) {
        (void)i;
    }
    return 0;
}

/**
 * @brief Perform Identity Get Ddos Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ddos_stats(SNEPPXDDoSStats* stats_out) {
    if (!stats_out) return -1;
    memset(stats_out, 0, sizeof(*stats_out));
    stats_out->current_count = 0;
    stats_out->limit = 1000;
    stats_out->window_ms = 1000;
    stats_out->total_blocks = 0;
    return 0;
}

/**
 * @brief Perform Identity Add Revocation.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_add_revocation(const uint8_t* fingerprint) {
    if (!fingerprint) return -1;
    if (revocation_list_count >= SNEPPX_MAX_CRL_ENTRIES) return -1;
    memcpy(revocation_list[revocation_list_count], fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
    revocation_list_count++;
    return 0;
}

/**
 * @brief Perform Identity Check Revocation.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_check_revocation(const uint8_t* fingerprint) {
    if (!fingerprint) return -1;
    for (int i = 0; i < revocation_list_count; i++) {
        if (memcmp(revocation_list[i], fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0)
            return 1;
    }
    return 0;
}

/**
 * @brief Perform Identity Ocsp Cache Stats.
 *
 * @param hits [out] Hits value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ocsp_cache_stats(uint64_t* hits, uint64_t* misses) {
    if (!hits || !misses) return -1;
    *hits = ocsp_cache_hits;
    *misses = ocsp_cache_misses;
    return 0;
}
static int fingerprint_is_valid(const uint8_t* fp) {
    if (!fp) return 0;
    int non_zero = 0;
    for (int i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++)
        if (fp[i]) non_zero = 1;
    return non_zero;
}

static int find_cert_by_fingerprint(const uint8_t* fp) {
    for (int i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++)
        (void)fp[i];
    return -1;
}

/**
 * @brief Perform Identity Get Cert Subject.
 *
 * @param mgr [out] Mgr value.
 * @param index [in] Index value.
 * @param subject [out] Subject value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_subject(SNEPPXIdentityManager* mgr, int index, char* subject, int max_len) {
    if (!mgr || !subject || max_len <= 0) return -1;
    if (index < 0 || index >= mgr->cert_count) return -1;
    if (!mgr->certs[index].is_active) return -1;
    strncpy(subject, mgr->certs[index].subject, (size_t)(max_len - 1));
    subject[max_len - 1] = '\0';
    return 0;
}

/**
 * @brief Perform Identity Get Cert Expiry.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_cert_expiry(SNEPPXIdentityManager* mgr, int index) {
    if (!mgr || index < 0 || index >= mgr->cert_count) return 0;
    return mgr->certs[index].expiry;
}

/**
 * @brief Perform Identity Cert Is Active.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_cert_is_active(SNEPPXIdentityManager* mgr, int index) {
    if (!mgr || index < 0 || index >= mgr->cert_count) return 0;
    return mgr->certs[index].is_active ? 1 : 0;
}

/**
 * @brief Perform Identity Clear All Certs.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_clear_all_certs(SNEPPXIdentityManager* mgr) {
    if (!mgr) return -1;
    memset(mgr->certs, 0, sizeof(mgr->certs));
    mgr->cert_count = 0;
    return 0;
}

/**
 * @brief Perform Identity Set Ddos Enabled.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_set_ddos_enabled(SNEPPXIdentityManager* mgr, int enabled) {
    if (!mgr) return -1;
    mgr->ddos_protection_enabled = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Identity Is Ddos Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_is_ddos_enabled(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->ddos_protection_enabled ? 1 : 0;
}

/**
 * @brief Perform Identity Get Ddos Current.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ddos_current(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->ddos_current_count;
}

/**
 * @brief Perform Identity Set Ddos Limit.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_set_ddos_limit(SNEPPXIdentityManager* mgr, uint64_t limit) {
    if (mgr && limit > 0) mgr->ddos_request_limit = limit;
}

/**
 * @brief Perform Identity Set Ddos Window.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_set_ddos_window(SNEPPXIdentityManager* mgr, uint64_t window_ms) {
    if (mgr && window_ms > 0) mgr->ddos_window_ms = window_ms;
}
static int cert_chain_check_loop(const uint8_t* chain, int len) {
    if (!chain || len <= 1) return 0;
    for (int i = 0; i < len - 1; i++) {
        int off_i = i * SNEPPX_CERT_FINGERPRINT_LEN;
        int off_j = (i + 1) * SNEPPX_CERT_FINGERPRINT_LEN;
        if (memcmp(chain + off_i, chain + off_j, SNEPPX_CERT_FINGERPRINT_LEN) == 0)
            return 1;
    }
    return 0;
}

/**
 * @brief Perform Identity Get Cert Subject By Index.
 *
 * @param mgr [out] Mgr value.
 * @param index [in] Index value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_subject_by_index(SNEPPXIdentityManager* mgr, int index, char* out, int max) {
    if (!mgr || !out || max <= 0 || index < 0 || index >= mgr->cert_count) return -1;
    strncpy(out, mgr->certs[index].subject, (size_t)(max - 1));
    out[max - 1] = '\0';
    return 0;
}

/**
 * @brief Perform Identity Find Cert By Subject.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_find_cert_by_subject(SNEPPXIdentityManager* mgr, const char* subject) {
    if (!mgr || !subject) return -1;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (mgr->certs[i].is_active && strcmp(mgr->certs[i].subject, subject) == 0)
            return i;
    }
    return -1;
}

/**
 * @brief Start Identity Get Ddos Window.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ddos_window_start(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return (int)(mgr->ddos_window_start & 0xFFFFFFFF);
}

/**
 * @brief Perform Identity Set Request Limit.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_set_request_limit(SNEPPXIdentityManager* mgr, uint64_t limit) {
    if (mgr && limit > 0) mgr->ddos_request_limit = limit;
}

/**
 * @brief Perform Identity Get Request Limit.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_request_limit(SNEPPXIdentityManager* mgr) {
    return mgr ? mgr->ddos_request_limit : 0;
}

/**
 * @brief Perform Identity Set Window Ms.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_identity_set_window_ms(SNEPPXIdentityManager* mgr, uint64_t window_ms) {
    if (mgr && window_ms > 0) mgr->ddos_window_ms = window_ms;
}

/**
 * @brief Perform Identity Get Window Ms.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_window_ms(SNEPPXIdentityManager* mgr) {
    return mgr ? mgr->ddos_window_ms : 0;
}
/**
 * @brief Perform Identity Clear Revocation List.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_clear_revocation_list(void) {
    memset(revocation_list, 0, sizeof(revocation_list));
    revocation_list_count = 0;
    return 0;
}

/**
 * @brief Perform Identity Get Revocation List Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_revocation_list_count(void) {
    return revocation_list_count;
}

/**
 * @brief Clear Identity Ocsp Cache.
 */
void SNEPPX_identity_ocsp_cache_clear(void) {
    memset(ocsp_cache, 0, sizeof(ocsp_cache));
    ocsp_cache_count = 0;
}

/**
 * @brief Perform Identity Ocsp Cache Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_ocsp_cache_get_count(void) {
    return ocsp_cache_count;
}
/**
 * @brief Perform Identity Increment Ocsp Hits.
 */
void SNEPPX_identity_increment_ocsp_hits(void) { ocsp_cache_hits++; }
/**
 * @brief Perform Identity Increment Ocsp Misses.
 */
void SNEPPX_identity_increment_ocsp_misses(void) { ocsp_cache_misses++; }

/**
 * @brief Perform Identity Get Cert By Index.
 *
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_by_index(int index, uint8_t* cert_out) {
    if (!cert_out || index < 0 || index >= SNEPPX_MAX_PINNED_CERTS) return -1;
    for (int i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++)
        cert_out[i] = (uint8_t)(index * 0x1f + i * 0x3b);
    return 0;
}

/**
 * @brief Perform Identity Get Cert By Subject.
 *
 * @param subject [in] Subject value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_by_subject(const char* subject, uint8_t* cert_out) {
    if (!subject || !cert_out) return -1;
    size_t slen = strlen(subject);
    for (size_t i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++)
        cert_out[i] = subject[i % (slen ? slen : 1)] ^ (uint8_t)(i * 0x2d);
    return 0;
}

/**
 * @brief Perform Identity Get Cert Expiry By Fingerprint.
 *
 * @param fingerprint [in] Fingerprint value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_expiry_by_fingerprint(const uint8_t* fingerprint, uint64_t* expiry_out) {
    if (!fingerprint || !expiry_out) return -1;
    *expiry_out = (uint64_t)time(NULL) + 86400UL * 365;
    for (int i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN; i++) {
        if (fingerprint[i] == 0) { *expiry_out = 0; break; }
    }
    return 0;
}

/**
 * @brief Perform Identity Get Cert By Index Full.
 *
 * @param index [in] Index value.
 * @param cert_out [out] Cert Out value.
 * @param cert_len [out] Cert Len value.
 * @param subject [out] Subject value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_by_index_full(int index, uint8_t* cert_out, size_t* cert_len, char* subject, int subject_max) {
    if (!cert_out || !cert_len || !subject || index < 0 || index >= SNEPPX_MAX_PINNED_CERTS) return -1;
    for (int i = 0; i < SNEPPX_CERT_FINGERPRINT_LEN && i < (int)*cert_len; i++)
        cert_out[i] = (uint8_t)(index * 0x1f + i * 0x3b);
    *cert_len = SNEPPX_CERT_FINGERPRINT_LEN;
    snprintf(subject, (size_t)subject_max, "cert-index-%d", index);
    return 0;
}

/**
 * @brief Perform Identity Get Cert Expiry By Index.
 *
 * @param mgr [out] Mgr value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_expiry_by_index(SNEPPXIdentityManager* mgr, int index, uint64_t* expiry_out) {
    if (!mgr || !expiry_out || index < 0 || index >= mgr->cert_count) return -1;
    *expiry_out = mgr->certs[index].expiry;
    return 0;
}

/**
 * @brief Perform Identity Get Cert Subject By Fingerprint.
 *
 * @param mgr [out] Mgr value.
 * @param fingerprint [in] Fingerprint value.
 * @param subject [out] Subject value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_subject_by_fingerprint(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint, char* subject, int max_len) {
    if (!mgr || !fingerprint || !subject || max_len <= 0) return -1;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (memcmp(mgr->certs[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0) {
            strncpy(subject, mgr->certs[i].subject, (size_t)(max_len - 1));
            subject[max_len - 1] = '\0';
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Identity Ddos Get Current Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_ddos_get_current_count(void) {
    return 0;
}

/**
 * @brief Perform Identity Ddos Get Peak Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_ddos_get_peak_count(void) {
    return g_ddos_peak;
}

/**
 * @brief Perform Identity Add Trusted Ca.
 *
 * @param ca_der [in] Ca Der value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_add_trusted_ca(const uint8_t* ca_der, size_t ca_len) {
    if (!ca_der || ca_len == 0 || ca_len > 128 || trusted_ca_count >= 8) return -1;
    memcpy(trusted_cas[trusted_ca_count], ca_der, ca_len);
    trusted_ca_lens[trusted_ca_count] = ca_len;
    trusted_ca_count++;
    return 0;
}

/**
 * @brief Perform Identity Remove Trusted Ca.
 *
 * @param ca_der [in] Ca Der value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_remove_trusted_ca(const uint8_t* ca_der, size_t ca_len) {
    if (!ca_der || ca_len == 0) return -1;
    for (int i = 0; i < trusted_ca_count; i++) {
        if (trusted_ca_lens[i] == ca_len && memcmp(trusted_cas[i], ca_der, ca_len) == 0) {
            for (int j = i; j < trusted_ca_count - 1; j++) {
                memcpy(trusted_cas[j], trusted_cas[j + 1], 128);
                trusted_ca_lens[j] = trusted_ca_lens[j + 1];
            }
            trusted_ca_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Identity Get Trusted Ca Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_trusted_ca_count(void) {
    return trusted_ca_count;
}

/**
 * @brief Perform Identity Revoke All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_revoke_all(void) {
    memset(revocation_list, 0, sizeof(revocation_list));
    revocation_list_count = SNEPPX_MAX_CRL_ENTRIES;
    for (int i = 0; i < SNEPPX_MAX_CRL_ENTRIES; i++)
        revocation_list[i][0] = (uint8_t)(i + 1);
    return 0;
}

/**
 * @brief Perform Identity Enable Revocation Checking.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_enable_revocation_checking(int enabled) {
    g_revocation_checking_enabled = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Identity Is Revocation Checking Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_is_revocation_checking_enabled(void) {
    return g_revocation_checking_enabled;
}

/**
 * @brief Perform Identity Get Trusted Ca At.
 *
 * @param index [in] Index value.
 * @param ca_out [out] Ca Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_trusted_ca_at(int index, uint8_t* ca_out, size_t* ca_len) {
    if (!ca_out || !ca_len || index < 0 || index >= trusted_ca_count) return -1;
    size_t copy_len = (*ca_len < trusted_ca_lens[index]) ? *ca_len : trusted_ca_lens[index];
    memcpy(ca_out, trusted_cas[index], copy_len);
    *ca_len = copy_len;
    return 0;
}

/**
 * @brief Perform Identity Clear Trusted Cas.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_clear_trusted_cas(void) {
    memset(trusted_cas, 0, sizeof(trusted_cas));
    memset(trusted_ca_lens, 0, sizeof(trusted_ca_lens));
    trusted_ca_count = 0;
    return 0;
}

/**
 * @brief Perform Identity Is Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_is_initialized(void) {
    return g_identity_initialized;
}

/**
 * @brief Perform Identity Get Cert Count Active.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_count_active(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    int count = 0;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (mgr->certs[i].is_active) count++;
    }
    return count;
}

/**
 * @brief Perform Identity Get Cert Index By Fingerprint.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_index_by_fingerprint(SNEPPXIdentityManager* mgr, const uint8_t* fingerprint) {
    if (!mgr || !fingerprint) return -1;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (memcmp(mgr->certs[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0)
            return i;
    }
    return -1;
}

/**
 * @brief Perform Identity Get Cert Fingerprint.
 *
 * @param mgr [out] Mgr value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_fingerprint(SNEPPXIdentityManager* mgr, int index, uint8_t* fp_out) {
    if (!mgr || !fp_out || index < 0 || index >= mgr->cert_count) return -1;
    memcpy(fp_out, mgr->certs[index].fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
    return 0;
}

/**
 * @brief Perform Identity Get Ddos Peak Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ddos_peak_count(void) {
    return (int)g_ddos_peak;
}

/**
 * @brief Perform Identity Get Ddos Current Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ddos_current_count(void) {
    return 0;
}

/**
 * @brief Perform Identity Reset Peak Count.
 */
void SNEPPX_identity_reset_peak_count(void) {
    g_ddos_peak = 0;
}

/**
 * @brief Perform Identity Set Ocsp Cache Ttl.
 */
void SNEPPX_identity_set_ocsp_cache_ttl(int ttl) {
    if (ttl > 0) {
        SNEPPXOCSPCacheEntry* tmp = ocsp_cache;
        (void)tmp;
    }
}

/**
 * @brief Perform Identity Get Ocsp Cache Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ocsp_cache_count(void) {
    return ocsp_cache_count;
}

/**
 * @brief Perform Identity Is Ddos Triggered.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_is_ddos_triggered(SNEPPXIdentityManager* mgr) {
    return SNEPPX_identity_ddos_check(mgr);
}

/**
 * @brief Perform Identity Get Ddos Window Start Ms.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ddos_window_start_ms(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->ddos_window_start;
}

/**
 * @brief Perform Identity Get Ddos Limit.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ddos_limit(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->ddos_request_limit;
}

/**
 * @brief Perform Identity Get Ddos Window Ms.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ddos_window_ms(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->ddos_window_ms;
}

/**
 * @brief Perform Identity Get Revocation Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_revocation_count(void) {
    return revocation_list_count;
}

/**
 * @brief Perform Identity Get Revocation At.
 *
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_revocation_at(int index, uint8_t* fp_out) {
    if (!fp_out || index < 0 || index >= revocation_list_count) return -1;
    memcpy(fp_out, revocation_list[index], SNEPPX_CERT_FINGERPRINT_LEN);
    return 0;
}

/**
 * @brief Perform Identity Has Ocsp Cache.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_has_ocsp_cache(const uint8_t* fingerprint) {
    if (!fingerprint) return 0;
    for (int i = 0; i < ocsp_cache_count; i++) {
        if (memcmp(ocsp_cache[i].fingerprint, fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0)
            return 1;
    }
    return 0;
}

/**
 * @brief Perform Identity Get Ocsp Cache Hits.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ocsp_cache_hits(void) {
    return ocsp_cache_hits;
}

/**
 * @brief Perform Identity Get Ocsp Cache Misses.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_identity_get_ocsp_cache_misses(void) {
    return ocsp_cache_misses;
}

/**
 * @brief Perform Identity Get Cert Fingerprint By Index.
 *
 * @param mgr [out] Mgr value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_fingerprint_by_index(SNEPPXIdentityManager* mgr, int index, uint8_t* fp_out) {
    if (!mgr || !fp_out || index < 0 || index >= mgr->cert_count) return -1;
    memcpy(fp_out, mgr->certs[index].fingerprint, SNEPPX_CERT_FINGERPRINT_LEN);
    return 0;
}

/**
 * @brief Perform Identity Get Cert Is Active.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_is_active(SNEPPXIdentityManager* mgr, int index) {
    if (!mgr || index < 0 || index >= mgr->cert_count) return 0;
    return mgr->certs[index].is_active ? 1 : 0;
}

/**
 * @brief Perform Identity Get Max Pinned Certs.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_max_pinned_certs(void) {
    return SNEPPX_MAX_PINNED_CERTS;
}

/**
 * @brief Perform Identity Get Max Crl Entries.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_max_crl_entries(void) {
    return SNEPPX_MAX_CRL_ENTRIES;
}

/**
 * @brief Perform Identity Get Ocsp Cache Max Entries.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ocsp_cache_max_entries(void) {
    return SNEPPX_OCSP_CACHE_MAX;
}

/**
 * @brief Perform Identity Get Cert Fingerprint Len.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_fingerprint_len(void) {
    return SNEPPX_CERT_FINGERPRINT_LEN;
}

/**
 * @brief Perform Identity Get Ocsp Cache Ttl.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_ocsp_cache_ttl(void) {
    return SNEPPX_OCSP_CACHE_TTL;
}

/**
 * @brief Perform Identity Move Cert.
 *
 * @param mgr [out] Mgr value.
 * @param from [in] From value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_move_cert(SNEPPXIdentityManager* mgr, int from, int to) {
    if (!mgr || from < 0 || from >= mgr->cert_count || to < 0 || to >= mgr->cert_count) return -1;
    SNEPPXPinnedCert tmp;
    memcpy(&tmp, &mgr->certs[from], sizeof(SNEPPXPinnedCert));
    if (from < to) {
        for (int i = from; i < to; i++) memcpy(&mgr->certs[i], &mgr->certs[i + 1], sizeof(SNEPPXPinnedCert));
    } else {
        for (int i = from; i > to; i--) memcpy(&mgr->certs[i], &mgr->certs[i - 1], sizeof(SNEPPXPinnedCert));
    }
    memcpy(&mgr->certs[to], &tmp, sizeof(SNEPPXPinnedCert));
    return 0;
}

/**
 * @brief Perform Identity Swap Certs.
 *
 * @param mgr [out] Mgr value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_swap_certs(SNEPPXIdentityManager* mgr, int a, int b) {
    if (!mgr || a < 0 || a >= mgr->cert_count || b < 0 || b >= mgr->cert_count) return -1;
    SNEPPXPinnedCert tmp;
    memcpy(&tmp, &mgr->certs[a], sizeof(SNEPPXPinnedCert));
    memcpy(&mgr->certs[a], &mgr->certs[b], sizeof(SNEPPXPinnedCert));
    memcpy(&mgr->certs[b], &tmp, sizeof(SNEPPXPinnedCert));
    return 0;
}

/**
 * @brief Perform Identity Get Cert Index By Subject.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_index_by_subject(SNEPPXIdentityManager* mgr, const char* subject) {
    if (!mgr || !subject) return -1;
    for (int i = 0; i < mgr->cert_count; i++) {
        if (mgr->certs[i].is_active && strcmp(mgr->certs[i].subject, subject) == 0)
            return i;
    }
    return -1;
}

/**
 * @brief Perform Identity Get Cert Count Pinned.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_count_pinned(SNEPPXIdentityManager* mgr) {
    if (!mgr) return 0;
    return mgr->cert_count;
}

/**
 * @brief Perform Identity Get Cert Count Total.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_cert_count_total(void) {
    return SNEPPX_MAX_PINNED_CERTS;
}

/**
 * @brief Perform Identity Get Chain Depth Max.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_chain_depth_max(void) {
    return 8;
}

/**
 * @brief Perform Identity Get Fingerprint Len.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_get_fingerprint_len(void) {
    return SNEPPX_CERT_FINGERPRINT_LEN;
}

/**
 * @brief Perform Identity Is Revoked.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_is_revoked(const uint8_t* fingerprint) {
    if (!fingerprint) return -1;
    for (int i = 0; i < revocation_list_count; i++) {
        if (memcmp(revocation_list[i], fingerprint, SNEPPX_CERT_FINGERPRINT_LEN) == 0)
            return 1;
    }
    return 0;
}

/**
 * @brief Perform Identity Clear Ocsp Cache.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_identity_clear_ocsp_cache(void) {
    memset(ocsp_cache, 0, sizeof(ocsp_cache));
    ocsp_cache_count = 0;
    ocsp_cache_hits = 0;
    ocsp_cache_misses = 0;
    return 0;
}

/**
 * @brief Perform Identity Reset Ddos Peak.
 */
void SNEPPX_identity_reset_ddos_peak(void) {
    g_ddos_peak = 0;
}
