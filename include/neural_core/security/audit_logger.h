#ifndef SNEPPX_AUDIT_LOGGER_H
#define SNEPPX_AUDIT_LOGGER_H
/*
 * SNEPPX - Audit Logger
 *
 * WHAT
 *   Audit Logger.
 *
 * CONCEPT
 *   Provides structured logging.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S6 Security UI — Audit Logger
 * Tamper-evident audit log for security events, key access, and violations.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AUDIT_MAX_ENTRIES 1024
#define SNEPPX_AUDIT_DESC_LEN 256

typedef struct {
    uint64_t timestamp;
    int event_type;
    char description[SNEPPX_AUDIT_DESC_LEN];
    uint64_t related_address;
    uint32_t crc;
} SNEPPXAuditEntry;

typedef struct {
    SNEPPXAuditEntry entries[SNEPPX_AUDIT_MAX_ENTRIES];
    int entry_count;
    int enabled;
    const char* log_file_path;
    uint32_t chain_crc;
} SNEPPXAuditLogger;

/**
 * @brief Initialize Audit.
 *
 * @param audit [out] Audit value.
 * @param log_path [in] Log Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_audit_init(SNEPPXAuditLogger* audit, const char* log_path);
/**
 * @brief Perform Audit Shutdown.
 *
 * @param audit [out] Audit value.
 */
void SNEPPX_audit_shutdown(SNEPPXAuditLogger* audit);
/**
 * @brief Perform Audit Log.
 *
 * @param audit [out] Audit value.
 * @param event_type [in] Event Type value.
 * @param description [in] Description value.
 * @param related_address [in] Related Address value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_audit_log(SNEPPXAuditLogger* audit, int event_type,
                     const char* description, uint64_t related_address);
/**
 * @brief Perform Audit Export.
 *
 * @param audit [out] Audit value.
 * @param output_path [in] Output Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_audit_export(SNEPPXAuditLogger* audit, const char* output_path);
/**
 * @brief Perform Audit Verify Chain.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_audit_verify_chain(SNEPPXAuditLogger* audit);
/**
 * @brief Perform Audit Search.
 *
 * @param audit [out] Audit value.
 * @param event_type [in] Event Type value.
 * @param results [out] Results value.
 * @param max_results [in] Max Results value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_audit_search(SNEPPXAuditLogger* audit, int event_type,
                        SNEPPXAuditEntry* results, int max_results);

#ifdef __cplusplus
}
#endif
#endif
