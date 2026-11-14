#ifndef SNEPPX_SELF_AUDIT_H
#define SNEPPX_SELF_AUDIT_H
/*
 * SNEPPX - Self Audit
 *
 * WHAT
 *   Self Audit.
 *
 * CONCEPT
 *   Provides audit logging.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S9 Penetration Testing — Self-Audit Framework
 * Automated security assessment, vulnerability scanning, and
 * compliance verification.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_AUDIT_MAX_CHECKS 128
#define SNEPPX_CHECK_DESC_LEN 256

typedef enum {
    SNEPPX_AUDIT_PASS = 0,
    SNEPPX_AUDIT_FAIL = 1,
    SNEPPX_AUDIT_WARN = 2,
    SNEPPX_AUDIT_INFO = 3,
} SNEPPXAuditStatus;

typedef struct {
    char check_name[SNEPPX_CHECK_DESC_LEN];
    SNEPPXAuditStatus status;
    char details[SNEPPX_CHECK_DESC_LEN];
} SNEPPXAuditCheck;

typedef struct {
    SNEPPXAuditCheck checks[SNEPPX_AUDIT_MAX_CHECKS];
    int check_count;
    int total_passed;
    int total_failed;
    int total_warnings;
    double security_score;
} SNEPPXSelfAudit;

/**
 * @brief Initialize Self Audit.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_init(SNEPPXSelfAudit* audit);
/**
 * @brief Destroy Self Audit.
 *
 * @param audit [out] Audit value.
 */
void SNEPPX_self_audit_destroy(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Add Check.
 *
 * @param audit [out] Audit value.
 * @param name [in] Name value.
 * @param status [in] Status value.
 * @param details [in] Details value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_add_check(SNEPPXSelfAudit* audit, const char* name,
                                 SNEPPXAuditStatus status, const char* details);
/**
 * Add a check with an explicit category. The category is one of the
 * canonical self-audit category identifiers (crypto, memory, ... pentest).
 */
int  SNEPPX_self_audit_add_check_ex(SNEPPXSelfAudit* audit, const char* name,
                                   SNEPPXAuditStatus status, const char* details,
                                   const char* category);
/** Return the number of checks recorded for a canonical category, or -1. */
int  SNEPPX_self_audit_get_category_count(SNEPPXSelfAudit* audit, const char* category);
/** Return the percentage of PASS checks in a category, or -1 for invalid input. */
int  SNEPPX_self_audit_get_category_score(SNEPPXSelfAudit* audit, const char* category);
/** Format the current audit summary into the caller-provided buffer. */
int  SNEPPX_self_audit_get_summary(SNEPPXSelfAudit* audit, char* summary_out, size_t size);
/**
 * @brief Perform Self Audit Run All.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_run_all(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Run Category.
 *
 * @param audit [out] Audit value.
 * @param category [in] Category value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_run_category(SNEPPXSelfAudit* audit, const char* category);
/**
 * @brief Perform Self Audit Score.
 *
 * @param audit [out] Audit value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_self_audit_score(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Export Report.
 *
 * @param audit [out] Audit value.
 * @param output_path [in] Output Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_export_report(SNEPPXSelfAudit* audit, const char* output_path);
/**
 * @brief Perform Self Audit Check Crypto.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_check_crypto(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Check Memory.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_check_memory(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Check Network.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_check_network(SNEPPXSelfAudit* audit);
/**
 * @brief Perform Self Audit Check Ai Safety.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_self_audit_check_ai_safety(SNEPPXSelfAudit* audit);

#ifdef __cplusplus
}
#endif
#endif
