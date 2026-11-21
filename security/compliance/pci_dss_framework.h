#ifndef SNEPPX_PCI_DSS_FRAMEWORK_H
#define SNEPPX_PCI_DSS_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define SNEPPX_PCI_DSS_REQUIREMENTS 12
#define SNEPPX_PCI_DSS_SUB_REQUIREMENTS 78
#define SNEPPX_PCI_DSS_MAX_CARDHOLDER_RECORDS 65536

/*
 * SNEPPX - Pci Dss Framework
 *
 * WHAT
 *   Pci Dss Framework.
 *
 * CONCEPT
 *   Provides the Pci Dss Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_PCI_DSS_REQ_BUILD_MAINTAIN_NETWORK,
    SNEPPX_PCI_DSS_REQ_PROTECT_CARDHOLDER_DATA,
    SNEPPX_PCI_DSS_REQ_MAINTAIN_VULNERABILITY,
    SNEPPX_PCI_DSS_REQ_ACCESS_CONTROL,
    SNEPPX_PCI_DSS_REQ_MONITOR_TEST,
    SNEPPX_PCI_DSS_REQ_MAINTAIN_POLICY
} SNEPPXPCIDSSRequirement;

typedef enum {
    SNEPPX_PCI_STATUS_NOT_APPLICABLE,
    SNEPPX_PCI_STATUS_IN_PLACE,
    SNEPPX_PCI_STATUS_PARTIALLY_IN_PLACE,
    SNEPPX_PCI_STATUS_NOT_IN_PLACE,
    SNEPPX_PCI_STATUS_REMEDIATION_SCHEDULED
} SNEPPXPCIStatus;

typedef struct {
    uint32_t req_number;
    uint32_t sub_req_number;
    char requirement_text[512];
    char testing_procedure[512];
    SNEPPXPCIStatus status;
    char evidence[256];
    char remediation_plan[512];
    uint64_t remediation_deadline;
    uint8_t* testing_results;
    size_t testing_results_len;
    uint8_t automated_test : 1;
    uint8_t manual_test : 1;
} SNEPPXPCIDSSRequirementItem;

typedef struct {
    char pan[19];
    char cardholder_name[64];
    char expiration[8];
    uint8_t svc_code[4];
    uint8_t pan_truncated : 1;
    uint8_t pan_masked : 1;
    uint8_t pan_hashed : 1;
    uint8_t pan_encrypted : 1;
    uint8_t pan_tokenized : 1;
    uint8_t* encrypted_pan;
    size_t encrypted_pan_len;
    uint8_t* token;
    size_t token_len;
    uint64_t stored_at;
    uint64_t accessed_at;
    uint32_t access_count;
    uint8_t* access_log;
    size_t access_log_len;
} SNEPPXPCICardholderRecord;

typedef struct {
    uint32_t num_records;
    SNEPPXPCICardholderRecord records[SNEPPX_PCI_DSS_MAX_CARDHOLDER_RECORDS];
    uint32_t num_stored;
    uint32_t num_tokenized;
    uint32_t num_encrypted;
    uint64_t total_storage_bytes;
    uint32_t num_access_events_30d;
} SNEPPXPCICardholderDataEnv;

typedef struct {
    uint32_t num_requirements;
    SNEPPXPCIDSSRequirementItem requirements[SNEPPX_PCI_DSS_SUB_REQUIREMENTS];
    uint64_t assessment_date;
    char assessor_company[256];
    char assessor_name[128];
    uint8_t qualified_security_assessor : 1;
    uint8_t internal_scan : 1;
    uint8_t external_scan : 1;
    uint8_t asv_scan : 1;
    uint32_t total_passed;
    uint32_t total_failed;
    uint32_t total_na;
    uint32_t remediations_open;
    char saq_type[16];
    uint8_t* asv_scan_results;
    size_t scan_results_len;
} SNEPPXPCIDSSAssessment;

int snepx_pci_dss_assessment_init(SNEPPXPCIDSSAssessment* assessment, const char* saq_type);
int snepx_pci_dss_add_requirement(SNEPPXPCIDSSAssessment* assessment, const SNEPPXPCIDSSRequirementItem* req);
int snepx_pci_dss_update_status(SNEPPXPCIDSSAssessment* assessment, uint32_t req_number, uint32_t sub_req, SNEPPXPCIStatus status);
int snepx_pci_dss_generate_asc(SNEPPXPCIDSSAssessment* assessment, uint8_t* asc_out, size_t* asc_len);
int snepx_pci_dss_generate_roc(SNEPPXPCIDSSAssessment* assessment, uint8_t* roc_out, size_t* roc_len);
int snepx_pci_dss_summary(SNEPPXPCIDSSAssessment* assessment, char* summary_out, size_t summary_max);

int snepx_pci_dss_cardholder_store(SNEPPXPCICardholderDataEnv* env, const SNEPPXPCICardholderRecord* record);
int snepx_pci_dss_cardholder_retrieve(SNEPPXPCICardholderDataEnv* env, const char* pan, SNEPPXPCICardholderRecord* record);
int snepx_pci_dss_cardholder_delete(SNEPPXPCICardholderDataEnv* env, const char* pan);
int snepx_pci_dss_cardholder_purge_expired(SNEPPXPCICardholderDataEnv* env);
int snepx_pci_dss_cardholder_mask(SNEPPXPCICardholderDataEnv* env);
int snepx_pci_dss_cardholder_encrypt(SNEPPXPCICardholderDataEnv* env);
int snepx_pci_dss_cardholder_tokenize(SNEPPXPCICardholderDataEnv* env);

int snepx_pci_dss_scan_external(const char* target, uint8_t* scan_report, size_t* report_len);
int snepx_pci_dss_scan_internal(const char* range, uint8_t* scan_report, size_t* report_len);
int snepx_pci_dss_scan_asv(const char* asv_service, uint8_t* scan_report, size_t* report_len);

#endif
