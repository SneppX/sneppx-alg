#ifndef SNEPPX_GDPR_FRAMEWORK_H
#define SNEPPX_GDPR_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define SNEPPX_GDPR_MAX_DATA_SUBJECTS 1000000
#define SNEPPX_GDPR_MAX_PROCESSING_RECORDS 1024
#define SNEPPX_GDPR_RIGHTS_ARTICLES 8

/*
 * SNEPPX - Gdpr Framework
 *
 * WHAT
 *   Gdpr Framework.
 *
 * CONCEPT
 *   Provides the Gdpr Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_GDPR_RIGHT_ACCESS,
    SNEPPX_GDPR_RIGHT_RECTIFICATION,
    SNEPPX_GDPR_RIGHT_ERASURE,
    SNEPPX_GDPR_RIGHT_RESTRICT,
    SNEPPX_GDPR_RIGHT_PORTABILITY,
    SNEPPX_GDPR_RIGHT_OBJECTION,
    SNEPPX_GDPR_RIGHT_AUTOMATED_DECISIONS,
    SNEPPX_GDPR_RIGHT_BREACH_NOTIFICATION
} SNEPPXGDPRDataSubjectRight;

typedef enum {
    SNEPPX_GDPR_CONSENT_EXPLICIT,
    SNEPPX_GDPR_CONSENT_IMPLIED,
    SNEPPX_GDPR_CONSENT_OPT_OUT,
    SNEPPX_GDPR_CONSENT_NONE
} SNEPPXGDPRConsentType;

typedef struct {
    char data_subject_id[128];
    char personal_data_category[64];
    char processing_purpose[256];
    SNEPPXGDPRConsentType consent;
    uint64_t consent_granted_at;
    uint64_t consent_expires_at;
    uint8_t consent_withdrawn : 1;
    uint8_t data_minimized : 1;
    uint8_t pseudonymized : 1;
    uint8_t encrypted_at_rest : 1;
    uint8_t encrypted_in_transit : 1;
    uint32_t retention_days;
    uint64_t created_at;
    uint64_t last_accessed;
    uint64_t deletion_scheduled;
    char data_controller[256];
    char data_processor[256];
    char third_party_transfers[512];
} SNEPPXGDPRDataRecord;

typedef struct {
    char record_id[64];
    char controller_name[256];
    char controller_representative[256];
    char dp_officer[256];
    char processing_activities[1024];
    uint32_t data_categories_count;
    char data_categories[8][64];
    uint32_t data_subject_categories_count;
    char data_subject_categories[8][64];
    uint8_t cross_border_transfer : 1;
    char third_countries[512];
    uint64_t expected_retention;
    char security_measures[1024];
} SNEPPXGDPRProcessingRecord;

typedef struct {
    uint64_t request_id;
    SNEPPXGDPRDataSubjectRight right_type;
    char data_subject_id[128];
    uint64_t request_received;
    uint64_t response_deadline;
    uint64_t responded_at;
    uint8_t fulfilled : 1;
    uint8_t extended_deadline : 1;
    uint8_t rejected : 1;
    char rejection_reason[512];
} SNEPPXGDPRSubjectRequest;

typedef struct {
    uint32_t breach_id;
    uint64_t discovered_at;
    uint64_t notified_supervisory_at;
    uint64_t notified_subjects_at;
    char breach_description[1024];
    char affected_data_categories[512];
    uint32_t affected_subjects_count;
    char likely_consequences[1024];
    char measures_taken[1024];
    uint8_t risk_to_rights : 1;
    uint8_t notified_supervisory : 1;
    uint8_t notified_subjects : 1;
    char supervisory_authority[256];
    uint32_t fine_amount;
} SNEPPXGDPRBreachNotification;

int snepx_gdpr_register_data_record(SNEPPXGDPRDataRecord* record);
int snepx_gdpr_find_data_subject(const char* subject_id, SNEPPXGDPRDataRecord* records, size_t* num_records);
int snepx_gdpr_consent_check(const SNEPPXGDPRDataRecord* record, const char* purpose);
int snepx_gdpr_consent_withdraw(SNEPPXGDPRDataRecord* record);
int snepx_gdpr_right_to_access(const char* subject_id, uint8_t* data_package, size_t* package_len);
int snepx_gdpr_right_to_erasure(const char* subject_id, uint8_t* deletion_log, size_t* log_len);
int snepx_gdpr_right_to_portability(const char* subject_id, uint8_t* portable_data, size_t* data_len);
int snepx_gdpr_right_to_restrict(SNEPPXGDPRDataRecord* record);
int snepx_gdpr_processing_record_add(const SNEPPXGDPRProcessingRecord* record);
int snepx_gdpr_processing_record_list(SNEPPXGDPRProcessingRecord* records_out, size_t* num_records);
int snepx_gdpr_dpia_create(const char* description, const char* necessity, const char* risks, const char* measures);
int snepx_gdpr_breach_report(SNEPPXGDPRBreachNotification* breach);
int snepx_gdpr_breach_notify_supervisory(SNEPPXGDPRBreachNotification* breach, uint32_t hours_delayed);
int snepx_gdpr_breach_notify_subjects(SNEPPXGDPRBreachNotification* breach);
int snepx_gdpr_ropa_export(uint8_t* ropa_xml, size_t* xml_len);
int snepx_gdpr_data_mapping(const char* system_boundary, uint8_t* map_out, size_t* map_len);

#endif
