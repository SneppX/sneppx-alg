#ifndef SNEPPX_HIPAA_FRAMEWORK_H
#define SNEPPX_HIPAA_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define SNEPPX_HIPAA_MAX_PHI_FIELDS 128
#define SNEPPX_HIPAA_SAFEGUARDS_ADMIN 3
#define SNEPPX_HIPAA_SAFEGUARDS_PHYSICAL 3
#define SNEPPX_HIPAA_SAFEGUARDS_TECHNICAL 5

/*
 * SNEPPX - Hipaa Framework
 *
 * WHAT
 *   Hipaa Framework.
 *
 * CONCEPT
 *   Provides the Hipaa Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_PHI_TYPE_DEMOGRAPHIC,
    SNEPPX_PHI_TYPE_MEDICAL_RECORD,
    SNEPPX_PHI_TYPE_LAB_RESULT,
    SNEPPX_PHI_TYPE_PRESCRIPTION,
    SNEPPX_PHI_TYPE_INSURANCE,
    SNEPPX_PHI_TYPE_BILLING,
    SNEPPX_PHI_TYPE_CLINICAL_NOTES,
    SNEPPX_PHI_TYPE_GENETIC
} SNEPPXPHIType;

typedef struct {
    char phi_field_name[64];
    SNEPPXPHIType phi_type;
    uint8_t de_identified : 1;
    uint8_t masked : 1;
    uint8_t encrypted : 1;
    uint8_t access_logged : 1;
    uint8_t minimum_necessary : 1;
    uint64_t last_accessed;
    uint32_t access_count;
    char authorized_purpose[256];
} SNEPPXPHIField;

typedef struct {
    char patient_id[64];
    char provider_id[64];
    SNEPPXPHIField phi_fields[SNEPPX_HIPAA_MAX_PHI_FIELDS];
    uint32_t num_phi_fields;
    uint8_t* ephi_data;
    size_t ephi_data_len;
    uint64_t consent_expires;
    uint8_t consent_authorized : 1;
    uint8_t breach_affected : 1;
    uint64_t breach_notification_sent;
    uint8_t accounting_disclosures : 1;
} SNEPPXPHIRecord;

typedef struct {
    uint8_t access_control_implemented : 1;
    uint8_t audit_controls_implemented : 1;
    uint8_t integrity_controls_implemented : 1;
    uint8_t person_auth_implemented : 1;
    uint8_t transmission_security_implemented : 1;
    uint8_t encryption_at_rest : 1;
    uint8_t encryption_in_transit : 1;
    uint8_t automatic_logoff : 1;
    uint32_t automatic_logoff_minutes;
    uint8_t unique_user_ids : 1;
    uint8_t emergency_access : 1;
    uint8_t automatic_logoff_enforced : 1;
} SNEPPXHIPAATechnicalSafeguards;

typedef struct {
    uint8_t security_officer_assigned : 1;
    uint8_t workforce_trained : 1;
    uint8_t contingency_plan : 1;
    uint8_t facility_access_controls : 1;
    uint8_t workstation_security : 1;
    uint8_t device_media_controls : 1;
    uint8_t risk_analysis_completed : 1;
    uint8_t risk_management_program : 1;
    uint8_t sanction_policy : 1;
    uint8_t information_activity_review : 1;
    uint64_t last_risk_assessment;
    char contingency_plan_path[256];
} SNEPPXHIPAAAdminSafeguards;

typedef struct {
    uint32_t breach_id;
    char patient_ids[1024];
    uint32_t affected_count;
    char breach_type[64];
    uint64_t discovered;
    uint64_t reported_to_hhs;
    uint64_t reported_to_patients;
    uint8_t* breach_description;
    size_t description_len;
    char remediation[1024];
    uint32_t hhs_fine;
    uint8_t reported : 1;
    uint8_t resolved : 1;
} SNEPPXHIPAABreach;

int snepx_hipaa_phi_register(SNEPPXPHIRecord* record, const char* patient_id);
int snepx_hipaa_phi_access_log(SNEPPXPHIRecord* record, const char* user_id, const char* purpose);
int snepx_hipaa_phi_mask(SNEPPXPHIRecord* record, const char* field_name);
int snepx_hipaa_phi_deidentify(SNEPPXPHIRecord* record, uint8_t* deidentified_out, size_t* deidentified_len);
int snepx_hipaa_phi_encrypt(SNEPPXPHIRecord* record);
int snepx_hipaa_phi_decrypt(SNEPPXPHIRecord* record, const uint8_t* key, size_t key_len);
int snepx_hipaa_phi_minimize(SNEPPXPHIRecord* record);

int snepx_hipaa_safeguards_technical_evaluate(SNEPPXHIPAATechnicalSafeguards* safeguards);
int snepx_hipaa_safeguards_admin_evaluate(SNEPPXHIPAAAdminSafeguards* safeguards);
int snepx_hipaa_safeguards_physical_evaluate(uint8_t* report, size_t* report_len);

int snepx_hipaa_breach_report(SNEPPXHIPAABreach* breach);
int snepx_hipaa_breach_notify(SNEPPXHIPAABreach* breach, uint32_t days_delayed);
int snepx_hipaa_breach_risk_assessment(SNEPPXHIPAABreach* breach, double* risk_score);

int snepx_hipaa_baa_create(const char* business_associate, const char* covered_entity, uint8_t* baa_out, size_t* baa_len);
int snepx_hipaa_baa_verify(const uint8_t* baa, size_t baa_len);
int snepx_hipaa_audit_trail_export(uint64_t start_date, uint64_t end_date, uint8_t* audit_out, size_t* audit_len);

#endif
