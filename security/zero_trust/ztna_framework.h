#ifndef SNEPPX_ZTNA_FRAMEWORK_H
#define SNEPPX_ZTNA_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_ZT_MAX_TRUST_SCORE 100
#define SNEPPX_ZT_MIN_TRUST_SCORE 0
#define SNEPPX_ZT_POLICY_MAX_CONDITIONS 64
#define SNEPPX_ZT_SESSION_TIMEOUT_NS 3600000000000ULL
#define SNEPPX_ZT_REAUTH_INTERVAL_NS 1800000000000ULL

/*
 * SNEPPX - Ztna Framework
 *
 * WHAT
 *   Ztna Framework.
 *
 * CONCEPT
 *   Provides the Ztna Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_ZT_ACCESS_DENY,
    SNEPPX_ZT_ACCESS_ALLOW,
    SNEPPX_ZT_ACCESS_ALLOW_WITH_MFA,
    SNEPPX_ZT_ACCESS_ALLOW_WITH_JUSTIFICATION,
    SNEPPX_ZT_ACCESS_ALLOW_TEMPORARY,
    SNEPPX_ZT_ACCESS_ALLOW_RESTRICTED
} SNEPPXZTDecision;

typedef enum {
    SNEPPX_ZT_CONDITION_USER_ROLE,
    SNEPPX_ZT_CONDITION_USER_CLEARANCE,
    SNEPPX_ZT_CONDITION_DEVICE_POSTURE,
    SNEPPX_ZT_CONDITION_DEVICE_COMPLIANCE,
    SNEPPX_ZT_CONDITION_NETWORK_LOCATION,
    SNEPPX_ZT_CONDITION_GEO_LOCATION,
    SNEPPX_ZT_CONDITION_TIME_OF_DAY,
    SNEPPX_ZT_CONDITION_RISK_SCORE,
    SNEPPX_ZT_CONDITION_BEHAVIOR_ANOMALY,
    SNEPPX_ZT_CONDITION_THREAT_INTEL,
    SNEPPX_ZT_CONDITION_SENSITIVITY_LABEL,
    SNEPPX_ZT_CONDITION_AUTH_STRENGTH,
    SNEPPX_ZT_CONDITION_SESSION_RISK,
    SNEPPX_ZT_CONDITION_NETWORK_TELEMETRY
} SNEPPXZTConditionType;

typedef enum {
    SNEPPX_ZT_DEVICE_POSTURE_UNKNOWN,
    SNEPPX_ZT_DEVICE_POSTURE_NON_COMPLIANT,
    SNEPPX_ZT_DEVICE_POSTURE_COMPLIANT,
    SNEPPX_ZT_DEVICE_POSTURE_HARDENED,
    SNEPPX_ZT_DEVICE_POSTURE_COMPROMISED
} SNEPPXZTDevicePosture;

typedef struct {
    uint8_t* subject_id;
    size_t subject_id_len;
    uint8_t* subject_roles[16];
    size_t role_lens[16];
    uint32_t num_roles;
    uint8_t* device_id;
    size_t device_id_len;
    SNEPPXZTDevicePosture device_posture;
    uint8_t* ip_address;
    size_t ip_len;
    float geo_latitude;
    float geo_longitude;
    uint8_t* country_code;
    size_t country_code_len;
    float behavioral_risk;
    float auth_strength;
    uint8_t mfa_verified : 1;
    uint8_t* session_id;
    size_t session_id_len;
    uint64_t session_start_ns;
    uint64_t last_reauth_ns;
} SNEPPXZTSubject;

typedef struct {
    uint8_t* resource_id;
    size_t resource_id_len;
    uint8_t* resource_type;
    size_t resource_type_len;
    uint8_t* sensitivity_label;
    size_t sensitivity_label_len;
    uint8_t* owner;
    size_t owner_len;
    uint8_t* classification;
    size_t classification_len;
    uint8_t requires_mfa : 1;
    uint8_t requires_encryption : 1;
    uint8_t requires_audit : 1;
} SNEPPXZTResource;

typedef struct {
    char* condition_name;
    size_t condition_name_len;
    SNEPPXZTConditionType condition_type;
    uint8_t* expected_value;
    size_t expected_value_len;
    uint8_t* actual_value;
    size_t actual_value_len;
    uint8_t satisfied : 1;
    uint8_t required : 1;
} SNEPPXZTCondition;

typedef struct {
    uint8_t* policy_id;
    size_t policy_id_len;
    SNEPPXZTCondition conditions[SNEPPX_ZT_POLICY_MAX_CONDITIONS];
    uint32_t num_conditions;
    SNEPPXZTDecision default_decision;
    uint32_t session_timeout_seconds;
    uint32_t reauth_interval_seconds;
    uint8_t require_continuous_verification : 1;
    uint8_t require_justification : 1;
    uint8_t log_all_requests : 1;
} SNEPPXZTPolicy;

typedef struct {
    SNEPPXZTSubject subject;
    SNEPPXZTResource resource;
    SNEPPXZTDecision decision;
    uint8_t* justification;
    size_t justification_len;
    uint64_t request_time_ns;
    uint64_t decision_time_ns;
    float trust_score;
    float risk_score;
    uint8_t* session_token;
    size_t session_token_len;
    uint8_t mfa_required : 1;
    uint8_t approved : 1;
} SNEPPXZTRequest;

int snepx_zt_create_subject(SNEPPXZTSubject* subject, const uint8_t* id, size_t id_len);
int snepx_zt_create_resource(SNEPPXZTResource* resource, const uint8_t* id, size_t id_len);
int snepx_zt_create_policy(SNEPPXZTPolicy* policy, const uint8_t* id, size_t id_len, SNEPPXZTDecision default_decision);
int snepx_zt_policy_add_condition(SNEPPXZTPolicy* policy, const SNEPPXZTCondition* condition);
int snepx_zt_evaluate(const SNEPPXZTPolicy* policy, const SNEPPXZTSubject* subject, const SNEPPXZTResource* resource, SNEPPXZTRequest* request);
int snepx_zt_compute_trust_score(const SNEPPXZTSubject* subject, float* score);
int snepx_zt_compute_risk_score(const SNEPPXZTSubject* subject, const SNEPPXZTResource* resource, float* score);
int snepx_zt_session_create(SNEPPXZTRequest* request, uint8_t* session_token, size_t* token_len);
int snepx_zt_session_validate(const uint8_t* session_token, size_t token_len);
int snepx_zt_session_revoke(const uint8_t* session_token, size_t token_len);
int snepx_zt_continuous_verification(SNEPPXZTSubject* subject, SNEPPXZTRequest* request);
int snepx_zt_request_destroy(SNEPPXZTRequest* request);

#endif
