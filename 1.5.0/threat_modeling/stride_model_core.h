#ifndef SNEPPX_THREAT_MODEL_STRIDE_H
#define SNEPPX_THREAT_MODEL_STRIDE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TM_MAX_THREATS 2048
#define SNEPPX_TM_MAX_MITIGATIONS 64

/*
 * SNEPPX - Stride Model Core
 *
 * WHAT
 *   Stride Model Core.
 *
 * CONCEPT
 *   Provides the Stride Model Core.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TM_SPOOFING,
    SNEPPX_TM_TAMPERING,
    SNEPPX_TM_REPUDIATION,
    SNEPPX_TM_INFO_DISCLOSURE,
    SNEPPX_TM_DOS,
    SNEPPX_TM_ELEVATION_PRIVILEGE
} SNEPPXTmStrideCategory;

typedef enum {
    SNEPPX_TM_RISK_CRITICAL,
    SNEPPX_TM_RISK_HIGH,
    SNEPPX_TM_RISK_MEDIUM,
    SNEPPX_TM_RISK_LOW
} SNEPPXTmRiskRating;

typedef enum {
    SNEPPX_TM_MITIGATED,
    SNEPPX_TM_ACCEPTED,
    SNEPPX_TM_TRANSFERRED,
    SNEPPX_TM_OPEN
} SNEPPXTmMitigationStatus;

typedef struct {
    uint64_t threat_id;
    SNEPPXTmStrideCategory category;
    uint8_t* threat_name;
    size_t name_len;
    uint8_t* description;
    size_t desc_len;
    uint8_t* affected_component;
    size_t component_len;
    uint8_t* attack_vector;
    size_t vector_len;
    float dread_damage;
    float dread_reproducibility;
    float dread_exploitability;
    float dread_affected;
    float dread_discoverability;
    float dread_total;
    SNEPPXTmRiskRating risk_rating;
    uint8_t* mitigations[SNEPPX_TM_MAX_MITIGATIONS];
    size_t mitigation_lens[SNEPPX_TM_MAX_MITIGATIONS];
    uint32_t num_mitigations;
    SNEPPXTmMitigationStatus mitigation_status;
    uint8_t* verification_evidence;
    size_t evidence_len;
} SNEPPXTmThreat;

typedef struct {
    uint8_t* model_name;
    size_t name_len;
    uint8_t* model_version;
    size_t version_len;
    uint8_t* system_description;
    size_t system_desc_len;
    SNEPPXTmThreat* threats;
    uint32_t num_threats;
    uint32_t critical_count;
    uint32_t high_count;
    uint32_t medium_count;
    uint32_t low_count;
    float avg_dread;
    uint8_t reviewed : 1;
    uint8_t approved : 1;
} SNEPPXTmModel;

int snepx_tm_model_create(SNEPPXTmModel* model, const uint8_t* name, size_t name_len, const uint8_t* version, size_t version_len);
int snepx_tm_model_add_threat(SNEPPXTmModel* model, const SNEPPXTmThreat* threat);
int snepx_tm_model_compute_risk(SNEPPXTmModel* model);
int snepx_tm_model_compute_dread(SNEPPXTmThreat* threat);
int snepx_tm_model_export_report(const SNEPPXTmModel* model, uint8_t* out, size_t* out_len);
int snepx_tm_model_export_attack_tree(const SNEPPXTmModel* model, uint8_t* out, size_t* out_len);
int snepx_tm_model_destroy(SNEPPXTmModel* model);

#endif
