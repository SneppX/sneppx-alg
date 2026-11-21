#ifndef SNEPPX_SOC2_FRAMEWORK_H
#define SNEPPX_SOC2_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define SNEPPX_SOC2_MAX_CONTROLS 512
#define SNEPPX_SOC2_MAX_EVIDENCE 4096
#define SNEPPX_SOC2_TRUST_CRITERIA 5

/*
 * SNEPPX - Soc2 Framework
 *
 * WHAT
 *   Soc2 Framework.
 *
 * CONCEPT
 *   Provides the Soc2 Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_SOC2_SECURITY,
    SNEPPX_SOC2_AVAILABILITY,
    SNEPPX_SOC2_CONFIDENTIALITY,
    SNEPPX_SOC2_PROCESSING_INTEGRITY,
    SNEPPX_SOC2_PRIVACY
} SNEPPXSOC2TrustCriteria;

typedef enum {
    SNEPPX_SOC2_CONTROL_DESIGNED,
    SNEPPX_SOC2_CONTROL_IMPLEMENTED,
    SNEPPX_SOC2_CONTROL_OPERATING_EFFECTIVE,
    SNEPPX_SOC2_CONTROL_DEVIATION,
    SNEPPX_SOC2_CONTROL_REMEDIATED
} SNEPPXSOC2ControlStatus;

typedef struct {
    uint32_t control_id;
    char control_name[128];
    char control_description[512];
    SNEPPXSOC2TrustCriteria criteria;
    SNEPPXSOC2ControlStatus status;
    char control_owner[128];
    uint64_t last_reviewed;
    uint64_t next_review;
    uint8_t* evidence_links;
    size_t evidence_links_len;
    uint32_t risk_score;
    uint8_t automated : 1;
    uint8_t critical : 1;
} SNEPPXSOC2Control;

typedef struct {
    uint32_t num_controls;
    SNEPPXSOC2Control controls[SNEPPX_SOC2_MAX_CONTROLS];
    uint64_t assessment_period_start;
    uint64_t assessment_period_end;
    char assessor_name[256];
    char assessor_company[256];
    uint32_t total_controls;
    uint32_t designed_controls;
    uint32_t implemented_controls;
    uint32_t effective_controls;
    uint32_t deviations;
    uint32_t remediated;
    uint8_t* report;
    size_t report_len;
    uint8_t opinion_qualified : 1;
    uint8_t opinion_adverse : 1;
    uint8_t opinion_disclaimer : 1;
    uint8_t opinion_unqualified : 1;
} SNEPPXSOC2Assessment;

int snepx_soc2_assessment_init(SNEPPXSOC2Assessment* assessment, const char* assessor, const char* company);
int snepx_soc2_add_control(SNEPPXSOC2Assessment* assessment, const SNEPPXSOC2Control* control);
int snepx_soc2_update_control(SNEPPXSOC2Assessment* assessment, uint32_t control_id, SNEPPXSOC2ControlStatus status);
int snepx_soc2_add_evidence(SNEPPXSOC2Assessment* assessment, uint32_t control_id, const uint8_t* evidence, size_t evidence_len);
int snepx_soc2_generate_report(SNEPPXSOC2Assessment* assessment, uint8_t* report_out, size_t* report_len);
int snepx_soc2_summary(SNEPPXSOC2Assessment* assessment, char* summary_out, size_t summary_max);
int snepx_soc2_opinion(SNEPPXSOC2Assessment* assessment, uint8_t qualified, uint8_t adverse, uint8_t disclaimer, uint8_t unqualified);

#endif
