#ifndef SNEPPX_FEDRAMP_FRAMEWORK_H
#define SNEPPX_FEDRAMP_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define SNEPPX_FEDRAMP_MAX_CONTROLS 400
#define SNEPPX_FEDRAMP_IMPACT_LEVELS 3

/*
 * SNEPPX - Fedramp Framework
 *
 * WHAT
 *   Fedramp Framework.
 *
 * CONCEPT
 *   Provides the Fedramp Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_FEDRAMP_LOW,
    SNEPPX_FEDRAMP_MODERATE,
    SNEPPX_FEDRAMP_HIGH
} SNEPPPXFedRAMPImpactLevel;

typedef enum {
    SNEPPX_FEDRAMP_CONTROL_FAMILY_AC,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_AT,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_AU,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_CA,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_CM,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_CP,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_IA,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_IR,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_MA,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_MP,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_PS,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_PE,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_PL,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_PM,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_RA,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_SA,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_SC,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_SI,
    SNEPPX_FEDRAMP_CONTROL_FAMILY_SR
} SNEPPPXFedRAMPControlFamily;

typedef struct {
    char control_id[16];
    SNEPPPXFedRAMPControlFamily family;
    char control_name[256];
    char control_text[1024];
    uint8_t implemented : 1;
    uint8_t assessed : 1;
    uint8_t authorized : 1;
    uint8_t ongoing : 1;
    uint8_t* implementation_description;
    size_t implementation_desc_len;
    uint64_t last_assessment;
    uint32_t risk_remaining;
    char poaam[512];
} SNEPPPXFedRAMPControl;

typedef struct {
    uint32_t num_controls;
    SNEPPPXFedRAMPControl controls[SNEPPX_FEDRAMP_MAX_CONTROLS];
    SNEPPPXFedRAMPImpactLevel impact_level;
    char system_name[256];
    char system_id[64];
    char agency[256];
    char authorizing_official[256];
    uint64_t authorization_date;
    uint64_t reauthorization_date;
    uint8_t* ssp;
    size_t ssp_len;
    uint8_t* sap;
    size_t sap_len;
    uint8_t* sar;
    size_t sar_len;
    uint8_t* poaam_doc;
    size_t poaam_len;
} SNEPPPXFedRAMPAuthPackage;

int snepx_fedramp_init(SNEPPPXFedRAMPAuthPackage* package, const char* system_name, SNEPPPXFedRAMPImpactLevel level);
int snepx_fedramp_add_control(SNEPPPXFedRAMPAuthPackage* package, const SNEPPPXFedRAMPControl* control);
int snepx_fedramp_assess_control(SNEPPPXFedRAMPAuthPackage* package, const char* control_id, uint32_t risk);
int snepx_fedramp_generate_ssp(SNEPPPXFedRAMPAuthPackage* package, uint8_t* ssp_out, size_t* ssp_len);
int snepx_fedramp_generate_sap(SNEPPPXFedRAMPAuthPackage* package, uint8_t* sap_out, size_t* sap_len);
int snepx_fedramp_generate_sar(SNEPPPXFedRAMPAuthPackage* package, uint8_t* sar_out, size_t* sar_len);
int snepx_fedramp_continuous_monitoring(SNEPPPXFedRAMPAuthPackage* package, uint8_t* monitor_report, size_t* report_len);
int snepx_fedramp_readiness_assessment(SNEPPPXFedRAMPAuthPackage* package, double* readiness_score);
int snepx_fedramp_compliance_percentage(SNEPPPXFedRAMPAuthPackage* package, double* compliance_pct);

#endif
