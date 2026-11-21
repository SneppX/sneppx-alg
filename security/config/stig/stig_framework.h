#ifndef SNEPPX_STIG_FRAMEWORK_H
#define SNEPPX_STIG_FRAMEWORK_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_STIG_MAX_VULNS 4096
#define SNEPPX_STIG_MAX_CAT_I 1024
#define SNEPPX_STIG_MAX_CAT_II 2048
#define SNEPPX_STIG_MAX_CAT_III 1024

/*
 * SNEPPX - Stig Framework
 *
 * WHAT
 *   Stig Framework.
 *
 * CONCEPT
 *   Provides the Stig Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_STIG_CATEGORY_I,
    SNEPPX_STIG_CATEGORY_II,
    SNEPPX_STIG_CATEGORY_III
} SNEPPXSTIGCategory;

typedef enum {
    SNEPPX_STIG_STATUS_OPEN,
    SNEPPX_STIG_STATUS_NOT_A_FINDING,
    SNEPPX_STIG_STATUS_NOT_APPLICABLE,
    SNEPPX_STIG_STATUS_MITIGATED,
    SNEPPX_STIG_STATUS_ONGOING
} SNEPPXSTIGStatus;

typedef struct {
    uint8_t* vuln_id;
    size_t vuln_id_len;
    uint8_t* stig_id;
    size_t stig_id_len;
    uint8_t* rule_title;
    size_t rule_title_len;
    uint8_t* vuln_discussion;
    size_t discussion_len;
    SNEPPXSTIGCategory category;
    SNEPPXSTIGStatus status;
    uint8_t* check_content;
    size_t check_len;
    uint8_t* fix_text;
    size_t fix_len;
    uint8_t* cci_ref;
    size_t cci_len;
    uint8_t* ia_controls;
    size_t ia_len;
    uint8_t* severity;
    size_t severity_len;
    uint8_t* asset_type;
    size_t asset_type_len;
    uint8_t* finding_details;
    size_t details_len;
    uint8_t* mitigation_date;
    size_t mitigation_date_len;
    uint8_t automated_check : 1;
    uint8_t automated_fix : 1;
} SNEPPXSTIGVulnerability;

typedef struct {
    uint8_t* benchmark_name;
    size_t benchmark_name_len;
    uint8_t* benchmark_version;
    size_t version_len;
    uint8_t* release_info;
    size_t release_len;
    SNEPPXSTIGVulnerability* vulns;
    uint32_t num_vulns;
    uint32_t cat_i_count;
    uint32_t cat_ii_count;
    uint32_t cat_iii_count;
    uint32_t open_count;
    uint32_t mitigated_count;
    float open_percentage;
    uint8_t* ckl_path;
    size_t ckl_path_len;
    uint8_t audited : 1;
    uint8_t imported : 1;
} SNEPPXSTIGFramework;

int snepx_stig_framework_init(SNEPPXSTIGFramework* stig, const uint8_t* name, size_t name_len, const uint8_t* version, size_t version_len);
int snepx_stig_framework_add_vuln(SNEPPXSTIGFramework* stig, const SNEPPXSTIGVulnerability* vuln);
int snepx_stig_framework_import_ckl(SNEPPXSTIGFramework* stig, const uint8_t* ckl_data, size_t ckl_len);
int snepx_stig_framework_export_ckl(const SNEPPXSTIGFramework* stig, uint8_t* out, size_t* out_len);
int snepx_stig_framework_audit(SNEPPXSTIGFramework* stig);
int snepx_stig_framework_audit_vuln(SNEPPXSTIGFramework* stig, uint32_t vuln_index, uint8_t* finding_output, size_t finding_len);
int snepx_stig_framework_apply_fix(SNEPPXSTIGFramework* stig, uint32_t vuln_index);
int snepx_stig_framework_compute_stats(SNEPPXSTIGFramework* stig);
int snepx_stig_framework_generate_poam(const SNEPPXSTIGFramework* stig, uint8_t* poam_out, size_t* poam_len);
int snepx_stig_framework_destroy(SNEPPXSTIGFramework* stig);

#endif
