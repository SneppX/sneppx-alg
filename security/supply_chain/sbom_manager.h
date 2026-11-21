#ifndef SNEPPX_SUPPLY_CHAIN_SBOM_H
#define SNEPPX_SUPPLY_CHAIN_SBOM_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SC_MAX_COMPONENTS 65536
#define SNEPPX_SC_MAX_LICENSES 64
#define SNEPPX_SC_MAX_VULNERABILITIES 4096
#define SNEPPX_SC_HASH_SIZE 64

/*
 * SNEPPX - Sbom Manager
 *
 * WHAT
 *   Sbom Manager.
 *
 * CONCEPT
 *   Provides the Sbom Manager.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_SC_SBOM_FORMAT_SPDX,
    SNEPPX_SC_SBOM_FORMAT_CYCLONEDX,
    SNEPPX_SC_SBOM_FORMAT_SWID
} SNEPPXSCSBOMFormat;

typedef enum {
    SNEPPX_SC_VULN_CRITICAL,
    SNEPPX_SC_VULN_HIGH,
    SNEPPX_SC_VULN_MEDIUM,
    SNEPPX_SC_VULN_LOW,
    SNEPPX_SC_VULN_NONE
} SNEPPXSCSeverity;

typedef enum {
    SNEPPX_SC_POLICY_BLOCK,
    SNEPPX_SC_POLICY_WARN,
    SNEPPX_SC_POLICY_AUDIT,
    SNEPPX_SC_POLICY_IGNORE
} SNEPPXSCPolicyAction;

typedef struct {
    char* component_name;
    size_t name_len;
    char* component_version;
    size_t version_len;
    char* supplier;
    size_t supplier_len;
    char* licenses[SNEPPX_SC_MAX_LICENSES];
    size_t license_lens[SNEPPX_SC_MAX_LICENSES];
    uint32_t num_licenses;
    uint8_t* hash[SNEPPX_SC_HASH_SIZE];
    size_t hash_len;
    char* purl;
    size_t purl_len;
    char* cpe;
    size_t cpe_len;
    char* download_location;
    size_t download_len;
    uint8_t* source_file;
    size_t source_file_len;
    uint8_t is_modified : 1;
    uint8_t is_optional : 1;
} SNEPPXSCComponent;

typedef struct {
    char* vuln_id;
    size_t vuln_id_len;
    char* cve_id;
    size_t cve_id_len;
    SNEPPXSCSeverity severity;
    float cvss_score;
    char* cvss_vector;
    size_t cvss_vector_len;
    char* affected_component;
    size_t affected_component_len;
    char* affected_version;
    size_t affected_version_len;
    char* fix_version;
    size_t fix_version_len;
    uint8_t* description;
    size_t description_len;
    uint8_t* remediation;
    size_t remediation_len;
    uint8_t exploitable : 1;
    uint8_t has_exploit : 1;
    uint8_t has_patch : 1;
} SNEPPXSCVulnerability;

typedef struct {
    char* sbom_name;
    size_t sbom_name_len;
    char* sbom_version;
    size_t sbom_version_len;
    char* creator;
    size_t creator_len;
    uint64_t created_ns;
    SNEPPXSCSBOMFormat format;
    SNEPPXSCComponent* components;
    uint32_t num_components;
    SNEPPXSCVulnerability* vulnerabilities;
    uint32_t num_vulnerabilities;
    uint8_t* raw_xml;
    size_t raw_xml_len;
    uint8_t* raw_json;
    size_t raw_json_len;
    uint8_t verified : 1;
} SNEPPXSCSBOM;

int snepx_sc_sbom_create(SNEPPXSCSBOM* sbom, const char* name, const char* version, SNEPPXSCSBOMFormat format);
int snepx_sc_sbom_add_component(SNEPPXSCSBOM* sbom, const SNEPPXSCComponent* component);
int snepx_sc_sbom_add_vulnerability(SNEPPXSCSBOM* sbom, const SNEPPXSCVulnerability* vuln);
int snepx_sc_sbom_export_spdx(SNEPPXSCSBOM* sbom, uint8_t* out, size_t* out_len);
int snepx_sc_sbom_export_cyclonedx(SNEPPXSCSBOM* sbom, uint8_t* out, size_t* out_len);
int snepx_sc_sbom_import_spdx(SNEPPXSCSBOM* sbom, const uint8_t* in, size_t in_len);
int snepx_sc_sbom_import_cyclonedx(SNEPPXSCSBOM* sbom, const uint8_t* in, size_t in_len);
int snepx_sc_sbom_verify_integrity(SNEPPXSCSBOM* sbom);
int snepx_sc_sbom_policy_check(SNEPPXSCSBOM* sbom, uint8_t* violations_out, size_t* violations_len);
int snepx_sc_sbom_diff(SNEPPXSCSBOM* old_sbom, SNEPPXSCSBOM* new_sbom, uint8_t* diff_out, size_t* diff_len);
int snepx_sc_sbom_destroy(SNEPPXSCSBOM* sbom);

// Dependency chain analysis
int snepx_sc_dependency_depth(const SNEPPXSCSBOM* sbom, const char* component_name, uint32_t* depth);
int snepx_sc_dependency_tree(const SNEPPXSCSBOM* sbom, const char* root_component, uint8_t* tree_out, size_t* tree_len);
int snepx_sc_find_transitive_vulnerabilities(SNEPPXSCSBOM* sbom, const char* component_name, SNEPPXSCVulnerability* vulns, uint32_t* num_vulns);

#endif
