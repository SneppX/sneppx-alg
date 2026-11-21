#ifndef SNEPPX_CONTAINER_SECURITY_H
#define SNEPPX_CONTAINER_SECURITY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <time.h>

/*
 * SNEPPX - Container Security
 *
 * WHAT
 *   Container Security.
 *
 * CONCEPT
 *   Provides the Container Security.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    char media_type[64];
    int schema_version;
    int num_layers;
    char digest[72];
    uint64_t size;
    uint8_t config_hash[32];
} container_manifest_t;

typedef struct {
    char name[128];
    char version[64];
    char type[32];
    char supplier[128];
    uint8_t hash[32];
    uint64_t size;
    int has_checksum;
} sbom_component_t;

typedef struct {
    char image_name[128];
    char version[32];
    char format[32];
    char namespace_str[128];
    char created[32];
    int num_components;
    sbom_component_t components[64];
    int verified;
} sbom_doc_t;

typedef struct {
    char cve_id[32];
    char severity[16];
    double cvss_score;
    int fixed;
    char fix_version[64];
    char package[128];
} vuln_result_t;

typedef struct {
    int num_images;
    int num_vulns_in_db;
    int total_vulns_found;
    int critical_count;
    int high_count;
    int medium_count;
    int low_count;
} container_stats_t;

/**
 * @brief Perform Container Verify Image.
 *
 * @param manifest [in] Manifest value.
 * @param manifest_len [in] Manifest Len value.
 * @param signature [in] Signature value.
 * @param sig_len [in] Sig Len value.
 * @param pubkey [in] Pubkey value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_verify_image(const uint8_t *manifest, size_t manifest_len, const uint8_t *signature, size_t sig_len, const uint8_t *pubkey);
/**
 * @brief Perform Container Parse Manifest.
 *
 * @param manifest [in] Manifest value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_parse_manifest(const uint8_t *manifest, size_t len, container_manifest_t *out);
/**
 * @brief Perform Container Verify Layer.
 *
 * @param layer_data [in] Layer Data value.
 * @param layer_len [in] Layer Len value.
 * @param expected_digest [in] Expected Digest value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_verify_layer(const uint8_t *layer_data, size_t layer_len, const char *expected_digest);
/**
 * @brief Perform Sbom Generate.
 *
 * @param doc [out] Doc value.
 * @param image_name [in] Image Name value.
 * @param version [in] Version value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_generate(sbom_doc_t *doc, const char *image_name, const char *version);
/**
 * @brief Perform Sbom Add Component.
 *
 * @param doc [out] Doc value.
 * @param name [in] Name value.
 * @param version [in] Version value.
 * @param type [in] Type value.
 * @param supplier [in] Supplier value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_add_component(sbom_doc_t *doc, const char *name, const char *version, const char *type, const char *supplier);
/**
 * @brief Perform Sbom Validate.
 *
 * @param doc [out] Doc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_validate(sbom_doc_t *doc);
/**
 * @brief Perform Sbom Export Json.
 *
 * @param doc [out] Doc value.
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_export_json(sbom_doc_t *doc, char *out, size_t out_len);
/**
 * @brief Perform Container Scan Vulns.
 *
 * @param image_name [in] Image Name value.
 * @param layer_data [in] Layer Data value.
 * @param layer_len [in] Layer Len value.
 * @param results [out] Results value.
 * @param max_results [in] Max Results value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_scan_vulns(const char *image_name, const uint8_t *layer_data, size_t layer_len, vuln_result_t *results, int max_results);
/**
 * @brief Perform Container Add Vuln.
 *
 * @param cve_id [in] Cve Id value.
 * @param package [in] Package value.
 * @param severity [in] Severity value.
 * @param cvss [in] Cvss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_add_vuln(const char *cve_id, const char *package, const char *severity, double cvss);
/**
 * @brief Perform Container Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_get_stats(container_stats_t *stats);
/**
 * @brief Perform Container Init Vuln Db.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_init_vuln_db(void);


#ifdef __cplusplus
}
#endif
#endif
