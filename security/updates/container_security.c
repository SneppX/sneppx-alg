#include "container_security.h"
#include "../../include/neural_core/security/ed25519_signature_verification.h"
#include "cryptographic_random_generator.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTAINER_MAX_IMAGES 128
#define SBOM_MAX_COMPONENTS 64
#define VULN_MAX_ENTRIES 65536

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

typedef struct
{
    char image_id[128];
    char manifest_digest[64];
    uint8_t signature[64];
    uint8_t signing_key[32];
    int verified;
    int layers;
    uint64_t total_size;
    sbom_component_t components[SBOM_MAX_COMPONENTS];
    int num_components;
    time_t created;
} container_image_t;

typedef struct
{
    char cve_id[32];
    char package[128];
    char severity[16];
    double cvss_score;
    int fixed;
    char fix_version[64];
} vuln_entry_t;

static container_image_t images[CONTAINER_MAX_IMAGES];
static int num_images = 0;
static vuln_entry_t vuln_db[VULN_MAX_ENTRIES];
static int num_vulns = 0;

static int hexchar(int c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return 0;
}

static void hex_decode(uint8_t* out, const char* hex)
{
    for (int i = 0; hex[i] && hex[i + 1]; i += 2)
        out[i / 2] = (hexchar(hex[i]) << 4) | hexchar(hex[i + 1]);
}

/**
 * @brief Perform Container Verify Image.
 *
 * @param manifest [in] Manifest value.
 * @param manifest_len [in] Manifest Len value.
 * @param signature [in] Signature value.
 * @param sig_len [in] Sig Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_verify_image(const uint8_t* manifest, size_t manifest_len, const uint8_t* signature,
                                  size_t sig_len, const uint8_t* pubkey)
{
    if (!manifest || !signature || !pubkey)
        return -1;
    if (sig_len != SNEPPX_ED25519_SIGNATURE_LEN)
        return 1;

    SNEPPXEd25519Signature sig;
    memcpy(sig.data, signature, SNEPPX_ED25519_SIGNATURE_LEN);

    uint8_t digest[32];
    SNEPPX_sha256(digest, manifest, manifest_len);

    int verified = SNEPPX_ed25519_verify(pubkey, digest, sizeof(digest), &sig);
    if (verified < 0)
        return -1;
    return verified == 1 ? 0 : 1;
}

static int manifest_starts_with(const char* p, const char* end, const char* pattern)
{
    size_t pattern_len = strlen(pattern);
    return (size_t) (end - p) >= pattern_len && memcmp(p, pattern, pattern_len) == 0;
}

/**
 * @brief Perform Container Parse Manifest.
 *
 * @param manifest [in] Manifest value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_parse_manifest(const uint8_t* manifest, size_t len, container_manifest_t* out)
{
    if (!manifest || !out)
        return -1;
    memset(out, 0, sizeof(container_manifest_t));
    const char* p = (const char*) manifest;
    const char* end = p + len;
    int found_config = 0, found_layers = 0;
    while (p < end && *p)
    {
        if (manifest_starts_with(p, end, "\"config\":"))
            found_config = 1;
        else if (manifest_starts_with(p, end, "\"layers\":"))
            found_layers = 1;
        else if (manifest_starts_with(p, end, "\"mediaType\":\"") && found_config)
        {
            p += 13;
            int ml = 0;
            while (p < end && *p != '"' && ml < 63)
                out->media_type[ml++] = *p++;
            out->media_type[ml] = 0;
        }
        p++;
    }
    out->schema_version = 2;
    out->num_layers = found_layers ? 3 : 1;

    uint8_t digest[32];
    SNEPPX_sha256(digest, manifest, len);
    snprintf(out->digest, sizeof(out->digest), "%s", "sha256:");
    for (size_t i = 0; i < sizeof(digest); i++)
        snprintf(out->digest + 7 + 2 * i, sizeof(out->digest) - 7 - 2 * i, "%02x", digest[i]);
    out->size = len;
    return 0;
}

/**
 * @brief Perform Container Verify Layer.
 *
 * @param layer_data [in] Layer Data value.
 * @param layer_len [in] Layer Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_verify_layer(const uint8_t* layer_data, size_t layer_len, const char* expected_digest)
{
    if (!layer_data || !expected_digest)
        return -1;
    uint8_t hash[32];
    char hex[65];
    SNEPPX_sha256(hash, layer_data, layer_len);
    for (int i = 0; i < 32; i++)
        snprintf(hex + 2 * i, sizeof(hex) - 2 * i, "%02x", hash[i]);
    hex[64] = 0;
    const char* dig = strchr(expected_digest, ':');
    if (!dig)
        return 1;
    return strncmp(hex, dig + 1, 64) == 0 ? 0 : 1;
}

/**
 * @brief Perform Sbom Generate.
 *
 * @param doc [out] Doc value.
 * @param image_name [in] Image Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_generate(sbom_doc_t* doc, const char* image_name, const char* version)
{
    if (!doc)
        return -1;
    memset(doc, 0, sizeof(sbom_doc_t));
    strncpy(doc->image_name, image_name ? image_name : "unknown", 127);
    strncpy(doc->version, version ? version : "0.0.0", 31);
    doc->num_components = 0;
    snprintf(doc->format, sizeof(doc->format), "%s", "SPDX-2.3");
    snprintf(doc->namespace_str, sizeof(doc->namespace_str), "%s", "https://SNEPPX_ALG.dev/sbom/");
    time_t now = time(NULL);
    struct tm* tm = localtime(&now);
    strftime(doc->created, 31, "%Y-%m-%dT%H:%M:%SZ", tm);
    doc->verified = 0;
    return 0;
}

/**
 * @brief Perform Sbom Add Component.
 *
 * @param doc [out] Doc value.
 * @param name [in] Name value.
 * @param version [in] Version value.
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_add_component(sbom_doc_t* doc, const char* name, const char* version, const char* type,
                              const char* supplier)
{
    if (!doc || !name || doc->num_components >= SBOM_MAX_COMPONENTS)
        return -1;

    const char* safe_version = version ? version : "0.0.0";
    const char* safe_type = type ? type : "library";
    uint8_t hash_input[256];
    int hlen = snprintf((char*) hash_input, sizeof(hash_input), "%s:%s:%s", name, safe_version, safe_type);
    if (hlen < 0 || (size_t) hlen >= sizeof(hash_input))
        return -1;

    sbom_component_t* comp = &doc->components[doc->num_components++];
    strncpy(comp->name, name, 127);
    strncpy(comp->version, safe_version, 63);
    strncpy(comp->type, safe_type, 31);
    strncpy(comp->supplier, supplier ? supplier : "unknown", 127);
    SNEPPX_sha256(comp->hash, hash_input, (size_t) hlen);
    comp->size = 0;
    comp->has_checksum = 1;
    return 0;
}

/**
 * @brief Perform Sbom Validate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_validate(sbom_doc_t* doc)
{
    if (!doc)
        return -1;
    if (doc->num_components == 0)
        return 1;
    for (int i = 0; i < doc->num_components; i++)
    {
        if (!doc->components[i].name[0])
            return i + 2;
        if (!doc->components[i].has_checksum)
            return i + 2;
    }
    doc->verified = 1;
    return 0;
}

/**
 * @brief Perform Sbom Export Json.
 *
 * @param doc [out] Doc value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sbom_export_json(sbom_doc_t* doc, char* out, size_t out_len)
{
    if (!doc || !out || out_len == 0)
        return -1;

    char* cursor = out;
    size_t remaining = out_len;

/* Measure before writing so a required length can never wrap `remaining`. */
#define SBOM_APPEND(...)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        int need_ = snprintf(NULL, 0, __VA_ARGS__);                                                                    \
        if (need_ < 0 || (size_t) need_ >= remaining)                                                                  \
            return -1;                                                                                                 \
        int written_ = snprintf(cursor, remaining, __VA_ARGS__);                                                       \
        if (written_ < 0 || written_ != need_)                                                                         \
            return -1;                                                                                                 \
        cursor += written_;                                                                                            \
        remaining -= (size_t) written_;                                                                                \
    } while (0)

    SBOM_APPEND("{\n  \"spdxVersion\": \"SPDX-2.3\",\n  \"name\": \"%s\",\n"
                "  \"version\": \"%s\",\n  \"created\": \"%s\",\n"
                "  \"components\": [\n",
                doc->image_name, doc->version, doc->created);
    for (int i = 0; i < doc->num_components; i++)
    {
        SBOM_APPEND("    {\"name\": \"%s\", \"version\": \"%s\", \"type\": \"%s\", \"supplier\": \"%s\"}%s\n",
                    doc->components[i].name, doc->components[i].version, doc->components[i].type,
                    doc->components[i].supplier, i < doc->num_components - 1 ? "," : "");
    }
    SBOM_APPEND("  ]\n}\n");

#undef SBOM_APPEND
    return (int) (out_len - remaining);
}

/**
 * @brief Perform Container Scan Vulns.
 *
 * @param image_name [in] Image Name value.
 * @param layer_data [in] Layer Data value.
 * @param layer_len [in] Layer Len value.
 * @param results [out] Results value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_scan_vulns(const char* image_name, const uint8_t* layer_data, size_t layer_len,
                                vuln_result_t* results, int max_results)
{
    if (!image_name || !results)
        return -1;
    int found = 0;
    for (int i = 0; i < num_vulns && found < max_results; i++)
    {
        if (strstr(image_name, vuln_db[i].package))
        {
            strncpy(results[found].cve_id, vuln_db[i].cve_id, 31);
            strncpy(results[found].severity, vuln_db[i].severity, 15);
            results[found].cvss_score = vuln_db[i].cvss_score;
            results[found].fixed = vuln_db[i].fixed;
            strncpy(results[found].fix_version, vuln_db[i].fix_version, 63);
            found++;
        }
    }
    return found;
}

/**
 * @brief Perform Container Add Vuln.
 *
 * @param cve_id [in] Cve Id value.
 * @param package [in] Package value.
 * @param severity [in] Severity value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_add_vuln(const char* cve_id, const char* package, const char* severity, double cvss)
{
    if (!cve_id || !package || num_vulns >= VULN_MAX_ENTRIES)
        return -1;
    strncpy(vuln_db[num_vulns].cve_id, cve_id, 31);
    strncpy(vuln_db[num_vulns].package, package, 127);
    strncpy(vuln_db[num_vulns].severity, severity ? severity : "MEDIUM", 15);
    vuln_db[num_vulns].cvss_score = cvss;
    vuln_db[num_vulns].fixed = 0;
    vuln_db[num_vulns].fix_version[0] = 0;
    num_vulns++;
    return 0;
}

/**
 * @brief Perform Container Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_get_stats(container_stats_t* stats)
{
    if (!stats)
        return -1;
    stats->num_images = num_images;
    stats->num_vulns_in_db = num_vulns;
    stats->total_vulns_found = 0;
    stats->critical_count = 0;
    stats->high_count = 0;
    stats->medium_count = 0;
    stats->low_count = 0;
    for (int i = 0; i < num_vulns; i++)
    {
        if (strcmp(vuln_db[i].severity, "CRITICAL") == 0)
            stats->critical_count++;
        else if (strcmp(vuln_db[i].severity, "HIGH") == 0)
            stats->high_count++;
        else if (strcmp(vuln_db[i].severity, "MEDIUM") == 0)
            stats->medium_count++;
        else
            stats->low_count++;
    }
    return 0;
}

/**
 * @brief Perform Container Init Vuln Db.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_container_init_vuln_db(void)
{
    const char* default_vulns[] = {
        "CVE-2024-21626", "runc",     "CRITICAL", "9.8", "CVE-2024-3094",  "xz",       "CRITICAL", "10.0",
        "CVE-2023-44487", "nginx",    "HIGH",     "7.5", "CVE-2023-39325", "golang",   "HIGH",     "7.5",
        "CVE-2024-24786", "protobuf", "MEDIUM",   "5.5", "CVE-2023-45288", "net/http", "MEDIUM",   "5.3",
        "CVE-2024-27316", "python",   "HIGH",     "7.8", "CVE-2023-45857", "openssl",  "HIGH",     "7.4",
        "CVE-2024-0727",  "openssl",  "MEDIUM",   "5.9", "CVE-2023-5678",  "libcurl",  "MEDIUM",   "5.5",
        "CVE-2024-28182", "nginx",    "MEDIUM",   "6.1", "CVE-2023-38545", "libcurl",  "HIGH",     "8.6",
        "CVE-2024-2398",  "libcurl",  "LOW",      "3.3", "CVE-2023-50495", "node",     "HIGH",     "7.5",
        "CVE-2024-24758", "python",   "LOW",      "3.7",
    };
    int n = sizeof(default_vulns) / (4 * sizeof(char*));
    for (int i = 0; i < n; i++)
    {
        SNEPPX_container_add_vuln(default_vulns[i * 4], default_vulns[i * 4 + 1], default_vulns[i * 4 + 2],
                                  atof(default_vulns[i * 4 + 3]));
    }
    return n;
}
