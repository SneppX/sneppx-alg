#include "container_security.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Container Security
 *
 * WHAT
 *   Test Container Security.
 *
 * CONCEPT
 *   Provides the Test Container Security.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

static void test_sbom_export_tiny_buffer(void)
{
    sbom_doc_t doc;
    SNEPPX_sbom_generate(&doc, "sx:test", "1.0.0");
    SNEPPX_sbom_add_component(&doc, "lib", "1.0.0", "library", "vendor");

    char buf[16];
    memset(buf, 0x7f, sizeof(buf));
    int rc = SNEPPX_sbom_export_json(&doc, buf, sizeof(buf));
    SX_ASSERT(rc == -1, "tiny buffer is rejected");
    for (size_t i = 0; i < sizeof(buf); i++)
        SX_ASSERT((unsigned char) buf[i] == 0x7f, "tiny buffer is untouched");
}

static void test_sbom_add_component_rejects_oversized_hash_input(void)
{
    sbom_doc_t doc;
    SNEPPX_sbom_generate(&doc, "sx:test", "1.0.0");

    char long_version[300];
    memset(long_version, 'x', sizeof(long_version) - 1);
    long_version[sizeof(long_version) - 1] = '\0';
    int rc = SNEPPX_sbom_add_component(&doc, "lib", long_version, "library", "vendor");
    SX_ASSERT(rc == -1, "oversized hash input rejected");
    SX_ASSERT(doc.num_components == 0, "no component recorded");

    rc = SNEPPX_sbom_add_component(&doc, "lib", NULL, NULL, NULL);
    SX_ASSERT(rc == 0, "null metadata uses defaults");
    SX_ASSERT(strcmp(doc.components[0].version, "0.0.0") == 0, "default version");
    SX_ASSERT(strcmp(doc.components[0].type, "library") == 0, "default type");
    SX_ASSERT(strcmp(doc.components[0].supplier, "unknown") == 0, "default supplier");
}

static void test_manifest_digest_is_content_derived(void)
{
    const uint8_t manifest[] =
        "{\"mediaType\":\"application/vnd.oci.image.manifest.v1+json\",\"schemaVersion\":2,\"config\":{}}";
    container_manifest_t parsed;
    SX_ASSERT(SNEPPX_container_parse_manifest(manifest, strlen((const char*) manifest), &parsed) == 0,
              "parse manifest without trailing NUL");
    SX_ASSERT(strcmp(parsed.digest,
                     "sha256:1c6814dabf1efebbdb79879efd346f054f55160f03eaa620b78c97df7e4f6f66") == 0,
              "digest follows manifest bytes");
    SX_ASSERT(parsed.size == strlen((const char*) manifest), "size follows input length");
}

static void test_manifest_parse_accepts_non_nul_terminated_input(void)
{
    const uint8_t partial[18] = {'"', 'c', 'o', 'n', 'f', 'i', 'g', '"', ':',
                                 '"', 'l', 'a', 'y', 'e', 'r', 's', '"', ':'};
    container_manifest_t parsed;
    SX_ASSERT(SNEPPX_container_parse_manifest(partial, sizeof(partial), &parsed) == 0, "parse bounded buffer");
    SX_ASSERT(parsed.num_layers == 3, "bounded layers key detected");
    SX_ASSERT(strncmp(parsed.digest, "sha256:", 7) == 0, "bounded digest generated");
}

TEST(test_container_security, suite)
{
    container_manifest_t manifest;
    uint8_t test_manifest[] =
        "{\"mediaType\":\"application/vnd.oci.image.manifest.v1+json\",\"schemaVersion\":2,\"config\":{}}";
    if (SNEPPX_container_parse_manifest(test_manifest, sizeof(test_manifest), &manifest) != 0)
    {
        printf("FAIL: parse manifest\n");
        FAIL() << "early exit (legacy return 1)";
    }
    printf("Manifest parsed: mediaType=%s layers=%d\n", manifest.media_type, manifest.num_layers);
    sbom_doc_t doc;
    SNEPPX_sbom_generate(&doc, "SNEPPX-algo:latest", "0.7.5");
    SNEPPX_sbom_add_component(&doc, "libssl", "3.0.12", "library", "OpenSSL Foundation");
    SNEPPX_sbom_add_component(&doc, "libcurl", "8.4.0", "library", "curl project");
    if (SNEPPX_sbom_validate(&doc) != 0)
    {
        printf("FAIL: SBOM validate\n");
        FAIL() << "early exit (legacy return 1)";
    }
    char json[4096];
    SNEPPX_sbom_export_json(&doc, json, sizeof(json));
    printf("SBOM JSON:\n%s\n", json);
    SNEPPX_container_init_vuln_db();
    vuln_result_t results[10];
    int n = SNEPPX_container_scan_vulns("nginx:latest", NULL, 0, results, 10);
    printf("Found %d vulnerabilities for nginx\n", n);
    container_stats_t stats;
    SNEPPX_container_get_stats(&stats);
    printf("Container stats: vulns=%d crit=%d high=%d\n", stats.num_vulns_in_db, stats.critical_count,
           stats.high_count);
    printf("PASS: Container security OK\n");
    return;
}

TEST(test_container_security, export_tiny_buffer)
{
    test_sbom_export_tiny_buffer();
}

TEST(test_container_security, add_component_rejects_oversized_hash_input)
{
    test_sbom_add_component_rejects_oversized_hash_input();
}

TEST(test_container_security, manifest_digest_is_content_derived)
{
    test_manifest_digest_is_content_derived();
}

TEST(test_container_security, manifest_parse_accepts_non_nul_terminated_input)
{
    test_manifest_parse_accepts_non_nul_terminated_input();
}
