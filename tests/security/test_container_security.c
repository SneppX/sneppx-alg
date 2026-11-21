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


TEST(test_container_security, suite) {
    container_manifest_t manifest;
    uint8_t test_manifest[] = "{\"mediaType\":\"application/vnd.oci.image.manifest.v1+json\",\"schemaVersion\":2,\"config\":{}}";
    if (SNEPPX_container_parse_manifest(test_manifest, sizeof(test_manifest), &manifest) != 0) {
        printf("FAIL: parse manifest\n"); FAIL() << "early exit (legacy return 1)";
    }
    printf("Manifest parsed: mediaType=%s layers=%d\n", manifest.media_type, manifest.num_layers);
    sbom_doc_t doc;
    SNEPPX_sbom_generate(&doc, "SNEPPX-algo:latest", "0.7.5");
    SNEPPX_sbom_add_component(&doc, "libssl", "3.0.12", "library", "OpenSSL Foundation");
    SNEPPX_sbom_add_component(&doc, "libcurl", "8.4.0", "library", "curl project");
    if (SNEPPX_sbom_validate(&doc) != 0) { printf("FAIL: SBOM validate\n"); FAIL() << "early exit (legacy return 1)"; }
    char json[4096];
    SNEPPX_sbom_export_json(&doc, json, sizeof(json));
    printf("SBOM JSON:\n%s\n", json);
    SNEPPX_container_init_vuln_db();
    vuln_result_t results[10];
    int n = SNEPPX_container_scan_vulns("nginx:latest", NULL, 0, results, 10);
    printf("Found %d vulnerabilities for nginx\n", n);
    container_stats_t stats;
    SNEPPX_container_get_stats(&stats);
    printf("Container stats: vulns=%d crit=%d high=%d\n", stats.num_vulns_in_db, stats.critical_count, stats.high_count);
    printf("PASS: Container security OK\n");
    return;
}
