#ifndef SNEPPX_CIS_BENCHMARK_H
#define SNEPPX_CIS_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CIS_MAX_CONTROLS 1024
#define SNEPPX_CIS_MAX_TECHNOLOGIES 16

/*
 * SNEPPX - Cis Benchmark
 *
 * WHAT
 *   Cis Benchmark.
 *
 * CONCEPT
 *   Provides the Cis Benchmark.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_CIS_TECH_LINUX,
    SNEPPX_CIS_TECH_WINDOWS,
    SNEPPX_CIS_TECH_KUBERNETES,
    SNEPPX_CIS_TECH_DOCKER,
    SNEPPX_CIS_TECH_AWS,
    SNEPPX_CIS_TECH_AZURE,
    SNEPPX_CIS_TECH_GCP,
    SNEPPX_CIS_TECH_DATABASE_MYSQL,
    SNEPPX_CIS_TECH_DATABASE_POSTGRES,
    SNEPPX_CIS_TECH_DATABASE_MONGODB,
    SNEPPX_CIS_TECH_NETWORK_CISCO,
    SNEPPX_CIS_TECH_WEB_SERVER_NGINX,
    SNEPPX_CIS_TECH_WEB_SERVER_APACHE
} SNEPPXCISTechnology;

typedef enum {
    SNEPPX_CIS_CONTROL_PASSED,
    SNEPPX_CIS_CONTROL_FAILED,
    SNEPPX_CIS_CONTROL_NOT_APPLICABLE,
    SNEPPX_CIS_CONTROL_MANUALLY_CHECK,
    SNEPPX_CIS_CONTROL_ERROR
} SNEPPXCISControlStatus;

typedef enum {
    SNEPPX_CIS_LEVEL_1,
    SNEPPX_CIS_LEVEL_2
} SNEPPXCISLevel;

typedef struct {
    uint8_t* control_id;
    size_t control_id_len;
    uint8_t* control_title;
    size_t title_len;
    uint8_t* control_description;
    size_t description_len;
    SNEPPXCISLevel cis_level;
    SNEPPXCISControlStatus status;
    uint8_t* remediation;
    size_t remediation_len;
    uint8_t* audit_command;
    size_t audit_command_len;
    uint8_t* expected_output;
    size_t expected_output_len;
    uint8_t* actual_output;
    size_t actual_output_len;
    uint8_t* references;
    size_t references_len;
    uint8_t automated : 1;
    uint8_t scored : 1;
} SNEPPXCISControl;

typedef struct {
    SNEPPXCISTechnology technology;
    char* benchmark_version;
    size_t version_len;
    SNEPPXCISControl* controls;
    uint32_t num_controls;
    uint32_t passed_count;
    uint32_t failed_count;
    uint32_t na_count;
    uint32_t manual_count;
    uint32_t error_count;
    float compliance_score;
    uint8_t* report;
    size_t report_len;
    uint8_t audited : 1;
    uint8_t remediated : 1;
} SNEPPXCISBenchmark;

int snepx_cis_benchmark_init(SNEPPXCISBenchmark* bench, SNEPPXCISTechnology tech, const char* version);
int snepx_cis_benchmark_add_control(SNEPPXCISBenchmark* bench, const SNEPPXCISControl* control);
int snepx_cis_benchmark_audit(SNEPPXCISBenchmark* bench);
int snepx_cis_benchmark_audit_control(SNEPPXCISBenchmark* bench, uint32_t control_index, const uint8_t* actual_output, size_t output_len);
int snepx_cis_benchmark_remediate(SNEPPXCISBenchmark* bench, uint32_t control_index);
int snepx_cis_benchmark_remediate_all(SNEPPXCISBenchmark* bench);
int snepx_cis_benchmark_compute_score(SNEPPXCISBenchmark* bench);
int snepx_cis_benchmark_export_report(SNEPPXCISBenchmark* bench, uint8_t* out, size_t* out_len);
int snepx_cis_benchmark_export_csv(SNEPPXCISBenchmark* bench, uint8_t* out, size_t* out_len);
int snepx_cis_benchmark_compare(SNEPPXCISBenchmark* before, SNEPPXCISBenchmark* after, uint8_t* diff_out, size_t* diff_len);
int snepx_cis_benchmark_destroy(SNEPPXCISBenchmark* bench);

#endif
