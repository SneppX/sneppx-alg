#ifndef SNEPPX_IR_FORENSICS_ANALYZER_H
#define SNEPPX_IR_FORENSICS_ANALYZER_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_IR_FORENSICS_MAX_FINDINGS 4096
#define SNEPPX_IR_FORENSICS_MAX_FILE_SCAN_DEPTH 128
#define SNEPPX_IR_FORENSICS_CHUNK_SIZE 1048576

/*
 * SNEPPX - Forensics Analyzer
 *
 * WHAT
 *   Forensics Analyzer.
 *
 * CONCEPT
 *   Provides the Forensics Analyzer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_IR_ANALYSIS_MEMORY,
    SNEPPX_IR_ANALYSIS_DISK,
    SNEPPX_IR_ANALYSIS_NETWORK,
    SNEPPX_IR_ANALYSIS_REGISTRY,
    SNEPPX_IR_ANALYSIS_FILE_SYSTEM,
    SNEPPX_IR_ANALYSIS_PROCESS,
    SNEPPX_IR_ANALYSIS_MALWARE,
    SNEPPX_IR_ANALYSIS_ARTIFACT,
    SNEPPX_IR_ANALYSIS_TIMELINE,
    SNEPPX_IR_ANALYSIS_INDICATOR
} SNEPPXIRForensicsModule;

typedef enum {
    SNEPPX_IR_FINDING_CRITICAL,
    SNEPPX_IR_FINDING_HIGH,
    SNEPPX_IR_FINDING_MEDIUM,
    SNEPPX_IR_FINDING_LOW,
    SNEPPX_IR_FINDING_INFO
} SNEPPXIRFindingSeverity;

typedef struct {
    uint64_t finding_id;
    SNEPPXIRForensicsModule module;
    SNEPPXIRFindingSeverity severity;
    uint8_t* finding_title;
    size_t title_len;
    uint8_t* finding_description;
    size_t description_len;
    uint8_t* evidence_ref;
    size_t evidence_ref_len;
    uint8_t* ioc_ref;
    size_t ioc_ref_len;
    uint8_t* mitre_attack_id;
    size_t mitre_attack_id_len;
    uint8_t* raw_data;
    size_t raw_data_len;
    uint64_t timestamp_ns;
    float confidence_score;
    uint8_t* remediation;
    size_t remediation_len;
    uint8_t verified : 1;
    uint8_t false_positive : 1;
    uint8_t escalated : 1;
} SNEPPXIRForensicFinding;

typedef struct {
    uint8_t* output;
    size_t output_len;
    SNEPPXIRForensicFinding findings[SNEPPX_IR_FORENSICS_MAX_FINDINGS];
    uint32_t num_findings;
    uint64_t duration_ns;
    uint8_t* module_specific;
    size_t module_specific_len;
    uint8_t completed : 1;
    uint8_t errors : 1;
} SNEPPXIRForensicResult;

int snepx_ir_forensics_analyze_memory(const uint8_t* dump, size_t dump_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_analyze_disk_image(const uint8_t* image, size_t image_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_analyze_network_capture(const uint8_t* pcap, size_t pcap_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_analyze_registry_hive(const uint8_t* hive, size_t hive_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_analyze_file(const uint8_t* file_data, size_t file_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_analyze_process(const uint8_t* proc_dump, size_t dump_len, SNEPPXIRForensicResult* result);

int snepx_ir_forensics_strings(const uint8_t* data, size_t data_len, uint32_t min_len, uint8_t* out, size_t* out_len);
int snepx_ir_forensics_carve(const uint8_t* data, size_t data_len, const uint8_t* magic, size_t magic_len, uint8_t* out, size_t* out_len);
int snepx_ir_forensics_timeline(SNEPPXIRForensicFinding* findings, uint32_t num_findings, uint8_t* timeline_out, size_t* timeline_len);
int snepx_ir_forensics_super_timeline(SNEPPXIRForensicFinding* findings, uint32_t num_findings, uint8_t* st_out, size_t* st_len);

// Volatility-like plugin system
int snepx_ir_forensics_plugin_run(const char* plugin_name, const uint8_t* data, size_t data_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_plugin_scan_connections(const uint8_t* mem_dump, size_t dump_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_plugin_scan_processes(const uint8_t* mem_dump, size_t dump_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_plugin_scan_modules(const uint8_t* mem_dump, size_t dump_len, SNEPPXIRForensicResult* result);
int snepx_ir_forensics_plugin_scan_malware(const uint8_t* mem_dump, size_t dump_len, const uint8_t* yara_rules, size_t rules_len, SNEPPXIRForensicResult* result);

int snepx_ir_forensics_result_destroy(SNEPPXIRForensicResult* result);

#endif
