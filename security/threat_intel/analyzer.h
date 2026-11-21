#ifndef SNEPPX_THREAT_INTEL_ANALYZER_H
#define SNEPPX_THREAT_INTEL_ANALYZER_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TI_IOI_MAX_TAGS 64
#define SNEPPX_TI_IOI_MAX_CATEGORIES 32
#define SNEPPX_TI_ANALYSIS_MAX_KILL_CHAIN 16
#define SNEPPX_TI_ANALYSIS_MAX_MITRE_TTP 128

/*
 * SNEPPX - Analyzer
 *
 * WHAT
 *   Analyzer.
 *
 * CONCEPT
 *   Provides the Analyzer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TI_IOC_TYPE_IPV4,
    SNEPPX_TI_IOC_TYPE_IPV6,
    SNEPPX_TI_IOC_TYPE_DOMAIN,
    SNEPPX_TI_IOC_TYPE_URL,
    SNEPPX_TI_IOC_TYPE_HASH_MD5,
    SNEPPX_TI_IOC_TYPE_HASH_SHA1,
    SNEPPX_TI_IOC_TYPE_HASH_SHA256,
    SNEPPX_TI_IOC_TYPE_EMAIL,
    SNEPPX_TI_IOC_TYPE_CVE,
    SNEPPX_TI_IOC_TYPE_ASN,
    SNEPPX_TI_IOC_TYPE_MUTEX,
    SNEPPX_TI_IOC_TYPE_REGISTRY_KEY,
    SNEPPX_TI_IOC_TYPE_FILE_PATH,
    SNEPPX_TI_IOC_TYPE_CERTIFICATE_SERIAL,
    SNEPPX_TI_IOC_TYPE_YARA_RULE,
    SNEPPX_TI_IOC_TYPE_SSL_JA3,
    SNEPPX_TI_IOC_TYPE_SSL_JA3S,
    SNEPPX_TI_IOC_TYPE_USER_AGENT
} SNEPPXTIIoCType;

typedef enum {
    SNEPPX_TI_CONFIDENCE_NONE,
    SNEPPX_TI_CONFIDENCE_LOW,
    SNEPPX_TI_CONFIDENCE_MEDIUM,
    SNEPPX_TI_CONFIDENCE_HIGH,
    SNEPPX_TI_CONFIDENCE_VERIFIED
} SNEPPXTIConfidenceLevel;

typedef enum {
    SNEPPX_TI_KILL_CHAIN_RECONNAISSANCE,
    SNEPPX_TI_KILL_CHAIN_WEAPONIZATION,
    SNEPPX_TI_KILL_CHAIN_DELIVERY,
    SNEPPX_TI_KILL_CHAIN_EXPLOITATION,
    SNEPPX_TI_KILL_CHAIN_INSTALLATION,
    SNEPPX_TI_KILL_CHAIN_C2,
    SNEPPX_TI_KILL_CHAIN_ACTIONS_OBJECTIVES
} SNEPPXTIKillChainPhase;

typedef struct {
    uint64_t ioc_id;
    SNEPPXTIIOType ioc_type;
    SNEPPXTIConfidenceLevel confidence;
    uint8_t* indicator_value;
    size_t indicator_len;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
    uint32_t occurrence_count;
    uint8_t* tags[SNEPPX_TI_IOI_MAX_TAGS];
    size_t tag_lens[SNEPPX_TI_IOI_MAX_TAGS];
    uint32_t num_tags;
    uint8_t* categories[SNEPPX_TI_IOI_MAX_CATEGORIES];
    size_t category_lens[SNEPPX_TI_IOI_MAX_CATEGORIES];
    uint32_t num_categories;
    uint8_t* source_ref;
    size_t source_ref_len;
    char* mitre_attack_id[SNEPPX_TI_ANALYSIS_MAX_MITRE_TTP];
    uint32_t num_mitre_attack;
    uint8_t* threat_actor;
    size_t threat_actor_len;
    uint8_t* malware_family;
    size_t malware_family_len;
    uint8_t* campaign_ref;
    size_t campaign_ref_len;
    uint8_t* country_code;
    size_t country_code_len;
    float risk_score;
    float severity_score;
} SNEPPXTIIoC;

typedef struct {
    SNEPPXTIKillChainPhase phases[SNEPPX_TI_ANALYSIS_MAX_KILL_CHAIN];
    uint32_t num_phases;
    uint8_t* narrative;
    size_t narrative_len;
    uint8_t* recommendation;
    size_t recommendation_len;
    uint8_t* diamond_model;
    size_t diamond_model_len;
    uint8_t actor_attribution : 1;
    uint8_t campaign_linked : 1;
    uint8_t automated_response : 1;
} SNEPPXTIKillChainAnalysis;

typedef struct {
    SNEPPXTIIoC* iocs;
    uint32_t num_iocs;
    uint32_t ioc_capacity;
    SNEPPXTIKillChainAnalysis* analyses;
    uint32_t num_analyses;
    uint8_t* aggregation_key;
    size_t aggregation_key_len;
    uint64_t analysis_timestamp_ns;
    uint8_t analyzed : 1;
    uint8_t correlated : 1;
    uint8_t enriched : 1;
} SNEPPXTIAnalyzerResult;

int snepx_ti_analyzer_create_ioc(SNEPPXTIIoC* ioc, SNEPPXTIIOType type, const uint8_t* value, size_t value_len);
int snepx_ti_analyzer_add_tag(SNEPPXTIIoC* ioc, const uint8_t* tag, size_t tag_len);
int snepx_ti_analyzer_add_category(SNEPPXTIIoC* ioc, const uint8_t* category, size_t category_len);
int snepx_ti_analyzer_add_mitre_attack(SNEPPXTIIoC* ioc, const char* mitre_id);
int snepx_ti_analyzer_compute_risk(SNEPPXTIIoC* ioc);
int snepx_ti_analyzer_compute_confidence(SNEPPXTIIoC* ioc);
int snepx_ti_analyzer_kill_chain(SNEPPXTIIoC* ioc, SNEPPXTIKillChainAnalysis* analysis);
int snepx_ti_analyzer_correlate(SNEPPXTIIoC* iocs, uint32_t num_iocs, SNEPPXTIAnalyzerResult* result);
int snepx_ti_analyzer_cluster(SNEPPXTIIoC* iocs, uint32_t num_iocs, uint32_t* cluster_ids, uint32_t* num_clusters);
int snepx_ti_analyzer_temporal_analysis(SNEPPXTIIoC* iocs, uint32_t num_iocs, uint64_t* activity_spike_timestamps, uint32_t* num_spikes);
int snepx_ti_analyzer_diamond_model(SNEPPXTIIoC* ioc, SNEPPXTIKillChainAnalysis* analysis);
int snepx_ti_analyzer_ioc_destroy(SNEPPXTIIoC* ioc);
int snepx_ti_analyzer_result_destroy(SNEPPXTIAnalyzerResult* result);

#endif
