#ifndef SNEPPX_TI_ENRICHMENT_PIPELINE_H
#define SNEPPX_TI_ENRICHMENT_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TI_ENRICHMENT_MAX_STAGES 32
#define SNEPPX_TI_ENRICHMENT_MAX_CACHE_ENTRIES 131072
#define SNEPPX_TI_ENRICHMENT_CACHE_TTL_NS 3600000000000ULL

/*
 * SNEPPX - Enrichment Pipeline
 *
 * WHAT
 *   Enrichment Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TI_ENRICHMENT_STAGE_GEOIP,
    SNEPPX_TI_ENRICHMENT_STAGE_ASN_LOOKUP,
    SNEPPX_TI_ENRICHMENT_STAGE_RDNS,
    SNEPPX_TI_ENRICHMENT_STAGE_WHOIS,
    SNEPPX_TI_ENRICHMENT_STAGE_VT_REPUTATION,
    SNEPPX_TI_ENRICHMENT_STAGE_ABUSEIPDB,
    SNEPPX_TI_ENRICHMENT_STAGE_SSL_CERTIFICATE,
    SNEPPX_TI_ENRICHMENT_STAGE_PORT_SCAN,
    SNEPPX_TI_ENRICHMENT_STAGE_HTTP_FINGERPRINT,
    SNEPPX_TI_ENRICHMENT_STAGE_JA3_HASH,
    SNEPPX_TI_ENRICHMENT_STAGE_JARM_HASH,
    SNEPPX_TI_ENRICHMENT_STAGE_YARA_SCAN,
    SNEPPX_TI_ENRICHMENT_STAGE_SANDBOX_DETONATION,
    SNEPPX_TI_ENRICHMENT_STAGE_ML_CLASSIFICATION,
    SNEPPX_TI_ENRICHMENT_STAGE_THREAT_ACTOR_MATCHING,
    SNEPPX_TI_ENRICHMENT_STAGE_CAMPAIGN_ANALYSIS
} SNEPPXIEnrichmentStage;

typedef enum {
    SNEPPX_TI_ENRICHMENT_RESULT_SUCCESS,
    SNEPPX_TI_ENRICHMENT_RESULT_CACHE_HIT,
    SNEPPX_TI_ENRICHMENT_RESULT_RATE_LIMITED,
    SNEPPX_TI_ENRICHMENT_RESULT_TIMEOUT,
    SNEPPX_TI_ENRICHMENT_RESULT_NOT_FOUND,
    SNEPPX_TI_ENRICHMENT_RESULT_ERROR
} SNEPPXTEnrichmentResult;

typedef struct {
    SNEPPXIEnrichmentStage stage;
    SNEPPXTEnrichmentResult result;
    uint64_t stage_duration_ns;
    uint8_t* enriched_data;
    size_t enriched_data_len;
    uint8_t* raw_response;
    size_t raw_response_len;
    uint32_t http_status;
    uint8_t* error_message;
    size_t error_message_len;
    uint8_t cached : 1;
} SNEPPXTEnrichmentStageResult;

typedef struct {
    uint8_t* cache_key;
    size_t cache_key_len;
    uint8_t* cached_data;
    size_t cached_data_len;
    uint64_t cached_at_ns;
    uint64_t ttl_ns;
} SNEPPXTEnrichmentCacheEntry;

typedef struct {
    SNEPPXTEnrichmentStage stage_order[SNEPPX_TI_ENRICHMENT_MAX_STAGES];
    uint32_t num_stages;
    uint32_t parallelism;
    uint32_t timeout_ms;
    uint32_t max_retries;
    uint32_t rate_limit_rps;
    SNEPPXTEnrichmentCacheEntry* cache;
    uint32_t cache_entries;
    uint32_t cache_capacity;
    uint8_t* api_keys[16];
    size_t api_key_lens[16];
    uint32_t num_api_keys;
    uint8_t cache_enabled : 1;
    uint8_t fail_open : 1;
} SNEPPXTEnrichmentPipeline;

int snepx_ti_enrichment_pipeline_init(SNEPPXTEnrichmentPipeline* pipeline);
int snepx_ti_enrichment_pipeline_add_stage(SNEPPXTEnrichmentPipeline* pipeline, SNEPPXIEnrichmentStage stage);
int snepx_ti_enrichment_pipeline_remove_stage(SNEPPXTEnrichmentPipeline* pipeline, uint32_t stage_index);
int snepx_ti_enrichment_pipeline_set_api_key(SNEPPXTEnrichmentPipeline* pipeline, SNEPPXIEnrichmentStage stage, const uint8_t* key, size_t key_len);
int snepx_ti_enrichment_pipeline_run(SNEPPXTEnrichmentPipeline* pipeline, const uint8_t* indicator, size_t indicator_len, SNEPPXIEnrichmentStageResult* results, uint32_t* num_results);
int snepx_ti_enrichment_pipeline_run_parallel(SNEPPXTEnrichmentPipeline* pipeline, const uint8_t* indicators, size_t total_len, uint32_t num_indicators, SNEPPXIEnrichmentStageResult* results, uint32_t max_results);

int snepx_ti_enrichment_pipeline_cache_get(SNEPPXTEnrichmentPipeline* pipeline, const uint8_t* key, size_t key_len, uint8_t** data, size_t* data_len);
int snepx_ti_enrichment_pipeline_cache_put(SNEPPXTEnrichmentPipeline* pipeline, const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len);
int snepx_ti_enrichment_pipeline_cache_invalidate(SNEPPXTEnrichmentPipeline* pipeline, const uint8_t* key, size_t key_len);
int snepx_ti_enrichment_pipeline_cache_clear(SNEPPXTEnrichmentPipeline* pipeline);

int snepx_ti_enrichment_geoip(const char* ip, SNEPPXTEnrichmentStageResult* result);
int snepx_ti_enrichment_asn_lookup(uint32_t asn, SNEPPXTEnrichmentStageResult* result);
int snepx_ti_enrichment_rdns(const char* ip, SNEPPXTEnrichmentStageResult* result);
int snepx_ti_enrichment_whois(const char* domain, SNEPPXTEnrichmentStageResult* result);
int snepx_ti_enrichment_ssl_certificate(const char* host, uint16_t port, SNEPPXTEnrichmentStageResult* result);
int snepx_ti_enrichment_ja3_hash(const uint8_t* client_hello, size_t ch_len, uint8_t* ja3_out, size_t* ja3_len);
int snepx_ti_enrichment_yara_scan(const uint8_t* data, size_t data_len, const uint8_t* yara_rules, size_t rules_len, uint8_t** matches, uint32_t* num_matches);

int snepx_ti_enrichment_pipeline_destroy(SNEPPXTEnrichmentPipeline* pipeline);

#endif
