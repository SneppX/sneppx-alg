#ifndef SNEPPX_THREAT_INTEL_COLLECTOR_NETWORK_H
#define SNEPPX_THREAT_INTEL_COLLECTOR_NETWORK_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TI_MAX_PEERS 1024
#define SNEPPX_TI_BGP_TIMEOUT_MS 30000
#define SNEPPX_TI_DNS_BUF_SIZE 4096

/*
 * SNEPPX - Collector Network
 *
 * WHAT
 *   Collector Network.
 *
 * CONCEPT
 *   Provides the Collector Network.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TI_SOURCE_BGP_DUMP,
    SNEPPX_TI_SOURCE_DNS_PASSIVE,
    SNEPPX_TI_SOURCE_NETFLOW,
    SNEPPX_TI_SOURCE_DARKNET,
    SNEPPX_TI_SOURCE_HONEYPOT,
    SNEPPX_TI_SOURCE_OSINT_FEED,
    SNEPPX_TI_SOURCE_SSL_CERTIFICATE,
    SNEPPX_TI_SOURCE_WHOIS,
    SNEPPX_TI_SOURCE_RDNS,
    SNEPPX_TI_SOURCE_ABUSEIPDB,
    SNEPPX_TI_SOURCE_VIRUSTOTAL,
    SNEPPX_TI_SOURCE_SHODAN,
    SNEPPX_TI_SOURCE_CENSYS,
    SNEPPX_TI_SOURCE_GREYNOISE,
    SNEPPX_TI_SOURCE_ALIENVAULT_OTX,
    SNEPPX_TI_SOURCE_MISP
} SNEPPXTICollectorSource;

typedef enum {
    SNEPPX_TI_FEED_FORMAT_JSON,
    SNEPPX_TI_FEED_FORMAT_CSV,
    SNEPPX_TI_FEED_FORMAT_STIX,
    SNEPPX_TI_FEED_FORMAT_TAXII,
    SNEPPX_TI_FEED_FORMAT_MISP,
    SNEPPX_TI_FEED_FORMAT_OPENIOC,
    SNEPPX_TI_FEED_FORMAT_YARA_RULE
} SNEPPXTIFeedFormat;

typedef struct {
    SNEPPXTICollectorSource source;
    SNEPPXTIFeedFormat format;
    char* feed_url;
    size_t url_len;
    char* api_key;
    size_t api_key_len;
    uint32_t polling_interval_seconds;
    uint32_t timeout_ms;
    uint8_t tls_verify : 1;
    uint8_t proxy_enabled : 1;
    char* proxy_host;
    size_t proxy_host_len;
    uint16_t proxy_port;
    uint8_t* ca_bundle;
    size_t ca_bundle_len;
    uint8_t* client_cert;
    size_t client_cert_len;
    uint8_t* client_key;
    size_t client_key_len;
} SNEPPXTICollectorConfig;

typedef struct {
    uint8_t* raw_data;
    size_t raw_data_len;
    uint64_t collection_timestamp_ns;
    uint32_t ioc_count;
    uint32_t error_count;
    uint32_t duplicate_count;
    uint8_t* feed_name;
    size_t feed_name_len;
    uint8_t* feed_version;
    size_t feed_version_len;
    SNEPPXTIFeedFormat format;
    uint8_t parsed : 1;
    uint8_t enriched : 1;
    uint8_t deduplicated : 1;
} SNEPPXTICollectedData;

typedef struct {
    uint8_t* collector_id;
    size_t collector_id_len;
    SNEPPXTICollectorConfig* configs;
    uint32_t num_configs;
    uint64_t total_collected;
    uint64_t total_errors;
    uint64_t total_duplicates;
    uint8_t running : 1;
    uint8_t schedule_active : 1;
} SNEPPXTICollectorEngine;

int snepx_ti_collector_init(SNEPPXTICollectorEngine* engine);
int snepx_ti_collector_add_source(SNEPPXTICollectorEngine* engine, const SNEPPXTICollectorConfig* config);
int snepx_ti_collector_remove_source(SNEPPXTICollectorEngine* engine, uint32_t config_index);
int snepx_ti_collector_poll(SNEPPXTICollectorEngine* engine, uint32_t config_index, SNEPPXTICollectedData* data);
int snepx_ti_collector_poll_all(SNEPPXTICollectorEngine* engine, SNEPPXTICollectedData** data_array, uint32_t* data_count);

int snepx_ti_collector_parse_stix(SNEPPXTICollectedData* data);
int snepx_ti_collector_parse_taxii(SNEPPXTICollectedData* data);
int snepx_ti_collector_parse_misp(SNEPPXTICollectedData* data);
int snepx_ti_collector_parse_openioc(SNEPPXTICollectedData* data);
int snepx_ti_collector_parse_yara_rule(SNEPPXTICollectedData* data);

int snepx_ti_collector_bgp_dump(SNEPPXTICollectedData* data, const char* collector_url);
int snepx_ti_collector_dns_passive(SNEPPXTICollectedData* data, const char* dns_server, uint16_t port);
int snepx_ti_collector_darknet(SNEPPXTICollectedData* data, const char* interface, uint16_t* ports, uint32_t num_ports);

int snepx_ti_collector_deduplicate(SNEPPXTICollectedData* data);
int snepx_ti_collector_enrich(SNEPPXTICollectedData* data);
int snepx_ti_collector_destroy(SNEPPXTICollectedData* data);
int snepx_ti_collector_engine_destroy(SNEPPXTICollectorEngine* engine);

#endif
