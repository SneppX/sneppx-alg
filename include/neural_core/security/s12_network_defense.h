#ifndef SNEPPX_S12_NETWORK_DEFENSE_H
#define SNEPPX_S12_NETWORK_DEFENSE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_IDS_MAX_SIGNATURES 65536
#define SNEPPX_WAF_MAX_RULES 32768
#define SNEPPX_HONEYPOT_MAX_SERVICES 128
#define SNEPPX_SANDBOX_MAX_PROCESSES 256
#define SNEPPX_DEFENDER_MAX_POLICIES 512

/*
 * SNEPPX - S12 Network Defense
 *
 * WHAT
 *   S12 Network Defense.
 *
 * CONCEPT
 *   Provides the S12 Network Defense.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_NET_ACTION_ALLOW,
    SNEPPX_NET_ACTION_DROP,
    SNEPPX_NET_ACTION_REJECT,
    SNEPPX_NET_ACTION_RATE_LIMIT,
    SNEPPX_NET_ACTION_CAPTURE,
    SNEPPX_NET_ACTION_REDIRECT_HONEYPOT,
    SNEPPX_NET_ACTION_QUARANTINE,
    SNEPPX_NET_ACTION_THROTTLE
} SNEPPXNetAction;

typedef enum {
    SNEPPX_SIG_TYPE_EXACT,
    SNEPPX_SIG_TYPE_REGEX,
    SNEPPX_SIG_TYPE_PATTERN,
    SNEPPX_SIG_TYPE_ANOMALY,
    SNEPPX_SIG_TYPE_BEHAVIORAL,
    SNEPPX_SIG_TYPE_ML
} SNEPPXSigType;

typedef struct {
    uint32_t sig_id;
    SNEPPXSigType sig_type;
    uint8_t pattern[256];
    size_t pattern_len;
    uint32_t priority;
    SNEPPXNetAction action;
    uint32_t category_id;
    uint64_t expiry_epoch;
    char description[256];
    uint8_t severity;
    uint8_t mitre_attack_id[32];
} SNEPPXIdsSignature;

typedef struct {
    uint32_t num_signatures;
    SNEPPXIdsSignature signatures[SNEPPX_IDS_MAX_SIGNATURES];
    uint8_t* pattern_matcher_state;
    size_t pattern_matcher_size;
    uint64_t packets_processed;
    uint64_t alerts_raised;
    uint64_t false_positives;
} SNEPPXIdsEngine;

typedef struct {
    char param_name[64];
    char param_pattern[256];
    SNEPPXNetAction action;
    uint32_t rate_limit;
    uint8_t enabled : 1;
    uint8_t log_match : 1;
    uint8_t block_on_match : 1;
} SNEPPXWafRule;

typedef struct {
    uint32_t num_rules;
    SNEPPXWafRule rules[SNEPPX_WAF_MAX_RULES];
    uint64_t requests_blocked;
    uint64_t requests_passed;
    uint64_t total_inspection_time_ns;
} SNEPPXWafEngine;

typedef struct {
    uint16_t service_port;
    char service_banner[1024];
    uint8_t service_protocol;
    uint32_t interaction_count;
    uint64_t first_seen;
    uint64_t last_seen;
    uint8_t* captured_payloads;
    size_t captured_payloads_len;
    SNEPPXNetAction action_on_interact;
} SNEPPXHoneypotService;

typedef struct {
    uint32_t num_services;
    SNEPPXHoneypotService services[SNEPPX_HONEYPOT_MAX_SERVICES];
    uint64_t total_interactions;
    uint64_t attacker_ips_tracked;
    uint32_t* attacker_ip_set;
} SNEPPXHoneypotNetwork;

int snepx_ids_engine_init(SNEPPXIdsEngine* ids);
int snepx_ids_add_signature(SNEPPXIdsEngine* ids, const SNEPPXIdsSignature* sig);
int snepx_ids_remove_signature(SNEPPXIdsEngine* ids, uint32_t sig_id);
int snepx_ids_inspect_packet(SNEPPXIdsEngine* ids, const uint8_t* packet, size_t packet_len, SNEPPXNetAction* action, uint32_t* matched_sig_id);
int snepx_ids_train_ml(SNEPPXIdsEngine* ids, const uint8_t* benign_samples, size_t benign_count, const uint8_t* malicious_samples, size_t malicious_count);
int snepx_ids_export_stats(SNEPPXIdsEngine* ids, uint8_t* stats_json, size_t* json_len);

int snepx_waf_init(SNEPPXWafEngine* waf);
int snepx_waf_add_rule(SNEPPXWafEngine* waf, const SNEPPXWafRule* rule);
int snepx_waf_inspect_request(SNEPPXWafEngine* waf, const char* uri, const char* method, const char* headers, size_t headers_len, const uint8_t* body, size_t body_len, SNEPPXNetAction* action);
int snepx_waf_update_rule_set(SNEPPXWafEngine* waf, const char* rule_set_json, size_t rule_set_len);

int snepx_honeypot_init(SNEPPXHoneypotNetwork* honeypot, const uint16_t* ports, size_t num_ports);
int snepx_honeypot_start_service(SNEPPXHoneypotNetwork* honeypot, uint16_t port, const char* banner);
int snepx_honeypot_stop_service(SNEPPXHoneypotNetwork* honeypot, uint16_t port);
int snepx_honeypot_process_connection(SNEPPXHoneypotNetwork* honeypot, const uint8_t* src_ip, size_t ip_len, uint16_t src_port, uint16_t dst_port, const uint8_t* payload, size_t payload_len);
SNEPPXNetAction snepx_honeypot_get_action(SNEPPXHoneypotNetwork* honeypot, const uint8_t* src_ip, size_t ip_len);

// Active Defense
typedef struct {
    uint8_t decoy_token[64];
    uint32_t token_id;
    uint64_t deployment_time;
    uint8_t* trigger_script;
    size_t trigger_len;
    uint64_t triggered_count;
    uint32_t* triggered_by_ips;
} SNEPPXCanaryToken;

int snepx_canary_token_deploy(SNEPPXCanaryToken* token, const char* path, const uint8_t* script, size_t script_len);
int snepx_canary_token_check(SNEPPXCanaryToken* token);
int snepx_canary_token_alert(SNEPPXCanaryToken* token);

// Network Tarpit
typedef struct {
    uint16_t ports[32];
    uint32_t num_ports;
    uint64_t slow_response_delay_us;
    uint32_t max_connections;
    uint8_t* garbage_data;
    size_t garbage_len;
    uint64_t total_wasted_time_us;
} SNEPPXTarpit;

int snepx_tarpit_init(SNEPPXTarpit* tarpit, const uint16_t* ports, size_t num_ports);
int snepx_tarpit_accept_connection(SNEPPXTarpit* tarpit, int client_fd);
int snepx_tarpit_slow_response(SNEPPXTarpit* tarpit, int client_fd);
int snepx_tarpit_send_garbage(SNEPPXTarpit* tarpit, int client_fd);

#endif
