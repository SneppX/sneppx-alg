#include "ddos_mitigation.h"
#include "cryptographic_random_generator.h"
#include <string.h>
#include <time.h>

#define DDOS_MAX_CONNECTIONS 65536
#define DDOS_BUCKET_RATE 1000
#define DDOS_BUCKET_CAPACITY 10000
#define DDOS_WINDOW_SEC 60
#define DDOS_SYN_THRESHOLD 500

/*
 * SNEPPX - Ddos Mitigation
 *
 * WHAT
 *   Ddos Mitigation.
 *
 * CONCEPT
 *   Provides the Ddos Mitigation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    time_t first_seen;
    time_t last_seen;
    uint64_t packet_count;
    uint64_t byte_count;
    uint8_t state;
    uint8_t flags;
} ddos_flow_t;

typedef struct {
    uint32_t ip;
    uint64_t syn_count;
    uint64_t total_count;
    time_t window_start;
    uint8_t blocked;
    time_t block_until;
    double request_rate;
} ddos_source_t;

static ddos_flow_t flows[DDOS_MAX_CONNECTIONS];
static int flow_count = 0;
static ddos_source_t sources[DDOS_MAX_CONNECTIONS];
static int source_count = 0;
static ddos_config_t active_config = {
    .syn_flood_threshold = DDOS_SYN_THRESHOLD,
    .conn_rate_threshold = 100,
    .bandwidth_threshold = 1000000,
    .block_duration = 300,
    .enable_adaptive = 1,
    .enable_app_layer = 1,
    .enable_syn_cookies = 1
};
static uint64_t total_packets_dropped = 0;
static uint64_t total_blocks_issued = 0;
static uint64_t total_sources_blocked = 0;

static int find_source(uint32_t ip) {
    for (int i = 0; i < source_count; i++)
        if (sources[i].ip == ip) return i;
    return -1;
}

static int add_source(uint32_t ip) {
    if (source_count >= DDOS_MAX_CONNECTIONS) return -1;
    sources[source_count].ip = ip;
    sources[source_count].syn_count = 0;
    sources[source_count].total_count = 0;
    sources[source_count].window_start = time(NULL);
    sources[source_count].blocked = 0;
    sources[source_count].block_until = 0;
    sources[source_count].request_rate = 0;
    return source_count++;
}

/**
 * @brief Perform Ddos Detect Syn Flood.
 *
 * @param src_ip [in] Src Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_detect_syn_flood(uint32_t src_ip, uint32_t dst_ip) {
    int idx = find_source(src_ip);
    if (idx < 0) idx = add_source(src_ip);
    if (idx < 0) return 0;
    sources[idx].syn_count++;
    sources[idx].total_count++;
    time_t now = time(NULL);
    if (now - sources[idx].window_start > DDOS_WINDOW_SEC) {
        sources[idx].syn_count = 0;
        sources[idx].total_count = 0;
        sources[idx].window_start = now;
    }
    if (sources[idx].blocked && now < sources[idx].block_until) return 1;
    if (sources[idx].blocked && now >= sources[idx].block_until) {
        sources[idx].blocked = 0;
        sources[idx].syn_count = 0;
    }
    if (sources[idx].syn_count > active_config.syn_flood_threshold) {
        sources[idx].blocked = 1;
        sources[idx].block_until = now + active_config.block_duration;
        total_blocks_issued++;
        total_sources_blocked++;
        return 1;
    }
    double rate = (double)sources[idx].total_count / (double)(now - sources[idx].window_start + 1);
    sources[idx].request_rate = rate;
    if (rate > active_config.conn_rate_threshold && active_config.enable_adaptive) {
        sources[idx].blocked = 1;
        sources[idx].block_until = now + active_config.block_duration;
        total_blocks_issued++;
        total_sources_blocked++;
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Ddos Detect App Layer.
 *
 * @param src_ip [in] Src Ip value.
 * @param payload [in] Payload value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_detect_app_layer(uint32_t src_ip, const uint8_t *payload, size_t len) {
    if (!active_config.enable_app_layer) return 0;
    int idx = find_source(src_ip);
    if (idx < 0) idx = add_source(src_ip);
    if (idx < 0) return 0;
    time_t now = time(NULL);
    if (sources[idx].blocked && now < sources[idx].block_until) return 1;
    if (len > 0) {
        int slow_attack = 0;
        for (size_t i = 0; i < len && i < 64; i++) {
            if (payload[i] == 0) slow_attack++;
        }
        if (slow_attack > (int)(len > 64 ? 64 : len) / 2) {
            sources[idx].blocked = 1;
            sources[idx].block_until = now + active_config.block_duration;
            total_blocks_issued++;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Ddos Generate Syn Cookie.
 *
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_ddos_generate_syn_cookie(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    if (!active_config.enable_syn_cookies) return 0;
    time_t now = time(NULL);
    uint32_t cookie = src_ip ^ dst_ip ^ ((uint32_t)src_port << 16);
    cookie ^= (uint32_t)(now / 60) * 2654435761u;
    cookie ^= (uint32_t)(now % 60) * 2246822519u;
    return cookie & 0x00FFFFFF;
}

/**
 * @brief Perform Ddos Validate Syn Cookie.
 *
 * @param cookie [in] Cookie value.
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_validate_syn_cookie(uint32_t cookie, uint32_t src_ip, uint32_t dst_ip, uint16_t src_port) {
    if (!active_config.enable_syn_cookies) return 1;
    uint32_t expected = SNEPPX_ddos_generate_syn_cookie(src_ip, dst_ip, src_port);
    return (cookie == expected) ? 1 : 0;
}

/**
 * @brief Perform Ddos Apply Rate Limit.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_apply_rate_limit(uint32_t src_ip) {
    int idx = find_source(src_ip);
    if (idx < 0) idx = add_source(src_ip);
    if (idx < 0) return 0;
    time_t now = time(NULL);
    if (sources[idx].blocked && now < sources[idx].block_until) {
        total_packets_dropped++;
        return 1;
    }
    double rate = sources[idx].request_rate;
    if (rate > active_config.conn_rate_threshold * 2) {
        total_packets_dropped++;
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Ddos Track Connection.
 *
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 * @param src_port [in] Src Port value.
 * @param dst_port [in] Dst Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_track_connection(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port, uint64_t bytes) {
    if (flow_count < DDOS_MAX_CONNECTIONS) {
        flows[flow_count].src_ip = src_ip;
        flows[flow_count].dst_ip = dst_ip;
        flows[flow_count].src_port = src_port;
        flows[flow_count].dst_port = dst_port;
        flows[flow_count].first_seen = time(NULL);
        flows[flow_count].last_seen = time(NULL);
        flows[flow_count].packet_count = 1;
        flows[flow_count].byte_count = bytes;
        flows[flow_count].state = 1;
        flow_count++;
    }
    return 0;
}

/**
 * @brief Perform Ddos Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_get_stats(ddos_stats_t *stats) {
    if (!stats) return -1;
    stats->total_packets_dropped = total_packets_dropped;
    stats->total_blocks_issued = total_blocks_issued;
    stats->total_sources_blocked = total_sources_blocked;
    stats->active_flows = flow_count;
    stats->active_sources = source_count;
    stats->syn_flood_threshold = active_config.syn_flood_threshold;
    stats->conn_rate_threshold = active_config.conn_rate_threshold;
    stats->block_duration = active_config.block_duration;
    return 0;
}

/**
 * @brief Perform Ddos Update Config.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_update_config(const ddos_config_t *config) {
    if (!config) return -1;
    memcpy(&active_config, config, sizeof(ddos_config_t));
    return 0;
}

/**
 * @brief Reset Ddos.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_reset(void) {
    flow_count = 0;
    source_count = 0;
    total_packets_dropped = 0;
    total_blocks_issued = 0;
    total_sources_blocked = 0;
    memset(flows, 0, sizeof(flows));
    memset(sources, 0, sizeof(sources));
    return 0;
}
