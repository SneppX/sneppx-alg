#ifndef SNEPPX_DDOS_MITIGATION_H
#define SNEPPX_DDOS_MITIGATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

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
    int syn_flood_threshold;
    int conn_rate_threshold;
    int bandwidth_threshold;
    int block_duration;
    int enable_adaptive;
    int enable_app_layer;
    int enable_syn_cookies;
} ddos_config_t;

typedef struct {
    uint64_t total_packets_dropped;
    uint64_t total_blocks_issued;
    uint64_t total_sources_blocked;
    int active_flows;
    int active_sources;
    int syn_flood_threshold;
    int conn_rate_threshold;
    int block_duration;
} ddos_stats_t;

/**
 * @brief Perform Ddos Detect Syn Flood.
 *
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_detect_syn_flood(uint32_t src_ip, uint32_t dst_ip);
/**
 * @brief Perform Ddos Detect App Layer.
 *
 * @param src_ip [in] Src Ip value.
 * @param payload [in] Payload value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_detect_app_layer(uint32_t src_ip, const uint8_t *payload, size_t len);
/**
 * @brief Perform Ddos Generate Syn Cookie.
 *
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 * @param src_port [in] Src Port value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_ddos_generate_syn_cookie(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port);
/**
 * @brief Perform Ddos Validate Syn Cookie.
 *
 * @param cookie [in] Cookie value.
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 * @param src_port [in] Src Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_validate_syn_cookie(uint32_t cookie, uint32_t src_ip, uint32_t dst_ip, uint16_t src_port);
/**
 * @brief Perform Ddos Apply Rate Limit.
 *
 * @param src_ip [in] Src Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_apply_rate_limit(uint32_t src_ip);
/**
 * @brief Perform Ddos Track Connection.
 *
 * @param src_ip [in] Src Ip value.
 * @param dst_ip [in] Dst Ip value.
 * @param src_port [in] Src Port value.
 * @param dst_port [in] Dst Port value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_track_connection(uint32_t src_ip, uint32_t dst_ip, uint16_t src_port, uint16_t dst_port, uint64_t bytes);
/**
 * @brief Perform Ddos Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_get_stats(ddos_stats_t *stats);
/**
 * @brief Perform Ddos Update Config.
 *
 * @param config [in] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_update_config(const ddos_config_t *config);
/**
 * @brief Reset Ddos.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ddos_reset(void);


#ifdef __cplusplus
}
#endif
#endif
