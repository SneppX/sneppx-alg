/*
 * SneppX native firewall policy engine.
 *
 * This is an embeddable packet-metadata policy engine. It evaluates rules
 * supplied by a network adapter; it does not install OS packet-filter hooks.
 */
#ifndef SNEPPX_NEURAL_CORE_SECURITY_FIREWALL_H
#define SNEPPX_NEURAL_CORE_SECURITY_FIREWALL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_FW_MAX_RULES 256u
#define SNEPPX_FW_ADDRESS_BYTES 16u

#define SNEPPX_FW_OK 0
#define SNEPPX_FW_ERROR_INVALID (-1)
#define SNEPPX_FW_ERROR_CAPACITY (-2)
#define SNEPPX_FW_ERROR_NOT_FOUND (-3)

#define SNEPPX_FW_IPV4 4u
#define SNEPPX_FW_IPV6 6u

#define SNEPPX_FW_PROTOCOL_ANY 0u
#define SNEPPX_FW_PROTOCOL_ICMP 1u
#define SNEPPX_FW_PROTOCOL_TCP 6u
#define SNEPPX_FW_PROTOCOL_UDP 17u
#define SNEPPX_FW_PROTOCOL_ICMPV6 58u

typedef enum SNEPPXFirewallDirection {
    SNEPPX_FW_INBOUND = 0,
    SNEPPX_FW_OUTBOUND = 1,
    SNEPPX_FW_BOTH_DIRECTIONS = 2
} SNEPPXFirewallDirection;

typedef enum SNEPPXFirewallAction {
    SNEPPX_FW_DENY = 0,
    SNEPPX_FW_ALLOW = 1
} SNEPPXFirewallAction;

typedef struct SNEPPXFirewallRule {
    uint32_t id;                 /* 0 asks the engine to assign an ID. */
    uint32_t priority;           /* Larger values are evaluated first. */
    uint8_t family;              /* SNEPPX_FW_IPV4 or SNEPPX_FW_IPV6. */
    uint8_t protocol;            /* IP protocol number; 0 matches any. */
    uint8_t direction;           /* Inbound, outbound, or both. */
    uint8_t action;               /* Deny or allow. */
    uint8_t source_prefix;
    uint8_t destination_prefix;
    uint8_t match_source;
    uint8_t match_destination;
    uint8_t match_destination_port;
    uint8_t enabled;
    uint16_t destination_port_first;
    uint16_t destination_port_last;
    uint8_t source[SNEPPX_FW_ADDRESS_BYTES];
    uint8_t destination[SNEPPX_FW_ADDRESS_BYTES];
} SNEPPXFirewallRule;

typedef struct SNEPPXFirewallPacket {
    uint8_t family;
    uint8_t protocol;
    uint8_t direction;
    uint8_t reserved;
    uint16_t source_port;
    uint16_t destination_port;
    uint32_t payload_bytes;
    uint8_t source[SNEPPX_FW_ADDRESS_BYTES];
    uint8_t destination[SNEPPX_FW_ADDRESS_BYTES];
} SNEPPXFirewallPacket;

typedef struct SNEPPXFirewallResult {
    uint8_t action;
    uint8_t matched_rule;
    uint16_t reserved;
    uint32_t rule_id;
} SNEPPXFirewallResult;

typedef struct SNEPPXFirewallStats {
    uint64_t allowed_packets;
    uint64_t blocked_packets;
    uint64_t allowed_bytes;
    uint64_t blocked_bytes;
    uint64_t rules_evaluated;
} SNEPPXFirewallStats;

/* Initialize a zero-initialized engine exactly once before use. */
typedef struct SNEPPXFirewallEngine {
    void *internal;
} SNEPPXFirewallEngine;

int SNEPPX_firewall_init(SNEPPXFirewallEngine *engine);
void SNEPPX_firewall_destroy(SNEPPXFirewallEngine *engine);
int SNEPPX_firewall_set_default_action(
    SNEPPXFirewallEngine *engine, SNEPPXFirewallAction action);
int SNEPPX_firewall_add_rule(
    SNEPPXFirewallEngine *engine,
    const SNEPPXFirewallRule *rule,
    uint32_t *assigned_rule_id);
int SNEPPX_firewall_remove_rule(
    SNEPPXFirewallEngine *engine, uint32_t rule_id);
int SNEPPX_firewall_evaluate(
    SNEPPXFirewallEngine *engine,
    const SNEPPXFirewallPacket *packet,
    SNEPPXFirewallResult *result);
int SNEPPX_firewall_get_stats(
    SNEPPXFirewallEngine *engine, SNEPPXFirewallStats *stats);
size_t SNEPPX_firewall_rule_count(SNEPPXFirewallEngine *engine);

#ifdef __cplusplus
}
#endif

#endif
