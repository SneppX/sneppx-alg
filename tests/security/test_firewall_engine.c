#include "neural_core/security/sneppx_firewall.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
        return 1; \
    } \
} while (0)

int main(void) {
    SNEPPXFirewallEngine engine = {0};
    SNEPPXFirewallPacket packet;
    SNEPPXFirewallRule rule;
    SNEPPXFirewallResult result;
    SNEPPXFirewallStats stats;
    uint32_t allow_id = 0u;
    uint32_t deny_id = 0u;

    CHECK(SNEPPX_firewall_init(&engine) == SNEPPX_FW_OK, "engine initializes");
    memset(&packet, 0, sizeof(packet));
    packet.family = SNEPPX_FW_IPV4;
    packet.protocol = SNEPPX_FW_PROTOCOL_TCP;
    packet.direction = SNEPPX_FW_INBOUND;
    packet.source_port = 51000u;
    packet.destination_port = 8080u;
    packet.payload_bytes = 512u;
    packet.source[0] = 192u;
    packet.source[1] = 168u;
    packet.source[2] = 10u;
    packet.source[3] = 42u;

    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK,
          "default policy evaluates");
    CHECK(result.action == SNEPPX_FW_DENY && result.matched_rule == 0u,
          "default policy denies unmatched traffic");

    memset(&rule, 0, sizeof(rule));
    rule.family = SNEPPX_FW_IPV4;
    rule.protocol = SNEPPX_FW_PROTOCOL_TCP;
    rule.direction = SNEPPX_FW_INBOUND;
    rule.action = SNEPPX_FW_ALLOW;
    rule.priority = 10u;
    rule.source_prefix = 24u;
    rule.match_source = 1u;
    rule.enabled = 1u;
    rule.source[0] = 192u;
    rule.source[1] = 168u;
    rule.source[2] = 10u;
    CHECK(SNEPPX_firewall_add_rule(&engine, &rule, &allow_id) == SNEPPX_FW_OK,
          "IPv4 allow rule added");

    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK,
          "IPv4 allow rule evaluates");
    CHECK(result.action == SNEPPX_FW_ALLOW && result.rule_id == allow_id,
          "IPv4 CIDR source rule allows matching packet");

    rule.id = 0u;
    rule.action = SNEPPX_FW_DENY;
    rule.priority = 10u;
    rule.source_prefix = 32u;
    rule.source[3] = 42u;
    CHECK(SNEPPX_firewall_add_rule(&engine, &rule, &deny_id) == SNEPPX_FW_OK,
          "IPv4 host deny rule added");
    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK,
          "priority tie evaluates");
    CHECK(result.action == SNEPPX_FW_DENY && result.rule_id == deny_id,
          "deny wins equal-priority tie");

    CHECK(SNEPPX_firewall_remove_rule(&engine, deny_id) == SNEPPX_FW_OK,
          "deny rule removed");
    CHECK(SNEPPX_firewall_remove_rule(&engine, deny_id) == SNEPPX_FW_ERROR_NOT_FOUND,
          "unknown rule removal is reported");
    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK &&
          result.action == SNEPPX_FW_ALLOW, "allow rule remains active");

    memset(&rule, 0, sizeof(rule));
    rule.family = SNEPPX_FW_IPV4;
    rule.protocol = SNEPPX_FW_PROTOCOL_TCP;
    rule.direction = SNEPPX_FW_INBOUND;
    rule.action = SNEPPX_FW_DENY;
    rule.priority = 20u;
    rule.match_destination_port = 1u;
    rule.destination_port_first = 8000u;
    rule.destination_port_last = 8100u;
    rule.enabled = 1u;
    CHECK(SNEPPX_firewall_add_rule(&engine, &rule, NULL) == SNEPPX_FW_OK,
          "destination port rule added");
    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK &&
          result.action == SNEPPX_FW_DENY, "destination port range blocks packet");

    memset(&packet, 0, sizeof(packet));
    packet.family = SNEPPX_FW_IPV6;
    packet.protocol = SNEPPX_FW_PROTOCOL_UDP;
    packet.direction = SNEPPX_FW_INBOUND;
    packet.destination_port = 5353u;
    packet.payload_bytes = 128u;
    packet.source[0] = 0x20u;
    packet.source[1] = 0x01u;
    packet.source[2] = 0x0Du;
    packet.source[3] = 0xB8u;

    memset(&rule, 0, sizeof(rule));
    rule.family = SNEPPX_FW_IPV6;
    rule.protocol = SNEPPX_FW_PROTOCOL_UDP;
    rule.direction = SNEPPX_FW_INBOUND;
    rule.action = SNEPPX_FW_ALLOW;
    rule.priority = 30u;
    rule.source_prefix = 32u;
    rule.match_source = 1u;
    rule.enabled = 1u;
    rule.source[0] = 0x20u;
    rule.source[1] = 0x01u;
    rule.source[2] = 0x0Du;
    rule.source[3] = 0xB8u;
    CHECK(SNEPPX_firewall_add_rule(&engine, &rule, NULL) == SNEPPX_FW_OK,
          "IPv6 CIDR rule added");
    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK &&
          result.action == SNEPPX_FW_ALLOW, "IPv6 CIDR packet is allowed");

    CHECK(SNEPPX_firewall_get_stats(&engine, &stats) == SNEPPX_FW_OK,
          "statistics are readable");
    CHECK(stats.allowed_packets >= 2u && stats.blocked_packets >= 3u,
          "allow and deny packet counters increment");
    CHECK(stats.allowed_bytes >= 128u && stats.blocked_bytes >= 512u,
          "byte counters increment");
    CHECK(SNEPPX_firewall_rule_count(&engine) == 3u,
          "rule count reflects add and remove operations");

    memset(&rule, 0, sizeof(rule));
    rule.family = SNEPPX_FW_IPV4;
    rule.source_prefix = 33u;
    rule.enabled = 1u;
    CHECK(SNEPPX_firewall_add_rule(&engine, &rule, NULL) == SNEPPX_FW_ERROR_INVALID,
          "invalid CIDR prefix is rejected");

    CHECK(SNEPPX_firewall_set_default_action(&engine, SNEPPX_FW_ALLOW) == SNEPPX_FW_OK,
          "default action can be changed");
    memset(&packet, 0, sizeof(packet));
    packet.family = SNEPPX_FW_IPV4;
    packet.direction = SNEPPX_FW_OUTBOUND;
    CHECK(SNEPPX_firewall_evaluate(&engine, &packet, &result) == SNEPPX_FW_OK &&
          result.action == SNEPPX_FW_ALLOW && result.matched_rule == 0u,
          "configured default allow applies only when no rule matches");

    SNEPPX_firewall_destroy(&engine);
    CHECK(engine.internal == NULL, "destroy clears engine handle");
    puts("PASS: native firewall policy engine");
    return 0;
}
