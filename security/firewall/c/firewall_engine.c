#include "neural_core/security/sneppx_firewall.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef struct SNEPPXFirewallInternal {
    atomic_bool lock;
    SNEPPXFirewallRule rules[SNEPPX_FW_MAX_RULES];
    size_t rule_count;
    uint32_t next_rule_id;
    uint8_t default_action;
    SNEPPXFirewallStats stats;
} SNEPPXFirewallInternal;

static void firewall_lock(SNEPPXFirewallInternal *state) {
    while (atomic_exchange_explicit(&state->lock, 1, memory_order_acquire)) {
        /* Short critical sections; no allocation or I/O occurs while locked. */
    }
}

static void firewall_unlock(SNEPPXFirewallInternal *state) {
    atomic_store_explicit(&state->lock, 0, memory_order_release);
}

static size_t address_size(uint8_t family) {
    if (family == SNEPPX_FW_IPV4) return 4u;
    if (family == SNEPPX_FW_IPV6) return 16u;
    return 0u;
}

static int valid_rule(const SNEPPXFirewallRule *rule) {
    size_t bytes;
    unsigned bits;
    if (rule == NULL) return 0;
    bytes = address_size(rule->family);
    if (bytes == 0u) return 0;
    bits = (unsigned)(bytes * 8u);
    if (rule->source_prefix > bits || rule->destination_prefix > bits) return 0;
    if (rule->direction > SNEPPX_FW_BOTH_DIRECTIONS) return 0;
    if (rule->action != SNEPPX_FW_DENY && rule->action != SNEPPX_FW_ALLOW) return 0;
    if (rule->match_source > 1u || rule->match_destination > 1u ||
        rule->match_destination_port > 1u || rule->enabled > 1u) return 0;
    if (rule->match_destination_port &&
        rule->destination_port_first > rule->destination_port_last) return 0;
    return 1;
}

static int valid_packet(const SNEPPXFirewallPacket *packet) {
    if (packet == NULL || address_size(packet->family) == 0u) return 0;
    if (packet->direction != SNEPPX_FW_INBOUND &&
        packet->direction != SNEPPX_FW_OUTBOUND) return 0;
    return 1;
}

static int prefix_matches(
    const uint8_t *address, const uint8_t *network, unsigned prefix_bits) {
    unsigned full_bytes = prefix_bits / 8u;
    unsigned remaining_bits = prefix_bits % 8u;
    if (full_bytes > 0u && memcmp(address, network, full_bytes) != 0) return 0;
    if (remaining_bits > 0u) {
        uint8_t mask = (uint8_t)(0xFFu << (8u - remaining_bits));
        if ((address[full_bytes] & mask) != (network[full_bytes] & mask)) return 0;
    }
    return 1;
}

static int rule_matches(
    const SNEPPXFirewallRule *rule, const SNEPPXFirewallPacket *packet) {
    if (!rule->enabled || rule->family != packet->family) return 0;
    if (rule->protocol != SNEPPX_FW_PROTOCOL_ANY &&
        rule->protocol != packet->protocol) return 0;
    if (rule->direction != SNEPPX_FW_BOTH_DIRECTIONS &&
        rule->direction != packet->direction) return 0;
    if (rule->match_source &&
        !prefix_matches(packet->source, rule->source, rule->source_prefix)) return 0;
    if (rule->match_destination &&
        !prefix_matches(packet->destination, rule->destination,
                        rule->destination_prefix)) return 0;
    if (rule->match_destination_port &&
        (packet->destination_port < rule->destination_port_first ||
         packet->destination_port > rule->destination_port_last)) return 0;
    return 1;
}

int SNEPPX_firewall_init(SNEPPXFirewallEngine *engine) {
    SNEPPXFirewallInternal *state;
    if (engine == NULL) return SNEPPX_FW_ERROR_INVALID;
    if (engine->internal != NULL) return SNEPPX_FW_ERROR_INVALID;
    state = (SNEPPXFirewallInternal *)calloc(1u, sizeof(*state));
    if (state == NULL) return SNEPPX_FW_ERROR_CAPACITY;
    atomic_init(&state->lock, 0);
    state->next_rule_id = 1u;
    state->default_action = SNEPPX_FW_DENY;
    engine->internal = state;
    return SNEPPX_FW_OK;
}

void SNEPPX_firewall_destroy(SNEPPXFirewallEngine *engine) {
    if (engine == NULL) return;
    free(engine->internal);
    engine->internal = NULL;
}

int SNEPPX_firewall_set_default_action(
    SNEPPXFirewallEngine *engine, SNEPPXFirewallAction action) {
    SNEPPXFirewallInternal *state;
    if (engine == NULL || engine->internal == NULL ||
        (action != SNEPPX_FW_DENY && action != SNEPPX_FW_ALLOW)) {
        return SNEPPX_FW_ERROR_INVALID;
    }
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    state->default_action = (uint8_t)action;
    firewall_unlock(state);
    return SNEPPX_FW_OK;
}

int SNEPPX_firewall_add_rule(
    SNEPPXFirewallEngine *engine,
    const SNEPPXFirewallRule *rule,
    uint32_t *assigned_rule_id) {
    SNEPPXFirewallInternal *state;
    SNEPPXFirewallRule copy;
    size_t i;
    if (engine == NULL || engine->internal == NULL || !valid_rule(rule)) {
        return SNEPPX_FW_ERROR_INVALID;
    }
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    if (state->rule_count >= SNEPPX_FW_MAX_RULES) {
        firewall_unlock(state);
        return SNEPPX_FW_ERROR_CAPACITY;
    }
    copy = *rule;
    if (copy.id == 0u) {
        uint32_t candidate = state->next_rule_id;
        int found;
        do {
            found = 0;
            if (candidate == 0u) candidate = 1u;
            for (i = 0u; i < state->rule_count; ++i) {
                if (state->rules[i].id == candidate) {
                    found = 1;
                    candidate++;
                    break;
                }
            }
        } while (found);
        copy.id = candidate;
        state->next_rule_id = candidate + 1u;
    } else {
        for (i = 0u; i < state->rule_count; ++i) {
            if (state->rules[i].id == copy.id) {
                firewall_unlock(state);
                return SNEPPX_FW_ERROR_INVALID;
            }
        }
    }
    state->rules[state->rule_count++] = copy;
    if (assigned_rule_id != NULL) *assigned_rule_id = copy.id;
    firewall_unlock(state);
    return SNEPPX_FW_OK;
}

int SNEPPX_firewall_remove_rule(
    SNEPPXFirewallEngine *engine, uint32_t rule_id) {
    SNEPPXFirewallInternal *state;
    size_t i;
    if (engine == NULL || engine->internal == NULL || rule_id == 0u) {
        return SNEPPX_FW_ERROR_INVALID;
    }
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    for (i = 0u; i < state->rule_count; ++i) {
        if (state->rules[i].id == rule_id) {
            if (i + 1u < state->rule_count) {
                memmove(&state->rules[i], &state->rules[i + 1u],
                        (state->rule_count - i - 1u) * sizeof(state->rules[0]));
            }
            state->rule_count--;
            firewall_unlock(state);
            return SNEPPX_FW_OK;
        }
    }
    firewall_unlock(state);
    return SNEPPX_FW_ERROR_NOT_FOUND;
}

int SNEPPX_firewall_evaluate(
    SNEPPXFirewallEngine *engine,
    const SNEPPXFirewallPacket *packet,
    SNEPPXFirewallResult *result) {
    SNEPPXFirewallInternal *state;
    const SNEPPXFirewallRule *best = NULL;
    size_t i;
    uint8_t action;
    if (engine == NULL || engine->internal == NULL ||
        !valid_packet(packet) || result == NULL) {
        return SNEPPX_FW_ERROR_INVALID;
    }
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    for (i = 0u; i < state->rule_count; ++i) {
        const SNEPPXFirewallRule *candidate = &state->rules[i];
        state->stats.rules_evaluated++;
        if (!rule_matches(candidate, packet)) continue;
        if (best == NULL || candidate->priority > best->priority ||
            (candidate->priority == best->priority &&
             candidate->action == SNEPPX_FW_DENY &&
             best->action == SNEPPX_FW_ALLOW)) {
            best = candidate;
        }
    }
    action = best != NULL ? best->action : state->default_action;
    result->action = action;
    result->matched_rule = best != NULL ? 1u : 0u;
    result->reserved = 0u;
    result->rule_id = best != NULL ? best->id : 0u;
    if (action == SNEPPX_FW_ALLOW) {
        state->stats.allowed_packets++;
        state->stats.allowed_bytes += packet->payload_bytes;
    } else {
        state->stats.blocked_packets++;
        state->stats.blocked_bytes += packet->payload_bytes;
    }
    firewall_unlock(state);
    return SNEPPX_FW_OK;
}

int SNEPPX_firewall_get_stats(
    SNEPPXFirewallEngine *engine, SNEPPXFirewallStats *stats) {
    SNEPPXFirewallInternal *state;
    if (engine == NULL || engine->internal == NULL || stats == NULL) {
        return SNEPPX_FW_ERROR_INVALID;
    }
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    *stats = state->stats;
    firewall_unlock(state);
    return SNEPPX_FW_OK;
}

size_t SNEPPX_firewall_rule_count(SNEPPXFirewallEngine *engine) {
    SNEPPXFirewallInternal *state;
    size_t count;
    if (engine == NULL || engine->internal == NULL) return 0u;
    state = (SNEPPXFirewallInternal *)engine->internal;
    firewall_lock(state);
    count = state->rule_count;
    firewall_unlock(state);
    return count;
}
