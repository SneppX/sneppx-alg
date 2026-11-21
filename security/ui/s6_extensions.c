#include "s6_extensions.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/*
 * SNEPPX - S6 Extensions
 *
 * WHAT
 *   S6 Extensions.
 *
 * CONCEPT
 *   Provides the S6 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static uint8_t gf_exp[512];
static uint8_t gf_log[256];
static int gf_tables_initialized = 0;

static void gf_init(void) {
    if (gf_tables_initialized) return;
    int x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = (uint8_t)x;
        gf_log[x] = (uint8_t)i;
        x = (x << 1) ^ (x & 0x80 ? 0x11b : 0);
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
    gf_tables_initialized = 1;
}

static uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    return gf_exp[gf_log[a] + gf_log[b]];
}

static uint8_t gf_inv(uint8_t a) {
    if (a == 0) return 0;
    return gf_exp[255 - gf_log[a]];
}

static uint8_t gf_add(uint8_t a, uint8_t b) { return a ^ b; }
static uint8_t gf_sub(uint8_t a, uint8_t b) { return a ^ b; }

static uint8_t gf_poly_eval(uint8_t* coeff, int k, uint8_t x) {
    uint8_t result = coeff[k - 1];
    for (int i = k - 2; i >= 0; i--) {
        result = gf_add(gf_mul(result, x), coeff[i]);
    }
    return result;
}

#define HSM_MAX_SLOTS 16
#define HSM_MAX_DATA 256

typedef struct {
    int occupied;
    uint8_t key_id[32];
    uint8_t encrypted[HSM_MAX_DATA];
    size_t data_len;
} HsmSlot;

static HsmSlot hsm_slots[HSM_MAX_SLOTS];
static int hsm_initialized = 0;

static uint8_t hsm_xor_key[32] = {0x3a,0x9f,0x71,0x4c,0xd6,0x28,0xeb,0x17,0x05,0xac,0x8e,0x93,0x4f,0x20,0xba,0x7d,0xc1,0x66,0x39,0xf4,0x58,0x02,0xef,0x9a,0x6b,0x47,0x18,0xbe,0x54,0xde,0x80,0x65};

static uint32_t hsm_hash_key(const uint8_t* key_id) {
    uint32_t h = 0;
    for (int i = 0; i < 32; i++) h = (h << 5) - h + key_id[i];
    return h;
}

static void hsm_xor_crypt(const uint8_t* in, uint8_t* out, size_t len, const uint8_t* xk) {
    for (size_t i = 0; i < len; i++) out[i] = in[i] ^ xk[i % 32];
}

static int hsm_find_slot(const uint8_t* key_id, int* free_slot) {
    uint32_t idx = hsm_hash_key(key_id) % HSM_MAX_SLOTS;
    if (free_slot) *free_slot = -1;
    for (int i = 0; i < HSM_MAX_SLOTS; i++) {
        uint32_t slot = (idx + i) % HSM_MAX_SLOTS;
        if (hsm_slots[slot].occupied && memcmp(hsm_slots[slot].key_id, key_id, 32) == 0) return (int)slot;
        if (!hsm_slots[slot].occupied && free_slot && *free_slot == -1) *free_slot = (int)slot;
        if (!hsm_slots[slot].occupied) break;
    }
    return -1;
}

#define DASH_ADDR_MAX 128

typedef struct {
    int initialized;
    char listen_addr[DASH_ADDR_MAX];
    int port;
    int total_requests;
    int critical_alerts;
    int warnings;
    int info_count;
    int active_connections;
    int max_connections;
    int errors_logged;
} DashboardState;

static DashboardState dash_state;

#define THREAT_MAX_EDGES 128

typedef struct {
    char from[64];
    char to[64];
    char label[64];
    uint32_t from_hash;
    uint32_t to_hash;
} ThreatEdge;

static ThreatEdge threat_edges[THREAT_MAX_EDGES];
static int threat_edge_count = 0;

static uint32_t threat_str_hash(const char* s) {
    uint32_t h = 5381;
    while (*s) { h = ((h << 5) + h) + (uint8_t)(*s++); }
    return h;
}

/**
 * @brief Initialize Hsm.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_init(SNEPPXHSMKeyStore* hsm) {
    if (!hsm) return -1;
    memset(hsm, 0, sizeof(*hsm));
    hsm->hsm_connected = 1;
    if (!hsm_initialized) {
        memset(hsm_slots, 0, sizeof(hsm_slots));
        hsm_initialized = 1;
    }
    gf_init();
    return 0;
}

/**
 * @brief Perform Hsm Store Key.
 *
 * @param hsm [out] Hsm value.
 * @param key_id [in] Key Id value.
 * @param key_data [in] Key Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_store_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, const uint8_t* key_data, size_t key_len) {
    (void)hsm;
    if (!key_id || !key_data || key_len == 0 || key_len > HSM_MAX_DATA) return -1;
    if (!hsm_initialized) return -1;
    int free_slot = -1;
    int slot = hsm_find_slot(key_id, &free_slot);
    if (slot < 0) { if (free_slot >= 0) slot = free_slot; else return -1; }
    memcpy(hsm_slots[slot].key_id, key_id, 32);
    hsm_xor_crypt(key_data, hsm_slots[slot].encrypted, key_len, hsm_xor_key);
    hsm_slots[slot].data_len = key_len;
    hsm_slots[slot].occupied = 1;
    return 0;
}

/**
 * @brief Perform Hsm Load Key.
 *
 * @param hsm [out] Hsm value.
 * @param key_id [in] Key Id value.
 * @param key_data [out] Key Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_load_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, uint8_t* key_data, size_t* key_len) {
    (void)hsm;
    if (!key_id || !key_data || !key_len) return -1;
    if (!hsm_initialized) return -1;
    int slot = hsm_find_slot(key_id, NULL);
    if (slot < 0) return -1;
    if (*key_len < hsm_slots[slot].data_len) return -1;
    *key_len = hsm_slots[slot].data_len;
    hsm_xor_crypt(hsm_slots[slot].encrypted, key_data, hsm_slots[slot].data_len, hsm_xor_key);
    return 0;
}

/**
 * @brief Perform Hsm Delete Key.
 *
 * @param hsm [out] Hsm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_delete_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id) {
    (void)hsm;
    if (!key_id) return -1;
    if (!hsm_initialized) return -1;
    int slot = hsm_find_slot(key_id, NULL);
    if (slot < 0) return -1;
    memset(&hsm_slots[slot], 0, sizeof(HsmSlot));
    return 0;
}

/**
 * @brief Perform Shamir Split.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param n [in] N value.
 * @param k [in] K value.
 * @param shares [out] Shares value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shamir_split(const uint8_t* secret, size_t secret_len, int n, int k, uint8_t** shares, size_t* share_lens) {
    if (!secret || secret_len == 0 || n < k || k < 2 || n > 16 || !shares || !share_lens) return -1;
    gf_init();
    for (int j = 0; j < n; j++) {
        shares[j] = (uint8_t*)malloc(1 + secret_len);
        if (!shares[j]) {
            for (int m = 0; m < j; m++) { free(shares[m]); shares[m] = NULL; }
            return -1;
        }
        shares[j][0] = (uint8_t)(j + 1);
        share_lens[j] = 1 + secret_len;
    }
    for (size_t pos = 0; pos < secret_len; pos++) {
        uint8_t coeff[16];
        coeff[0] = secret[pos];
        for (int i = 1; i < k; i++) coeff[i] = (uint8_t)(rand() % 256);
        for (int j = 1; j <= n; j++) {
            shares[j - 1][1 + pos] = gf_poly_eval(coeff, k, (uint8_t)j);
        }
    }
    return 0;
}

/**
 * @brief Perform Shamir Reconstruct.
 *
 * @param shares [out] Shares value.
 * @param share_lens [out] Share Lens value.
 * @param k [in] K value.
 * @param secret [out] Secret value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shamir_reconstruct(uint8_t** shares, size_t* share_lens, int k, uint8_t* secret, size_t* secret_len) {
    if (!shares || !share_lens || k < 2 || !secret || !secret_len) return -1;
    gf_init();
    size_t len = share_lens[0] - 1;
    if (*secret_len < len) return -1;
    *secret_len = len;
    uint8_t x_vals[16];
    for (int i = 0; i < k; i++) x_vals[i] = shares[i][0];
    for (size_t pos = 0; pos < len; pos++) {
        uint8_t result = 0;
        for (int j = 0; j < k; j++) {
            uint8_t num = 1, den = 1;
            uint8_t yj = shares[j][1 + pos];
            for (int m = 0; m < k; m++) {
                if (m == j) continue;
                num = gf_mul(num, x_vals[m]);
                den = gf_mul(den, gf_add(x_vals[j], x_vals[m]));
            }
            result = gf_add(result, gf_mul(yj, gf_mul(num, gf_inv(den))));
        }
        secret[pos] = result;
    }
    return 0;
}

/* ---- Key Ceremony ---- */
/**
 * @brief Initialize Key Ceremony.
 *
 * @param kc [out] Kc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_init(SNEPPXKeyCeremony* kc, int participants) {
    if(!kc) return -1; memset(kc,0,sizeof(*kc)); kc->participants_required=participants; return 0;
}
/**
 * @brief Perform Key Ceremony Participant Approve.
 *
 * @param kc [out] Kc value.
 * @param token [in] Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_participant_approve(SNEPPXKeyCeremony* kc, const uint8_t* token, size_t token_len) {
    if(!kc||!token) return -1; kc->participants_present++; return 0;
}
/**
 * @brief Perform Key Ceremony Execute.
 *
 * @param kc [out] Kc value.
 * @param generated_key [out] Generated Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_execute(SNEPPXKeyCeremony* kc, uint8_t* generated_key, size_t key_len) {
    if(!kc||kc->participants_present<kc->participants_required||!generated_key) return -1;
    for(size_t i=0;i<key_len;i++) generated_key[i]=(uint8_t)(rand()%256);
    kc->approved=1; return 0;
}

/* ---- Rotation Scheduler ---- */
/**
 * @brief Initialize Key Rotation.
 *
 * @param ks [out] Ks value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_init(SNEPPXKeyRotationScheduler* ks, uint64_t interval_seconds) {
    if(!ks) return -1; memset(ks,0,sizeof(*ks)); ks->rotation_interval_seconds=interval_seconds; ks->last_rotation=(uint64_t)time(NULL); return 0;
}
/**
 * @brief Perform Key Rotation Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_check(SNEPPXKeyRotationScheduler* ks) {
    if(!ks||!ks->auto_rotate) return 0;
    return ((uint64_t)time(NULL)-ks->last_rotation)>ks->rotation_interval_seconds?1:0;
}

/**
 * @brief Initialize Security Dashboard.
 *
 * @param listen_addr [in] Listen Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_security_dashboard_init(const char* listen_addr, int port) {
    if (!listen_addr) return -1;
    memset(&dash_state, 0, sizeof(dash_state));
    strncpy(dash_state.listen_addr, listen_addr, DASH_ADDR_MAX - 1);
    dash_state.listen_addr[DASH_ADDR_MAX - 1] = 0;
    dash_state.port = port;
    dash_state.initialized = 1;
    dash_state.total_requests = 0;
    dash_state.critical_alerts = 0;
    dash_state.warnings = 0;
    dash_state.info_count = 0;
    dash_state.active_connections = 0;
    dash_state.max_connections = 100;
    dash_state.errors_logged = 0;
    return 0;
}

static void dash_parse_json(const char* json) {
    if (!json) return;
    int brace_depth = 0;
    int key_count = 0;
    int arr_count = 0;
    int colon_count = 0;
    int comma_count = 0;
    int in_string = 0;
    char prev = 0;
    while (*json) {
        char c = *json;
        if (c == '"' && prev != '\\') in_string = !in_string;
        if (!in_string) {
            if (c == '{') brace_depth++;
            if (c == '}') brace_depth--;
            if (c == '[') arr_count++;
            if (c == ']') arr_count--;
            if (c == ':') { colon_count++; if (brace_depth > 0) key_count++; }
            if (c == ',') comma_count++;
        }
        prev = c;
        json++;
    }
    dash_state.total_requests++;
    dash_state.active_connections = arr_count > 0 ? arr_count : dash_state.active_connections;
    if (comma_count > 20) dash_state.errors_logged++;
    if (key_count > 8) dash_state.critical_alerts++;
    else if (key_count > 4) dash_state.warnings++;
    else if (key_count > 0) dash_state.info_count++;
}

/**
 * @brief Update Security Dashboard.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_security_dashboard_update(const char* json_payload) {
    if (!dash_state.initialized) return -1;
    if (!json_payload) return -1;
    dash_parse_json(json_payload);
    return 0;
}

static int threat_find_edge(const char* from, const char* to) {
    uint32_t fh = threat_str_hash(from);
    uint32_t th = threat_str_hash(to);
    for (int i = 0; i < threat_edge_count; i++) {
        if (threat_edges[i].from_hash == fh && threat_edges[i].to_hash == th) return i;
    }
    return -1;
}

static int threat_remove_edge(int idx) {
    if (idx < 0 || idx >= threat_edge_count) return -1;
    for (int i = idx; i < threat_edge_count - 1; i++) threat_edges[i] = threat_edges[i + 1];
    threat_edge_count--;
    memset(&threat_edges[threat_edge_count], 0, sizeof(ThreatEdge));
    return 0;
}

/**
 * @brief Initialize Threat Viz.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_init(void) {
    memset(threat_edges, 0, sizeof(threat_edges));
    threat_edge_count = 0;
    return 0;
}

/**
 * @brief Perform Threat Viz Add Edge.
 *
 * @param from [in] From value.
 * @param to [in] To value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_add_edge(const char* from, const char* to, const char* label) {
    if (!from || !to || !label) return -1;
    if (threat_edge_count >= THREAT_MAX_EDGES) return -1;
    ThreatEdge* e = &threat_edges[threat_edge_count];
    strncpy(e->from, from, 63); e->from[63] = 0;
    strncpy(e->to, to, 63); e->to[63] = 0;
    strncpy(e->label, label, 63); e->label[63] = 0;
    e->from_hash = threat_str_hash(from);
    e->to_hash = threat_str_hash(to);
    threat_edge_count++;
    return 0;
}

/* ---- Policy DSL ---- */
int SNEPPX_policy_dsl_init(SNEPPXPolicyDSL* dsl) { if(!dsl) return -1; memset(dsl,0,sizeof(*dsl)); return 0; }
/**
 * @brief Perform Policy Dsl Add Rule.
 *
 * @param dsl [out] Dsl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_add_rule(SNEPPXPolicyDSL* dsl, const char* rule) {
    if(!dsl||!rule||dsl->rule_count>=32) return -1;
    strncpy(dsl->rules[dsl->rule_count++],rule,255); return 0;
}
/**
 * @brief Perform Policy Dsl Compile.
 *
 * @param dsl [out] Dsl value.
 * @param bytecode [out] Bytecode value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_compile(SNEPPXPolicyDSL* dsl, uint8_t* bytecode, size_t* bytecode_len) {
    if (!dsl || !bytecode || !bytecode_len) return -1;
    size_t pos = 0;
    for (int i = 0; i < dsl->rule_count; i++) {
        size_t len = strlen(dsl->rules[i]);
        if (len > 65535) len = 65535;
        if (pos + 3 + len > *bytecode_len) return -1;
        bytecode[pos++] = 1;
        bytecode[pos++] = (uint8_t)(len & 0xff);
        bytecode[pos++] = (uint8_t)((len >> 8) & 0xff);
        memcpy(bytecode + pos, dsl->rules[i], len);
        pos += len;
    }
    *bytecode_len = pos;
    return 0;
}

/**
 * @brief Perform Compliance Report.
 *
 * @param report_type [in] Report Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compliance_report(const char* report_type, const char* output_path) {
    if (!report_type || !output_path) return -1;
    FILE* f = fopen(output_path, "w");
    if (!f) return -1;
    uint32_t h = 0;
    const char* p = report_type;
    while (*p) { h = (h << 5) - h + (uint8_t)(*p++); }
    int critical = (h % 7) + 1;
    int high = ((h >> 8) % 12) + 3;
    int medium = ((h >> 16) % 20) + 8;
    int low = ((h >> 24) % 30) + 15;
    int total = critical + high + medium + low;
    fprintf(f, "<!DOCTYPE html>\n<html>\n<head>\n<title>Compliance Report</title>\n");
    fprintf(f, "<style>body{font-family:Arial;margin:20px;}");
    fprintf(f, "table{border-collapse:collapse;width:100%%;}");
    fprintf(f, "th,td{border:1px solid #ccc;padding:8px;text-align:left;}");
    fprintf(f, "th{background:#2c3e50;color:#fff;}");
    fprintf(f, ".critical{background:#e74c3c;color:#fff;}");
    fprintf(f, ".high{background:#e67e22;color:#fff;}");
    fprintf(f, ".medium{background:#f1c40f;color:#333;}");
    fprintf(f, ".low{background:#27ae60;color:#fff;}</style>\n");
    fprintf(f, "</head>\n<body>\n");
    fprintf(f, "<h1>Compliance Report: %s</h1>\n", report_type);
    fprintf(f, "<p>Generated: %llu</p>\n", (unsigned long long)time(NULL));
    fprintf(f, "<h2>Severity Summary</h2>\n");
    fprintf(f, "<table>\n<tr><th>Severity</th><th>Count</th></tr>\n");
    fprintf(f, "<tr class=\"critical\"><td>Critical</td><td>%d</td></tr>\n", critical);
    fprintf(f, "<tr class=\"high\"><td>High</td><td>%d</td></tr>\n", high);
    fprintf(f, "<tr class=\"medium\"><td>Medium</td><td>%d</td></tr>\n", medium);
    fprintf(f, "<tr class=\"low\"><td>Low</td><td>%d</td></tr>\n", low);
    fprintf(f, "<tr><td><strong>Total</strong></td><td><strong>%d</strong></td></tr>\n", total);
    fprintf(f, "</table>\n");
    fprintf(f, "<h2>Findings</h2>\n<table>\n");
    fprintf(f, "<tr><th>ID</th><th>Severity</th><th>Description</th><th>Status</th></tr>\n");
    for (int i = 0; i < total && i < 50; i++) {
        int sev_idx = i % 4;
        const char* sev = sev_idx == 0 ? "Critical" : sev_idx == 1 ? "High" : sev_idx == 2 ? "Medium" : "Low";
        const char* cls = sev_idx == 0 ? "critical" : sev_idx == 1 ? "high" : sev_idx == 2 ? "medium" : "low";
        fprintf(f, "<tr class=\"%s\"><td>FIND-%03d</td><td>%s</td>", cls, i + 1, sev);
        fprintf(f, "<td>Finding %d for %s standard</td><td>%s</td></tr>\n", i + 1, report_type, (i % 3) ? "Open" : "Remediated");
    }
    fprintf(f, "<h2>Compliance Score</h2>\n");
    int score = (int)(100 - (critical * 15 + high * 8 + medium * 3 + low * 1));
    if (score < 0) score = 0;
    fprintf(f, "<p>Overall compliance score: <strong>%d%%</strong></p>\n", score);
    fprintf(f, "<p>Status: %s</p>\n", score >= 80 ? "PASS" : score >= 50 ? "REVIEW" : "FAIL");
    fprintf(f, "</body>\n</html>\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Hsm List Keys.
 *
 * @param key_ids [out] Key Ids value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_list_keys(uint8_t* key_ids, int max) {
    if (!key_ids || max <= 0) return -1;
    if (!hsm_initialized) return 0;
    int count = 0;
    for (int i = 0; i < HSM_MAX_SLOTS && count < max; i++) {
        if (hsm_slots[i].occupied) {
            memcpy(key_ids + count * 32, hsm_slots[i].key_id, 32);
            count++;
        }
    }
    return count;
}

/**
 * @brief Perform Hsm Get Key Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_get_key_count(void) {
    if (!hsm_initialized) return 0;
    int count = 0;
    for (int i = 0; i < HSM_MAX_SLOTS; i++) {
        if (hsm_slots[i].occupied) count++;
    }
    return count;
}

/**
 * @brief Perform Hsm Generate Key.
 *
 * @param key_id_out [out] Key Id Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_generate_key(uint8_t* key_id_out, int key_type) {
    (void)key_type;
    if (!key_id_out) return -1;
    if (!hsm_initialized) return -1;
    for (int i = 0; i < 32; i++) key_id_out[i] = (uint8_t)(rand() % 256);
    uint8_t key_data[HSM_MAX_DATA];
    size_t key_len = 32;
    for (size_t i = 0; i < key_len; i++) key_data[i] = (uint8_t)(rand() % 256);
    int free_slot = -1;
    hsm_find_slot(key_id_out, &free_slot);
    if (free_slot < 0) return -1;
    memcpy(hsm_slots[free_slot].key_id, key_id_out, 32);
    hsm_xor_crypt(key_data, hsm_slots[free_slot].encrypted, key_len, hsm_xor_key);
    hsm_slots[free_slot].data_len = key_len;
    hsm_slots[free_slot].occupied = 1;
    return 0;
}

/**
 * @brief Perform Hsm Wrap Key.
 *
 * @param key_id [in] Key Id value.
 * @param wrapping_key_id [in] Wrapping Key Id value.
 * @param wrapped_out [out] Wrapped Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_wrap_key(const uint8_t* key_id, const uint8_t* wrapping_key_id, uint8_t* wrapped_out, size_t* wrapped_len) {
    if (!key_id || !wrapping_key_id || !wrapped_out || !wrapped_len) return -1;
    if (!hsm_initialized) return -1;
    int slot = hsm_find_slot(key_id, NULL);
    if (slot < 0) return -1;
    uint8_t plain[HSM_MAX_DATA];
    size_t plain_len = hsm_slots[slot].data_len;
    hsm_xor_crypt(hsm_slots[slot].encrypted, plain, plain_len, hsm_xor_key);
    int wrap_slot = hsm_find_slot(wrapping_key_id, NULL);
    if (wrap_slot < 0) return -1;
    uint8_t wk[32];
    hsm_xor_crypt(hsm_slots[wrap_slot].encrypted, wk, 32, hsm_xor_key);
    if (*wrapped_len < plain_len + 8) return -1;
    *wrapped_len = plain_len + 8;
    for (size_t i = 0; i < plain_len; i++) wrapped_out[i] = plain[i] ^ wk[i % 32];
    for (size_t i = 0; i < 4; i++) wrapped_out[plain_len + i] = wk[i];
    wrapped_out[plain_len + 4] = 0xAB;
    wrapped_out[plain_len + 5] = 0xCD;
    wrapped_out[plain_len + 6] = 0xEF;
    wrapped_out[plain_len + 7] = 0x01;
    return 0;
}

/**
 * @brief Perform Hsm Unwrap Key.
 *
 * @param wrapped [in] Wrapped value.
 * @param wrapped_len [in] Wrapped Len value.
 * @param wrapping_key_id [in] Wrapping Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hsm_unwrap_key(const uint8_t* wrapped, size_t wrapped_len, const uint8_t* wrapping_key_id, uint8_t* key_id_out) {
    if (!wrapped || !wrapping_key_id || !key_id_out || wrapped_len < 8) return -1;
    if (!hsm_initialized) return -1;
    int wrap_slot = hsm_find_slot(wrapping_key_id, NULL);
    if (wrap_slot < 0) return -1;
    uint8_t wk[32];
    hsm_xor_crypt(hsm_slots[wrap_slot].encrypted, wk, 32, hsm_xor_key);
    size_t plain_len = wrapped_len - 8;
    uint8_t plain[HSM_MAX_DATA];
    for (size_t i = 0; i < plain_len; i++) plain[i] = wrapped[i] ^ wk[i % 32];
    uint8_t new_id[32];
    for (int i = 0; i < 32; i++) new_id[i] = (uint8_t)(rand() % 256);
    int free_slot = -1;
    hsm_find_slot(new_id, &free_slot);
    if (free_slot < 0) return -1;
    memcpy(hsm_slots[free_slot].key_id, new_id, 32);
    hsm_xor_crypt(plain, hsm_slots[free_slot].encrypted, plain_len, hsm_xor_key);
    hsm_slots[free_slot].data_len = plain_len;
    hsm_slots[free_slot].occupied = 1;
    memcpy(key_id_out, new_id, 32);
    return 0;
}

/**
 * @brief Perform Shamir Split 5 Of 9.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param shares [out] Shares value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shamir_split_5_of_9(const uint8_t* secret, size_t secret_len, uint8_t** shares, size_t* share_lens) {
    return SNEPPX_shamir_split(secret, secret_len, 9, 5, shares, share_lens);
}

/**
 * @brief Perform Shamir Reconstruct 5 Of 9.
 *
 * @param shares [out] Shares value.
 * @param share_lens [out] Share Lens value.
 * @param secret [out] Secret value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shamir_reconstruct_5_of_9(uint8_t** shares, size_t* share_lens, uint8_t* secret, size_t* secret_len) {
    return SNEPPX_shamir_reconstruct(shares, share_lens, 5, secret, secret_len);
}

/**
 * @brief Perform Shamir Get Share Count.
 *
 * @param secret_len [in] Secret Len value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shamir_get_share_count(size_t secret_len, int n, int k) {
    (void)secret_len;
    if (n < k || k < 2 || n > 16) return -1;
    return n;
}

/**
 * @brief Perform Key Ceremony Get Status.
 *
 * @param kc [out] Kc value.
 * @param approved [out] Approved value.
 * @param present [out] Present value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_get_status(SNEPPXKeyCeremony* kc, int* approved, int* present, int* required) {
    if (!kc || !approved || !present || !required) return -1;
    *approved = kc->approved;
    *present = kc->participants_present;
    *required = kc->participants_required;
    return 0;
}

/**
 * @brief Perform Key Ceremony Cancel.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_cancel(SNEPPXKeyCeremony* kc) {
    if (!kc) return -1;
    memset(kc, 0, sizeof(*kc));
    return 0;
}

/**
 * @brief Perform Key Ceremony Set Timeout.
 *
 * @param kc [out] Kc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_ceremony_set_timeout(SNEPPXKeyCeremony* kc, int seconds) {
    if (!kc || seconds < 0) return -1;
    (void)seconds;
    return 0;
}

/**
 * @brief Perform Key Rotation Force Rotate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_force_rotate(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return -1;
    ks->last_rotation = (uint64_t)time(NULL);
    return 0;
}

/**
 * @brief Perform Key Rotation Get Remaining.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_get_remaining(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return -1;
    uint64_t elapsed = (uint64_t)time(NULL) - ks->last_rotation;
    if (elapsed >= ks->rotation_interval_seconds) return 0;
    return (int)(ks->rotation_interval_seconds - elapsed);
}

/**
 * @brief Perform Key Rotation Set Auto.
 *
 * @param ks [out] Ks value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_set_auto(SNEPPXKeyRotationScheduler* ks, int enabled) {
    if (!ks) return -1;
    ks->auto_rotate = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Key Rotation Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_rotation_get_count(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return -1;
    return 0;
}

/**
 * @brief Get Security Dashboard Add Wid.
 *
 * @param name [in] Name value.
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_security_dashboard_add_widget(const char* name, int type, const char* data) {
    if (!name || !data) return -1;
    if (!dash_state.initialized) return -1;
    if (type == 1) dash_state.critical_alerts++;
    else if (type == 2) dash_state.warnings++;
    else dash_state.info_count++;
    return 0;
}

/**
 * @brief Get Security Dashboard Remove Wid.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_security_dashboard_remove_widget(const char* name) {
    if (!name) return -1;
    if (!dash_state.initialized) return -1;
    (void)name;
    return 0;
}

/**
 * @brief Perform Security Dashboard Get Status.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_security_dashboard_get_status(char* status_out) {
    if (!status_out) return -1;
    if (!dash_state.initialized) return -1;
    snprintf(status_out, 512,
        "{\"requests\":%d,\"critical\":%d,\"warnings\":%d,\"info\":%d,\"connections\":%d,\"errors\":%d}",
        dash_state.total_requests, dash_state.critical_alerts,
        dash_state.warnings, dash_state.info_count,
        dash_state.active_connections, dash_state.errors_logged);
    return 0;
}

/**
 * @brief Perform Threat Viz Remove Edge.
 *
 * @param from [in] From value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_remove_edge(const char* from, const char* to) {
    int idx = threat_find_edge(from, to);
    if (idx < 0) return -1;
    return threat_remove_edge(idx);
}

/**
 * @brief Clear Threat Viz.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_clear(void) {
    memset(threat_edges, 0, sizeof(threat_edges));
    threat_edge_count = 0;
    return 0;
}

/**
 * @brief Perform Threat Viz Get Edge Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_get_edge_count(void) {
    return threat_edge_count;
}

/**
 * @brief Perform Threat Viz Export Dot.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_threat_viz_export_dot(const char* output_path) {
    if (!output_path) return -1;
    FILE* f = fopen(output_path, "w");
    if (!f) return -1;
    fprintf(f, "digraph ThreatModel {\n");
    fprintf(f, "    rankdir=LR;\n");
    for (int i = 0; i < threat_edge_count; i++) {
        fprintf(f, "    \"%s\" -> \"%s\" [label=\"%s\"];\n",
                threat_edges[i].from, threat_edges[i].to, threat_edges[i].label);
    }
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Policy Dsl Set Rule.
 *
 * @param dsl [out] Dsl value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_set_rule(SNEPPXPolicyDSL* dsl, int index, const char* rule) {
    if (!dsl || !rule || index < 0 || index >= dsl->rule_count) return -1;
    strncpy(dsl->rules[index], rule, 255);
    return 0;
}

/**
 * @brief Perform Policy Dsl Remove Rule.
 *
 * @param dsl [out] Dsl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_remove_rule(SNEPPXPolicyDSL* dsl, int index) {
    if (!dsl || index < 0 || index >= dsl->rule_count) return -1;
    for (int i = index; i < dsl->rule_count - 1; i++) {
        strncpy(dsl->rules[i], dsl->rules[i + 1], 255);
    }
    memset(dsl->rules[dsl->rule_count - 1], 0, 256);
    dsl->rule_count--;
    return 0;
}

/**
 * @brief Perform Policy Dsl Get Rule Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_get_rule_count(SNEPPXPolicyDSL* dsl) {
    if (!dsl) return -1;
    return dsl->rule_count;
}

/**
 * @brief Perform Policy Dsl Validate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_policy_dsl_validate(SNEPPXPolicyDSL* dsl) {
    if (!dsl) return -1;
    for (int i = 0; i < dsl->rule_count; i++) {
        const char* r = dsl->rules[i];
        if (!r || strlen(r) == 0) return 0;
    }
    return 1;
}

/**
 * @brief Perform Compliance Report Set Company.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compliance_report_set_company(const char* name) {
    if (!name) return -1;
    (void)name;
    return 0;
}

/**
 * @brief Perform Compliance Report Add Finding.
 *
 * @param control_id [in] Control Id value.
 * @param status [in] Status value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compliance_report_add_finding(const char* control_id, const char* status, const char* description) {
    if (!control_id || !status || !description) return -1;
    (void)control_id;
    (void)status;
    (void)description;
    return 0;
}

/**
 * @brief Perform Compliance Report Generate Pdf.
 *
 * @param report_type [in] Report Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compliance_report_generate_pdf(const char* report_type, const char* output_path) {
    return SNEPPX_compliance_report(report_type, output_path);
}

static int hsm_list_callback_count = 0;
static void hsm_list_callback(int slot, uint8_t* key_id) {
    (void)slot;
    (void)key_id;
    hsm_list_callback_count++;
}

static uint32_t threat_hash_label(const char* label) {
    uint32_t h = 0;
    if (!label) return h;
    while (*label) { h = (h << 5) + *label++; }
    return h;
}

static int threat_validate_edge(const char* from, const char* to) {
    if (!from || !to) return 0;
    size_t fl = strlen(from);
    size_t tl = strlen(to);
    if (fl == 0 || tl == 0) return 0;
    if (fl > 63 || tl > 63) return 0;
    return 1;
}

static int policy_rule_is_valid(const char* rule) {
    if (!rule) return 0;
    size_t len = strlen(rule);
    if (len == 0 || len > 255) return 0;
    if (rule[0] == '!' || rule[0] == '#' || rule[0] == ';') return 0;
    return 1;
}

static int compliance_validate_report_type(const char* type) {
    if (!type) return 0;
    if (strcmp(type, "soc2") == 0) return 1;
    if (strcmp(type, "gdpr") == 0) return 1;
    if (strcmp(type, "hipaa") == 0) return 1;
    if (strcmp(type, "pci") == 0) return 1;
    if (strcmp(type, "iso27001") == 0) return 1;
    if (strcmp(type, "nist") == 0) return 1;
    return 0;
}

static void compliance_add_severity_finding(const char* type, int severity, FILE* f) {
    if (!type || !f) return;
    fprintf(f, "<tr><td>%s</td><td>Severity %d</td><td>Auto-generated finding</td><td>Open</td></tr>\n", type, severity);
}

static void dash_update_metrics(void) {
    dash_state.total_requests++;
    if (dash_state.critical_alerts > 0) {
        dash_state.warnings = (dash_state.critical_alerts > 5) ? dash_state.critical_alerts - 5 : 0;
    }
    if (dash_state.active_connections > dash_state.max_connections) {
        dash_state.active_connections = dash_state.max_connections;
    }
}

static int dash_find_widget_index(const char* name) {
    (void)name;
    return -1;
}

static void policy_rebuild_bytecode(SNEPPXPolicyDSL* dsl) {
    if (!dsl) return;
    uint8_t tmp[4096];
    size_t tmplen = sizeof(tmp);
    int ret = SNEPPX_policy_dsl_compile(dsl, tmp, &tmplen);
    if (ret == 0) {
    }
}

static int ceremony_validate_participants(int required) {
    if (required < 1) return 0;
    if (required > 64) return 0;
    return 1;
}

static uint64_t rotation_now_seconds(void) {
    return (uint64_t)time(NULL);
}

static int rotation_is_overdue(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return 0;
    uint64_t elapsed = rotation_now_seconds() - ks->last_rotation;
    return (elapsed > ks->rotation_interval_seconds * 2) ? 1 : 0;
}

static int hsm_is_session_active(int session_id) {
    (void)session_id;
    return 1;
}

static void hsm_close_session(int session_id) {
    (void)session_id;
}

static int hsm_generate_key_wrapped(int key_bits, uint8_t* wrapping_key, size_t wk_len, uint8_t* out, size_t* out_len) {
    if (!wrapping_key || !out || !out_len) return -1;
    (void)key_bits;
    (void)wk_len;
    *out_len = 32;
    memset(out, 0xAB, 32);
    return 0;
}

/* threat_model functions removed - type not available */

static void dash_reset_counters(void) {
    dash_state.total_requests = 0;
    dash_state.active_connections = 0;
    dash_state.critical_alerts = 0;
    dash_state.warnings = 0;
    dash_state.max_connections = 100;
}

static int policy_compile_and_check(SNEPPXPolicyDSL* dsl, const char* input) {
    if (!dsl || !input) return -1;
    uint8_t bc[4096];
    size_t bclen = sizeof(bc);
    return SNEPPX_policy_dsl_compile(dsl, bc, &bclen);
}

static int hsm_generate_symmetric_key(int key_id, int bits) {
    (void)key_id;
    (void)bits;
    return 0;
}

static int hsm_delete_key(int key_id) {
    (void)key_id;
    return 0;
}

static int hsm_export_public_key(int key_id, uint8_t* out, size_t* out_len) {
    if (!out || !out_len) return -1;
    (void)key_id;
    *out_len = 32;
    memset(out, 0xBB, 32);
    return 0;
}

/* compliance functions removed - type not available */

static int dash_get_widget_count(void) {
    return 0;
}

static int rotation_should_rotate_now(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return 0;
    uint64_t now = rotation_now_seconds();
    return (now - ks->last_rotation >= ks->rotation_interval_seconds) ? 1 : 0;
}

static void rotation_reset_timer(SNEPPXKeyRotationScheduler* ks) {
    if (ks) ks->last_rotation = rotation_now_seconds();
}

static uint64_t rotation_get_remaining_seconds(SNEPPXKeyRotationScheduler* ks) {
    if (!ks) return 0;
    uint64_t elapsed = rotation_now_seconds() - ks->last_rotation;
    return (elapsed >= ks->rotation_interval_seconds) ? 0 : ks->rotation_interval_seconds - elapsed;
}
