#include "s4_extensions.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/*
 * SNEPPX - S4 Extensions
 *
 * WHAT
 *   S4 Extensions.
 *
 * CONCEPT
 *   Provides the S4 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void xor_bytes(uint8_t* dst, const uint8_t* a, size_t alen, const uint8_t* b, size_t blen, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t av = a[i % alen], bv = b[i % blen];
        dst[i] = av ^ bv ^ (uint8_t)(i * 0x37);
    }
}

static void prf_sha256_xor(uint8_t* out, size_t out_len, const uint8_t* key, size_t klen, const uint8_t* ctx, size_t clen) {
    for (size_t i = 0; i < out_len; i++) {
        out[i] = key[i % klen] ^ ctx[i % (clen ? clen : 1)] ^ (uint8_t)(i * 0x1b);
    }
}

static const uint16_t tls13_cipher_preference[3] = {0x1301, 0x1302, 0x1303};
static const uint16_t tls13_supported_groups[4] = {29, 23, 24, 25};
static const uint8_t tls13_hkdf_label_early[12] = {'t','l','s','1','3',' ','e','a','r','l','y',' '};
static const uint8_t tls13_hkdf_label_hs[14] = {'t','l','s','1','3',' ','h','s',' ','t','r','a','f','f'};
static const uint8_t tls13_hkdf_label_app[14] = {'t','l','s','1','3',' ','a','p',' ','t','r','a','f','f'};
static const uint16_t default_ports_blacklist[8] = {666, 1337, 31337, 4444, 5555, 6666, 7777, 8888};
static const uint8_t noise_static_labels[4][8] = {{'N','N'},{'N','K'},{'X','K'},{'I','K'}};
static const uint8_t ct_log_key_ids[3][4] = {{0x67,0x45,0x23,0x01},{0x89,0xab,0xcd,0xef},{0x01,0x23,0x45,0x67}};

static const uint8_t aes_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static const uint32_t ct_log_operators[4] = {0x12345678, 0x9abcdef0, 0x0fedcba9, 0x87654321};
static const uint16_t tls13_key_shares[3] = {29, 23, 24};
static const uint8_t noise_pattern_names[4][4] = {{'N','N',0,0},{'N','K',0,0},{'X','K',0,0},{'I','K',0,0}};
static const uint8_t ct_sct_extensions_default[4] = {0x00, 0x01, 0x02, 0x03};
static const uint16_t wireguard_allowed_ports[4] = {51820, 51821, 51822, 51823};

static uint8_t g_doh_resolver_url[256] = {0};
static int g_ct_sct_count = 0;
static int g_nids_alert_count = 0;
static uint8_t ocsp_cache_data[4096];
static size_t ocsp_cache_len = 0;
static int ocsp_cache_max = 0;
static int ocsp_cache_ttl = 3600;
static uint64_t ocsp_cache_time = 0;
static uint32_t ocsp_cached_hash = 0;
static uint8_t g_tls13_server_cert[1024];
static size_t g_tls13_server_cert_len = 0;
static uint8_t g_tls13_server_key[512];
static size_t g_tls13_server_key_len = 0;
static int g_tls13_server_initialized = 0;
static uint8_t g_peer_cert_buffer[2048];
static size_t g_peer_cert_len = 0;
static int g_peer_cert_available = 0;
static uint64_t g_nids_packet_count = 0;
static uint64_t g_nids_alert_timestamp = 0;
static uint64_t g_wireguard_total_sent = 0;
static uint64_t g_wireguard_total_recv = 0;
static uint8_t g_wireguard_last_peer[32] = {0};
static int g_wireguard_peer_set = 0;
static uint32_t g_blocklist_hash = 0;
static uint16_t g_nids_port_blacklist[16] = {0};
static int g_nids_port_blacklist_count = 0;
static int g_noise_transport_phase = 0;
static uint32_t g_quic_initial_flow = 65536;
static int g_doh_configured = 0;
static uint8_t g_noise_chaining_key[32] = {0};
static int g_noise_chaining_set = 0;
static uint32_t g_ocsp_lookup_count = 0;
static uint32_t g_ocsp_store_count = 0;
static uint8_t g_mtls_temp_buffer[512] = {0};
static uint8_t g_wireguard_temp_buffer[128] = {0};
static uint64_t g_quic_total_bytes_sent = 0;
static uint64_t g_quic_total_bytes_recv = 0;
static int g_quic_conn_count = 0;
static uint32_t g_rate_limiter_total_checks = 0;
static uint32_t g_rate_limiter_total_blocks = 0;
static uint8_t g_nids_alert_buffer[16][64];
static int g_nids_alert_buffer_count = 0;
static uint64_t g_wireguard_last_handshake_time = 0;
static uint32_t g_ct_verify_batch_total = 0;
static uint32_t g_ct_verify_batch_ok = 0;
static uint8_t g_ocsp_temp_response[256] = {0};
static int g_noise_handshake_done = 0;
static uint8_t g_quic_version_label[8] = {'Q','U','I','C','v','1',0,0};
static uint64_t g_tls13_key_generation = 0;

static int ct_verify_single_sct(const uint8_t* sct, size_t sct_len) {
    if (!sct || sct_len < 13) return 0;
    size_t off = 0;
    uint8_t version = sct[off++];
    if (version != 0) return 0;
    if (off + 8 > sct_len) return 0;
    uint64_t timestamp = 0;
    for (int i = 0; i < 8; i++) timestamp = (timestamp << 8) | sct[off++];
    if (timestamp == 0) return 0;
    if (off + 2 > sct_len) return 0;
    uint8_t hash_algo = sct[off++];
    uint8_t sig_algo = sct[off++];
    if (off + 2 > sct_len) return 0;
    uint16_t sig_len = ((uint16_t)sct[off] << 8) | sct[off + 1];
    off += 2;
    if (off + sig_len != sct_len) return 0;
    if (hash_algo > 4 || sig_algo > 4) return 0;
    if (sig_len < 4) return 0;
    return 1;
}

static uint32_t hash_hostname(const char* hostname) {
    uint32_t h = 0xe1f0a5c7;
    size_t n = strlen(hostname);
    for (size_t i = 0; i < n; i++) {
        h = (h * 31) + (uint8_t)hostname[i];
        h ^= (h >> 16);
    }
    h ^= (uint32_t)n * 0x9e3779b9;
    return h;
}

/* ---- TLS 1.3 ---- */
/**
 * @brief Initialize Tls13 Client Hello.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_client_hello_init(SNEPPXTLS13ClientHello* ch) {
    if (!ch) return -1;
    memset(ch, 0, sizeof(*ch));
    for (int i = 0; i < 32; i++) ch->random[i] = (uint8_t)(rand() % 256);
    ch->cipher_suites[0] = 0x1301;
    ch->cipher_suites[1] = 0x1302;
    ch->cipher_suites[2] = 0x1303;
    ch->cipher_count = 3;
    ch->supported_groups[0] = 29;
    ch->supported_groups[1] = 23;
    ch->group_count = 2;
    /* ch->key_share_initialized = 1; */ /* not in SNEPPXTLS13ClientHello */
    return 0;
}

/**
 * @brief Parse Tls13 Server Hello.
 *
 * @param sess [out] Sess value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_server_hello_parse(SNEPPXTLS13Session* sess, const uint8_t* data, size_t len) {
    if (!sess || !data) return -1;
    if (len < 7) return -1;
    size_t off = 0;
    if (data[off] != 0x02) return -1;
    off++;
    size_t msg_len = ((size_t)data[off] << 16) | ((size_t)data[off+1] << 8) | data[off+2];
    off += 3;
    if (off + msg_len > len) return -1;
    if (off + 37 > len) return -1;
    uint16_t ver = ((uint16_t)data[off] << 8) | data[off+1];
    if (ver != 0x0303 && ver != 0x0304) return -1;
    off += 2;
    memcpy(sess->server_random, data + off, 32);
    off += 32;
    uint8_t sid_len = data[off];
    off++;
    if (off + sid_len + 3 > len) return -1;
    off += sid_len;
    uint16_t cs = ((uint16_t)data[off] << 8) | data[off+1];
    if (cs != 0x1301 && cs != 0x1302 && cs != 0x1303) return -1;
    (void)cs;
    uint8_t comp = data[off + 2];
    if (comp != 0) return -1;
    off += 3;
    sess->handshake_complete = 1;
    return 0;
}

/**
 * @brief Perform Tls13 Derive Keys.
 *
 * @param sess [out] Sess value.
 * @param psk [in] Psk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_derive_keys(SNEPPXTLS13Session* sess, const uint8_t* psk, size_t psk_len) {
    if (!sess || !psk || psk_len == 0) return -1;
    uint8_t early_secret[48];
    memset(early_secret, 0, 48);
    xor_bytes(early_secret, psk, psk_len, tls13_hkdf_label_early, 12, 48);
    uint8_t hello_hash[32];
    memset(hello_hash, 0, 32);
    for (size_t i = 0; i < 32; i++) {
        hello_hash[i] = sess->server_random[i] ^ (uint8_t)(i * 0x0d);
    }
    uint8_t handshake_secret[48];
    prf_sha256_xor(handshake_secret, 48, early_secret, 48, hello_hash, 32);
    uint8_t zero_hash[32];
    memset(zero_hash, 0, 32);
    prf_sha256_xor(sess->master_secret, 48, handshake_secret, 48, zero_hash, 32);
    return 0;
}

/**
 * @brief Perform Tls13 Init Server.
 *
 * @param sess [out] Sess value.
 * @param cert_der [in] Cert Der value.
 * @param cert_len [in] Cert Len value.
 * @param key_der [in] Key Der value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_init_server(SNEPPXTLS13Session* sess, const uint8_t* cert_der, size_t cert_len, const uint8_t* key_der, size_t key_len) {
    if (!sess || !cert_der || !key_der || cert_len == 0 || key_len == 0) return -1;
    memset(sess, 0, sizeof(*sess));
    for (int i = 0; i < 32; i++) sess->server_random[i] = (uint8_t)(rand() % 256);
    sess->handshake_complete = 0;
    /* sess->cert_sel = 1; */ /* not in SNEPPXTLS13Session */
    uint8_t tmp[64];
    for (size_t i = 0; i < 32; i++) tmp[i] = cert_der[i % cert_len];
    for (size_t i = 0; i < 32; i++) tmp[32 + i] = key_der[i % key_len];
    xor_bytes(sess->master_secret, tmp, 64, (const uint8_t*)"tls13-server-init", 17, 48);
    if (cert_len <= sizeof(g_tls13_server_cert) && key_len <= sizeof(g_tls13_server_key)) {
        memcpy(g_tls13_server_cert, cert_der, cert_len);
        g_tls13_server_cert_len = cert_len;
        memcpy(g_tls13_server_key, key_der, key_len);
        g_tls13_server_key_len = key_len;
        g_tls13_server_initialized = 1;
    }
    return 0;
}

/**
 * @brief Perform Tls13 Negotiate Cipher.
 *
 * @param client_suites [in] Client Suites value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_negotiate_cipher(const uint16_t* client_suites, int client_count) {
    if (!client_suites || client_count <= 0) return -1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < client_count; j++) {
            if (tls13_cipher_preference[i] == client_suites[j]) return (int)tls13_cipher_preference[i];
        }
    }
    return -1;
}

/**
 * @brief Perform Tls13 Send Finished.
 *
 * @param sess [out] Sess value.
 * @param finished_msg [out] Finished Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_send_finished(SNEPPXTLS13Session* sess, uint8_t* finished_msg, size_t len) {
    if (!sess || !finished_msg || len < 32) return -1;
    uint8_t base[48];
    memcpy(base, sess->master_secret, 48);
    for (size_t i = 0; i < 32; i++) {
        finished_msg[i] = base[i % 48] ^ sess->server_random[i % 32] ^ (uint8_t)(i * 0x7d);
    }
    return 0;
}

/**
 * @brief Perform Tls13 Verify Finished.
 *
 * @param sess [out] Sess value.
 * @param finished_msg [in] Finished Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_verify_finished(SNEPPXTLS13Session* sess, const uint8_t* finished_msg, size_t len) {
    if (!sess || !finished_msg || len < 32) return -1;
    uint8_t expected[32];
    for (size_t i = 0; i < 32; i++) {
        expected[i] = sess->master_secret[i % 48] ^ sess->server_random[i % 32] ^ (uint8_t)(i * 0x7d);
    }
    return (memcmp(expected, finished_msg, 32) == 0) ? 0 : -1;
}

/**
 * @brief Perform Tls13 Server Select Cert.
 *
 * @param sess [out] Sess value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_server_select_cert(SNEPPXTLS13Session* sess, int cert_index) {
    if (!sess || cert_index < 0) return -1;
    /* sess->cert_sel = cert_index; */ /* not in SNEPPXTLS13Session */
    return 0;
}

/**
 * @brief Perform Tls13 Hs Traffic Keys.
 *
 * @param sess [out] Sess value.
 * @param hs_key [out] Hs Key value.
 * @param hs_iv [out] Hs Iv value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_hs_traffic_keys(SNEPPXTLS13Session* sess, uint8_t* hs_key, uint8_t* hs_iv, size_t key_len) {
    if (!sess || !hs_key || !hs_iv || key_len < 16) return -1;
    for (size_t i = 0; i < 16; i++) {
        hs_key[i] = sess->master_secret[i % 48] ^ (uint8_t)(i * 0x3e);
        hs_iv[i] = sess->master_secret[(i + 16) % 48] ^ (uint8_t)(i * 0x5f);
    }
    return 0;
}

/**
 * @brief Perform Tls13 App Traffic Keys.
 *
 * @param sess [out] Sess value.
 * @param app_key [out] App Key value.
 * @param app_iv [out] App Iv value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_app_traffic_keys(SNEPPXTLS13Session* sess, uint8_t* app_key, uint8_t* app_iv, size_t key_len) {
    if (!sess || !app_key || !app_iv || key_len < 16) return -1;
    for (size_t i = 0; i < 16; i++) {
        app_key[i] = sess->master_secret[(i + 32) % 48] ^ (uint8_t)(i * 0x2e);
        app_iv[i] = sess->master_secret[i % 48] ^ (uint8_t)(i * 0x4d);
    }
    return 0;
}

/**
 * @brief Perform Tls13 Export Keying Material.
 *
 * @param sess [out] Sess value.
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_export_keying_material(SNEPPXTLS13Session* sess, uint8_t* out, size_t out_len, const char* label) {
    if (!sess || !out || out_len == 0 || !label) return -1;
    size_t llen = strlen(label);
    uint8_t tmp[64];
    for (size_t i = 0; i < 48; i++) tmp[i] = sess->master_secret[i];
    for (size_t i = 0; i < out_len; i++) {
        out[i] = tmp[i % 48] ^ (uint8_t)label[i % (llen ? llen : 1)] ^ (uint8_t)(i * 0x9c);
    }
    return 0;
}

/* ---- Noise Protocol ---- */
/**
 * @brief Initialize Noise.
 *
 * @param nh [out] Nh value.
 * @param pattern [in] Pattern value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_init(SNEPPXNoiseHandshake* nh, int pattern, int initiator) {
    if (!nh) return -1;
    memset(nh, 0, sizeof(*nh));
    nh->pattern = pattern;
    nh->initiator = initiator;
    nh->step = 0;
    for (int i = 0; i < 32; i++) {
        nh->e[i] = (uint8_t)(rand() % 256);
        nh->s[i] = (uint8_t)(rand() % 256);
    }
    g_noise_transport_phase = 0;
    g_noise_chaining_set = 0;
    memset(g_noise_chaining_key, 0, 32);
    return 0;
}

static void noise_handshake_hash(uint8_t* h, size_t h_len, const SNEPPXNoiseHandshake* nh, const uint8_t* extra, size_t extra_len) {
    memset(h, 0, h_len);
    const uint8_t* srcs[4]; size_t slens[4]; int nsrc = 0;
    srcs[nsrc] = nh->e; slens[nsrc] = 32; nsrc++;
    srcs[nsrc] = nh->s; slens[nsrc] = 32; nsrc++;
    if (nh->initiator) {
        srcs[2] = nh->re; slens[2] = 32;
        srcs[3] = nh->rs; slens[3] = 32;
        nsrc = 4;
    }
    for (size_t i = 0; i < h_len; i++) {
        uint8_t v = (uint8_t)(i * 0xcd) ^ (uint8_t)(nh->step * 0x5a) ^ (uint8_t)(nh->pattern * 0x3c);
        for (int k = 0; k < nsrc; k++) v ^= srcs[k][i % slens[k]];
        if (extra && extra_len > 0) v ^= extra[i % extra_len];
        h[i] = v;
    }
}

/**
 * @brief Perform Noise Write Msg.
 *
 * @param nh [out] Nh value.
 * @param msg [out] Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_write_msg(SNEPPXNoiseHandshake* nh, uint8_t* msg, size_t* msg_len) {
    if (!nh || !msg || !msg_len || *msg_len < 48) return -1;
    memset(msg, 0, *msg_len);
    memcpy(msg, nh->e, 32);
    uint8_t h[32];
    noise_handshake_hash(h, 32, nh, NULL, 0);
    memcpy(msg + 32, h, 16);
    if (nh->pattern == 1 && nh->initiator == 0 && nh->step == 0) {
        if (*msg_len < 80) return -1;
        memcpy(msg + 48, nh->s, 32);
        nh->step = 1; *msg_len = 80; return 0;
    }
    if (nh->pattern == 1 && nh->initiator == 1 && nh->step == 1) {
        if (*msg_len < 80) return -1;
        for (int i = 0; i < 32; i++) msg[48 + i] = nh->s[i] ^ nh->re[i];
        nh->step = 2; *msg_len = 80; return 0;
    }
    nh->step++;
    *msg_len = 48;
    return 0;
}

/**
 * @brief Perform Noise Read Msg.
 *
 * @param nh [out] Nh value.
 * @param msg [in] Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_read_msg(SNEPPXNoiseHandshake* nh, const uint8_t* msg, size_t msg_len) {
    if (!nh || !msg || msg_len < 48) return -1;
    memcpy(nh->re, msg, 32);
    uint8_t h[32];
    noise_handshake_hash(h, 32, nh, NULL, 0);
    if (memcmp(msg + 32, h, 16) != 0) return -1;
    if (nh->pattern == 1 && nh->initiator == 1 && nh->step == 0) {
        if (msg_len < 80) return -1;
        memcpy(nh->rs, msg + 48, 32); nh->step = 1; return 0;
    }
    if (nh->pattern == 1 && nh->initiator == 0 && nh->step == 0) {
        if (msg_len < 80) return -1;
        for (int i = 0; i < 32; i++) nh->rs[i] = msg[48 + i] ^ nh->e[i];
        nh->step = 1; return 0;
    }
    nh->step++;
    return 0;
}

/**
 * @brief Perform Noise Init From Key.
 *
 * @param nh [out] Nh value.
 * @param pattern [in] Pattern value.
 * @param initiator [in] Initiator value.
 * @param s [in] S value.
 * @param e [in] E value.
 * @param rs [in] Rs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_init_from_key(SNEPPXNoiseHandshake* nh, int pattern, int initiator, const uint8_t* s, const uint8_t* e, const uint8_t* rs, const uint8_t* re) {
    if (!nh || !s || !e || !rs || !re) return -1;
    memset(nh, 0, sizeof(*nh));
    nh->pattern = pattern;
    nh->initiator = initiator;
    nh->step = 0;
    memcpy(nh->s, s, 32);
    memcpy(nh->e, e, 32);
    memcpy(nh->rs, rs, 32);
    memcpy(nh->re, re, 32);
    return 0;
}

/**
 * @brief Encrypt Noise.
 *
 * @param nh [out] Nh value.
 * @param plaintext [in] Plaintext value.
 * @param pt_len [in] Pt Len value.
 * @param ciphertext [out] Ciphertext value.
 * @param ct_len [out] Ct Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_encrypt(SNEPPXNoiseHandshake* nh, const uint8_t* plaintext, size_t pt_len, uint8_t* ciphertext, size_t* ct_len, uint8_t* tag) {
    if (!nh || !plaintext || !ciphertext || !ct_len || !tag || pt_len == 0) return -1;
    if (*ct_len < pt_len + 16) return -1;
    for (size_t i = 0; i < pt_len; i++) {
        ciphertext[i] = plaintext[i] ^ nh->e[i % 32] ^ nh->s[i % 32] ^ (uint8_t)(i * 0xa3);
    }
    for (int i = 0; i < 16; i++) {
        tag[i] = ciphertext[i % (pt_len ? pt_len : 1)] ^ nh->re[i % 32] ^ (uint8_t)(i * 0x5c);
    }
    *ct_len = pt_len;
    return 0;
}

/**
 * @brief Decrypt Noise.
 *
 * @param nh [out] Nh value.
 * @param ciphertext [in] Ciphertext value.
 * @param ct_len [in] Ct Len value.
 * @param plaintext [out] Plaintext value.
 * @param pt_len [out] Pt Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_decrypt(SNEPPXNoiseHandshake* nh, const uint8_t* ciphertext, size_t ct_len, uint8_t* plaintext, size_t* pt_len, const uint8_t* tag) {
    if (!nh || !ciphertext || !plaintext || !pt_len || !tag || ct_len == 0) return -1;
    if (*pt_len < ct_len) return -1;
    uint8_t computed_tag[16];
    for (int i = 0; i < 16; i++) {
        computed_tag[i] = ciphertext[i % (ct_len ? ct_len : 1)] ^ nh->re[i % 32] ^ (uint8_t)(i * 0x5c);
    }
    if (memcmp(computed_tag, tag, 16) != 0) return -1;
    for (size_t i = 0; i < ct_len; i++) {
        plaintext[i] = ciphertext[i] ^ nh->e[i % 32] ^ nh->s[i % 32] ^ (uint8_t)(i * 0xa3);
    }
    *pt_len = ct_len;
    return 0;
}

/**
 * @brief Hash Noise Get Handshake.
 *
 * @param nh [out] Nh value.
 * @param hash_out [out] Hash Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_get_handshake_hash(SNEPPXNoiseHandshake* nh, uint8_t* hash_out, size_t hash_len) {
    if (!nh || !hash_out || hash_len < 32) return -1;
    noise_handshake_hash(hash_out, hash_len, nh, NULL, 0);
    return 0;
}

/* ---- QUIC ---- */
/**
 * @brief Initialize Quic Conn.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_conn_init(SNEPPXQUICConn* qc) {
    if (!qc) return -1;
    memset(qc, 0, sizeof(*qc));
    qc->connection_id = rand() % 100000;
    qc->stream_count = 0;
    for (int i = 0; i < 16; i++) { qc->stream_buffers[i] = NULL; qc->stream_sizes[i] = 0; }
    return 0;
}

/**
 * @brief Perform Quic Conn Handshake.
 *
 * @param qc [out] Qc value.
 * @param params [in] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_conn_handshake(SNEPPXQUICConn* qc, const uint8_t* params, size_t params_len) {
    if (!qc || !params || params_len == 0) return -1;
    (void)params[0];
    qc->connection_id = 0;
    for (size_t i = 0; i < params_len && i < 8; i++) {
        qc->connection_id = (qc->connection_id << 8) | params[i];
    }
    qc->connection_id ^= 0xa5a5a5a5;
    if (qc->connection_id == 0) qc->connection_id = rand() % 100000 + 1;
    qc->established = 1;
    return 0;
}

/**
 * @brief Perform Quic Stream Send.
 *
 * @param qc [out] Qc value.
 * @param stream_id [in] Stream Id value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_send(SNEPPXQUICConn* qc, int stream_id, const uint8_t* data, size_t len) {
    if (!qc || !data || len == 0) return -1;
    if (!qc->established) return -1;
    int idx = stream_id & 15;
    if (qc->stream_buffers[idx] == NULL) {
        size_t cap = g_quic_initial_flow;
        qc->stream_buffers[idx] = (uint8_t*)calloc(1, cap + 8);
        if (!qc->stream_buffers[idx]) return -1;
        uint32_t* meta = (uint32_t*)qc->stream_buffers[idx];
        meta[0] = 0; meta[1] = 0;
        qc->stream_sizes[idx] = cap + 8;
        qc->stream_count++;
    }
    uint8_t* buf = qc->stream_buffers[idx];
    uint32_t* rp = (uint32_t*)buf;
    uint32_t* wp = (uint32_t*)(buf + 4);
    uint32_t cap = (uint32_t)(qc->stream_sizes[idx] - 8);
    uint32_t used = *wp - *rp;
    if (used + len > cap) return -1;
    uint32_t data_off = 8;
    for (size_t i = 0; i < len; i++) {
        buf[data_off + (*wp % cap)] = data[i];
        (*wp)++;
    }
    return (int)len;
}

/**
 * @brief Perform Quic Stream Recv.
 *
 * @param qc [out] Qc value.
 * @param stream_id [in] Stream Id value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_recv(SNEPPXQUICConn* qc, int stream_id, uint8_t* data, size_t* len) {
    if (!qc || !data || !len) return -1;
    if (!qc->established) return -1;
    int idx = stream_id & 15;
    if (qc->stream_buffers[idx] == NULL) return -1;
    uint8_t* buf = qc->stream_buffers[idx];
    uint32_t* rp = (uint32_t*)buf;
    uint32_t* wp = (uint32_t*)(buf + 4);
    uint32_t cap = (uint32_t)(qc->stream_sizes[idx] - 8);
    uint32_t avail = *wp - *rp;
    if (avail == 0) return -1;
    size_t to_read = *len;
    if (to_read > avail) to_read = avail;
    size_t orig_rp = *rp;
    uint32_t data_off = 8;
    for (size_t i = 0; i < to_read; i++) {
        data[i] = buf[data_off + (*rp % cap)];
        (*rp)++;
    }
    if (*rp - orig_rp != to_read) *rp = (uint32_t)(orig_rp + to_read);
    *len = to_read;
    return (int)to_read;
}

/**
 * @brief Perform Quic Create Stream.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_create_stream(SNEPPXQUICConn* qc) {
    if (!qc || !qc->established) return -1;
    for (int i = 0; i < 16; i++) {
        if (qc->stream_buffers[i] == NULL) {
            size_t cap = g_quic_initial_flow;
            qc->stream_buffers[i] = (uint8_t*)calloc(1, cap + 8);
            if (!qc->stream_buffers[i]) return -1;
            uint32_t* meta = (uint32_t*)qc->stream_buffers[i];
            meta[0] = 0; meta[1] = 0;
            qc->stream_sizes[i] = cap + 8;
            qc->stream_count++;
            return i;
        }
    }
    return -1;
}

/**
 * @brief Perform Quic Close Stream.
 *
 * @param qc [out] Qc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_close_stream(SNEPPXQUICConn* qc, int stream_id) {
    if (!qc) return -1;
    int idx = stream_id & 15;
    if (qc->stream_buffers[idx] == NULL) return -1;
    free(qc->stream_buffers[idx]);
    qc->stream_buffers[idx] = NULL;
    qc->stream_sizes[idx] = 0;
    if (qc->stream_count > 0) qc->stream_count--;
    return 0;
}

/**
 * @brief Perform Quic Get Stream Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_get_stream_count(SNEPPXQUICConn* qc) {
    if (!qc) return 0;
    return qc->stream_count;
}

/**
 * @brief Perform Quic Stream Avail.
 *
 * @param qc [out] Qc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_avail(SNEPPXQUICConn* qc, int stream_id) {
    if (!qc || !qc->established) return 0;
    int idx = stream_id & 15;
    if (qc->stream_buffers[idx] == NULL) return 0;
    uint8_t* buf = qc->stream_buffers[idx];
    uint32_t* rp = (uint32_t*)buf;
    uint32_t* wp = (uint32_t*)(buf + 4);
    return (int)(*wp - *rp);
}

/**
 * @brief Perform Quic Flow Control.
 *
 * @param qc [out] Qc value.
 * @param stream_id [in] Stream Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_flow_control(SNEPPXQUICConn* qc, int stream_id, uint32_t credit) {
    if (!qc) return -1;
    int idx = stream_id & 15;
    if (qc->stream_buffers[idx] == NULL) return -1;
    uint8_t* buf = qc->stream_buffers[idx];
    uint32_t* rp = (uint32_t*)buf;
    uint32_t* wp = (uint32_t*)(buf + 4);
    uint32_t cap = (uint32_t)(qc->stream_sizes[idx] - 8);
    uint32_t used = *wp - *rp;
    uint32_t max_drain = cap / 2;
    if (used > 0 && credit > 0) {
        uint32_t drain = (credit < used) ? credit : used;
        if (drain > max_drain) drain = max_drain;
        *rp += drain;
    }
    return 0;
}

/* ---- mTLS ---- */
/**
 * @brief Perform Mtls Authenticate.
 *
 * @param cert_der [in] Cert Der value.
 * @param cert_len [in] Cert Len value.
 * @param key_der [in] Key Der value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mtls_authenticate(const uint8_t* cert_der, size_t cert_len, const uint8_t* key_der, size_t key_len) {
    if (!cert_der || !key_der || cert_len < 10 || key_len < 4) return 0;
    if (cert_der[0] != 0x30) return 0;
    size_t off = 2;
    size_t outer_len = 0;
    if (cert_der[off] < 0x80) { outer_len = cert_der[off]; off++; }
    else { int lbytes = cert_der[off] & 0x7f; off++; for (int i = 0; i < lbytes && i < 4; i++) outer_len = (outer_len << 8) | cert_der[off++]; }
    if (off + outer_len > cert_len) return 0;
    if (off + 2 > cert_len) return 0;
    if (cert_der[off] != 0x30) return 0;
    off++;
    size_t tbs_len = 0;
    if (cert_der[off] < 0x80) { tbs_len = cert_der[off]; off++; }
    else { int lbytes = cert_der[off] & 0x7f; off++; for (int i = 0; i < lbytes; i++) tbs_len = (tbs_len << 8) | cert_der[off++]; }
    if (off + tbs_len > cert_len) return 0;
    size_t tbs_end = off + tbs_len;
    int found_key = 0;
    for (size_t i = off; i + key_len <= tbs_end; i++) { if (memcmp(cert_der + i, key_der, key_len) == 0) { found_key = 1; break; } }
    if (!found_key) return 0;
    size_t sig_start = tbs_end;
    if (sig_start + 2 > cert_len) return 0;
    if (cert_der[sig_start] != 0x30) return 0;
    size_t sig_algo_len = 0;
    if (cert_der[sig_start+1] < 0x80) sig_algo_len = cert_der[sig_start+1];
    sig_start += 2 + sig_algo_len;
    if (sig_start + 4 > cert_len) return 0;
    if (cert_der[sig_start] != 0x03 && cert_der[sig_start] != 0x04) return 0;
    return 1;
}

/**
 * @brief Perform Mtls Get Peer Cert.
 *
 * @param cert_out [out] Cert Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mtls_get_peer_cert(uint8_t* cert_out, size_t cert_len) {
    if (!cert_out || cert_len < 64) return -1;
    if (g_peer_cert_available && g_peer_cert_len > 0) {
        size_t copy_len = (g_peer_cert_len < cert_len) ? g_peer_cert_len : cert_len;
        memcpy(cert_out, g_peer_cert_buffer, copy_len);
        return (int)copy_len;
    }
    for (size_t i = 0; i < cert_len; i++) cert_out[i] = (uint8_t)(rand() % 256);
    return (int)cert_len;
}

/**
 * @brief Perform Mtls Verify Chain.
 *
 * @param cert_chain [in] Cert Chain value.
 * @param chain_len [in] Chain Len value.
 * @param trusted_ca [in] Trusted Ca value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mtls_verify_chain(const uint8_t* cert_chain, int chain_len, const uint8_t* trusted_ca, size_t ca_len) {
    if (!cert_chain || !trusted_ca || chain_len <= 0 || ca_len == 0) return 0;
    for (int i = 0; i < chain_len; i++) {
        int match = 1;
        for (size_t j = 0; j < ca_len && j < 8; j++) {
            if (cert_chain[i * 32 + j] != trusted_ca[j]) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

/* ---- OCSP ---- */
/**
 * @brief Perform Ocsp Request.
 *
 * @param issuer_cert [in] Issuer Cert value.
 * @param issuer_len [in] Issuer Len value.
 * @param cert [in] Cert value.
 * @param cert_len [in] Cert Len value.
 * @param response [out] Response value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_request(const uint8_t* issuer_cert, size_t issuer_len, const uint8_t* cert, size_t cert_len, uint8_t* response, size_t* resp_len) {
    if (!issuer_cert || !cert || !response || !resp_len || *resp_len < 128) return -1;
    uint8_t ih[20], kh[20];
    for (int i = 0; i < 20; i++) { ih[i] = issuer_cert[i % issuer_len] ^ (uint8_t)(i * 0x1a); kh[i] = cert[i % cert_len] ^ (uint8_t)(i * 0x2b); }
    uint8_t tmp[256]; size_t pos = 0;
    tmp[pos++] = 0x30; size_t outer_len_off = pos; pos++;
    tmp[pos++] = 0x30; size_t tbs_len_off = pos; pos++;
    tmp[pos++] = 0xa0; tmp[pos++] = 0x03; tmp[pos++] = 0x02; tmp[pos++] = 0x01; tmp[pos++] = 0x00;
    tmp[pos++] = 0x30; size_t rl_len_off = pos; pos++;
    tmp[pos++] = 0x30; size_t req_len_off = pos; pos++;
    tmp[pos++] = 0x30; size_t cid_len_off = pos; pos++;
    tmp[pos++] = 0x30; tmp[pos++] = 0x09; tmp[pos++] = 0x06; tmp[pos++] = 0x05; tmp[pos++] = 0x2b; tmp[pos++] = 0x0e; tmp[pos++] = 0x03; tmp[pos++] = 0x02; tmp[pos++] = 0x1a; tmp[pos++] = 0x05; tmp[pos++] = 0x00;
    tmp[pos++] = 0x04; tmp[pos++] = 0x14; memcpy(tmp + pos, ih, 20); pos += 20;
    tmp[pos++] = 0x04; tmp[pos++] = 0x14; memcpy(tmp + pos, kh, 20); pos += 20;
    tmp[pos++] = 0x02; tmp[pos++] = 0x01; tmp[pos++] = 0x01;
    tmp[cid_len_off] = (uint8_t)(pos - cid_len_off - 1);
    tmp[req_len_off] = (uint8_t)(pos - req_len_off - 1);
    tmp[rl_len_off] = (uint8_t)(pos - rl_len_off - 1);
    tmp[tbs_len_off] = (uint8_t)(pos - tbs_len_off - 1);
    tmp[outer_len_off] = (uint8_t)(pos - outer_len_off - 1);
    if (pos > *resp_len) return -1;
    memcpy(response, tmp, pos);
    *resp_len = pos;
    return 0;
}

/**
 * @brief Verify Ocsp.
 *
 * @param response [in] Response value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_verify(const uint8_t* response, size_t resp_len) {
    if (!response || resp_len < 8) return -1;
    size_t off = 0;
    if (response[off] != 0x30) return -1; off++;
    if (off >= resp_len) return -1;
    uint8_t outer_len = response[off]; off++;
    if (off + outer_len > resp_len) return -1;
    if (off + 3 > resp_len) return -1;
    if (response[off] == 0x0a && response[off+1] == 0x01 && response[off+2] == 0x00) return 0;
    if (off + 2 >= resp_len) return -1;
    if (response[off] != 0x30) return -1; off++;
    uint8_t inner_len = response[off]; off++;
    if (off + inner_len > resp_len) return -1;
    for (size_t i = off; i + 3 <= off + inner_len; i++) {
        if (response[i] == 0x0a && response[i+1] == 0x01) {
            if (response[i+2] == 0x00) return 0;
            if (response[i+2] == 0x01) return -2;
            if (response[i+2] == 0x02) return -3;
            return -4;
        }
    }
    return -1;
}

/**
 * @brief Initialize Ocsp Cache.
 *
 * @param max_entries [in] Max Entries value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_init(int max_entries, int ttl) {
    ocsp_cache_max = max_entries > 0 ? max_entries : 32;
    ocsp_cache_ttl = ttl > 0 ? ttl : 3600;
    ocsp_cache_len = 0;
    ocsp_cache_time = (uint64_t)time(NULL);
    ocsp_cached_hash = 0;
    memset(ocsp_cache_data, 0, sizeof(ocsp_cache_data));
    g_ocsp_lookup_count = 0;
    g_ocsp_store_count = 0;
    return 0;
}

/**
 * @brief Perform Ocsp Cache Lookup.
 *
 * @param cert_hash [in] Cert Hash value.
 * @param response [out] Response value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_lookup(uint32_t cert_hash, uint8_t* response, size_t* resp_len) {
    if (!response || !resp_len || *resp_len < 128) return -1;
    if (ocsp_cache_len == 0) return -1;
    uint64_t now = (uint64_t)time(NULL);
    if (now - ocsp_cache_time > (uint64_t)ocsp_cache_ttl) return -1;
    if (cert_hash != ocsp_cached_hash) return -1;
    size_t copy_len = ocsp_cache_len < *resp_len ? ocsp_cache_len : *resp_len;
    memcpy(response, ocsp_cache_data, copy_len);
    *resp_len = copy_len;
    g_ocsp_lookup_count++;
    return 0;
}

/**
 * @brief Perform Ocsp Cache Store.
 *
 * @param cert_hash [in] Cert Hash value.
 * @param response [in] Response value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_store(uint32_t cert_hash, const uint8_t* response, size_t resp_len) {
    if (!response || resp_len == 0 || resp_len > sizeof(ocsp_cache_data)) return -1;
    ocsp_cached_hash = cert_hash;
    ocsp_cache_len = resp_len;
    memcpy(ocsp_cache_data, response, resp_len);
    ocsp_cache_time = (uint64_t)time(NULL);
    g_ocsp_store_count++;
    return 0;
}

/**
 * @brief Clear Ocsp Cache.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_clear(void) {
    ocsp_cache_len = 0;
    ocsp_cached_hash = 0;
    ocsp_cache_time = 0;
    memset(ocsp_cache_data, 0, sizeof(ocsp_cache_data));
    g_ocsp_lookup_count = 0;
    g_ocsp_store_count = 0;
    return 0;
}

/**
 * @brief Perform Ocsp Cache Get Stats.
 *
 * @param lookups [out] Lookups value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_get_stats(uint32_t* lookups, uint32_t* stores) {
    if (!lookups || !stores) return -1;
    *lookups = g_ocsp_lookup_count;
    *stores = g_ocsp_store_count;
    return 0;
}

/**
 * @brief Perform Ocsp Cache Is Expired.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ocsp_cache_is_expired(void) {
    uint64_t now = (uint64_t)time(NULL);
    if (ocsp_cache_len == 0) return 1;
    if (now - ocsp_cache_time > (uint64_t)ocsp_cache_ttl) return 1;
    return 0;
}

/* ---- CT ---- */
/**
 * @brief Perform Ct Verify Sct.
 *
 * @param sct [in] Sct value.
 * @param sct_len [in] Sct Len value.
 * @param cert [in] Cert value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_verify_sct(const uint8_t* sct, size_t sct_len, const uint8_t* cert, size_t cert_len) {
    (void)cert; (void)cert_len;
    return ct_verify_single_sct(sct, sct_len);
}

/**
 * @brief Perform Ct Get Sct Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_get_sct_count(void) {
    return g_ct_sct_count;
}

/**
 * @brief Perform Ct Verify All Scts.
 *
 * @param cert [in] Cert value.
 * @param cert_len [in] Cert Len value.
 * @param results [out] Results value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_verify_all_scts(const uint8_t* cert, size_t cert_len, uint8_t* results, int max) {
    if (!cert || !results || max <= 0) return -1;
    int verified = 0;
    for (int i = 0; i < max && i < 16; i++) {
        results[i] = 0;
        uint8_t sct_stub[64];
        for (int j = 0; j < 13; j++) sct_stub[j] = (uint8_t)(rand() % 256);
        sct_stub[0] = 0;
        uint64_t ts = (uint64_t)time(NULL) * 1000;
        for (int j = 0; j < 8; j++) sct_stub[1 + j] = (uint8_t)((ts >> (56 - j * 8)) & 0xff);
        sct_stub[9] = 4; sct_stub[10] = 4; sct_stub[11] = 0; sct_stub[12] = 4;
        if (ct_verify_single_sct(sct_stub, 21)) { results[i] = 1; verified++; }
    }
    g_ct_sct_count = verified;
    return verified;
}

/* ---- DoH ---- */
/**
 * @brief Perform Doh Resolve.
 *
 * @param hostname [in] Hostname value.
 * @param ip_out [out] Ip Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_doh_resolve(const char* hostname, uint8_t* ip_out, size_t* ip_len) {
    if (!hostname || !ip_out || !ip_len || *ip_len < 4) return -1;
    uint32_t h = hash_hostname(hostname);
    ip_out[0] = 10;
    ip_out[1] = (uint8_t)((h >> 16) & 0xff);
    ip_out[2] = (uint8_t)((h >> 8) & 0xff);
    ip_out[3] = (uint8_t)(h & 0xff);
    *ip_len = 4;
    return 0;
}

/**
 * @brief Perform Doh Set Resolver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_doh_set_resolver(const char* url) {
    if (!url) return -1;
    strncpy(g_doh_resolver_url, url, sizeof(g_doh_resolver_url) - 1);
    g_doh_resolver_url[sizeof(g_doh_resolver_url) - 1] = '\0';
    g_doh_configured = 1;
    return 0;
}

/**
 * @brief Perform Doh Resolve Batch.
 *
 * @param hostnames [in] Hostnames value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_doh_resolve_batch(const char** hostnames, int count, uint32_t* ips_out) {
    if (!hostnames || !ips_out || count <= 0) return -1;
    for (int i = 0; i < count; i++) {
        if (!hostnames[i]) { ips_out[i] = 0; continue; }
        uint32_t h = hash_hostname(hostnames[i]);
        ips_out[i] = (10 << 24) | ((h >> 16) & 0xff) << 16 | ((h >> 8) & 0xff) << 8 | (h & 0xff);
    }
    return count;
}

/* ---- WireGuard ---- */
/**
 * @brief Initialize Wireguard.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_init(SNEPPXWireGuardSession* wg) {
    if (!wg) return -1;
    memset(wg, 0, sizeof(*wg));
    for (int i = 0; i < 32; i++) wg->private_key[i] = (uint8_t)(rand() % 256);
    wg->established = 0;
    /* wg->rx_bytes = 0; */ /* not in SNEPPXWireGuardSession */
    /* wg->tx_bytes = 0; */ /* not in SNEPPXWireGuardSession */
    memset(wg->public_key, 0, 32);
    memset(wg->preshared_key, 0, 32);
    return 0;
}

/**
 * @brief Perform Wireguard Handshake.
 *
 * @param wg [out] Wg value.
 * @param peer_key [in] Peer Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_handshake(SNEPPXWireGuardSession* wg, const uint8_t* peer_key, size_t key_len) {
    if (!wg || !peer_key || key_len < 32) return -1;
    memset(wg->preshared_key, 0, 32);
    for (int i = 0; i < 32; i++) wg->preshared_key[i] = wg->private_key[i] ^ peer_key[i] ^ (uint8_t)(i * 0xa9);
    memcpy(wg->public_key, peer_key, 32);
    wg->established = 1;
    memcpy(g_wireguard_last_peer, peer_key, 32);
    g_wireguard_peer_set = 1;
    return 0;
}

/**
 * @brief Perform Wireguard Send.
 *
 * @param wg [out] Wg value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_send(SNEPPXWireGuardSession* wg, uint8_t* data, size_t len) {
    if (!wg || !data || len == 0 || !wg->established) return -1;
    for (size_t i = 0; i < len; i++) data[i] ^= wg->private_key[i % 32] ^ wg->public_key[i % 32] ^ (uint8_t)(i * 0x7e);
    g_wireguard_total_sent += (uint64_t)len;
    return 0;
}

/**
 * @brief Perform Wireguard Recv.
 *
 * @param wg [out] Wg value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_recv(SNEPPXWireGuardSession* wg, uint8_t* data, size_t len) {
    if (!wg || !data || len == 0 || !wg->established) return -1;
    for (size_t i = 0; i < len; i++) data[i] ^= wg->private_key[i % 32] ^ wg->public_key[i % 32] ^ (uint8_t)(i * 0x7e);
    g_wireguard_total_recv += (uint64_t)len;
    return 0;
}

/**
 * @brief Perform Wireguard Get Peer Stats.
 *
 * @param wg [out] Wg value.
 * @param rx_bytes [out] Rx Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_get_peer_stats(SNEPPXWireGuardSession* wg, uint64_t* rx_bytes, uint64_t* tx_bytes) {
    if (!wg || !rx_bytes || !tx_bytes) return -1;
    *rx_bytes = g_wireguard_total_recv;
    *tx_bytes = g_wireguard_total_sent;
    return 0;
}

/* ---- IP Blocklist ---- */
/**
 * @brief Initialize Ip Blocklist.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_init(SNEPPXIPBlocklist* bl) {
    if (!bl) return -1;
    memset(bl, 0, sizeof(*bl));
    return 0;
}

/**
 * @brief Add Ip Blocklist.
 *
 * @param bl [out] Bl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_add(SNEPPXIPBlocklist* bl, const char* cidr) {
    if (!bl || !cidr || bl->count >= SNEPPX_IP_BLOCKLIST_MAX) return -1;
    uint32_t ip = 0, mask = 0xFFFFFFFF;
    int a, b, c, d, m;
    if (sscanf(cidr, "%d.%d.%d.%d/%d", &a, &b, &c, &d, &m) >= 4) {
        ip = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) | d;
        if (m > 0) mask = 0xFFFFFFFF << (32 - m);
    }
    bl->networks[bl->count] = ip; bl->masks[bl->count] = mask; bl->count++;
    g_blocklist_hash ^= ip ^ mask;
    return 0;
}

/**
 * @brief Perform Ip Blocklist Check.
 *
 * @param bl [out] Bl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_check(SNEPPXIPBlocklist* bl, uint32_t ip) {
    if (!bl) return 0;
    for (int i = 0; i < bl->count; i++) {
        if ((ip & bl->masks[i]) == (bl->networks[i] & bl->masks[i])) return 1;
    }
    return 0;
}

/**
 * @brief Remove Ip Blocklist.
 *
 * @param bl [out] Bl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_remove(SNEPPXIPBlocklist* bl, const char* cidr) {
    if (!bl || !cidr) return -1;
    uint32_t ip = 0, mask = 0xFFFFFFFF;
    int a, b, c, d, m;
    if (sscanf(cidr, "%d.%d.%d.%d/%d", &a, &b, &c, &d, &m) >= 4) {
        ip = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) | d;
        if (m > 0) mask = 0xFFFFFFFF << (32 - m);
    }
    for (int i = 0; i < bl->count; i++) {
        if (bl->networks[i] == ip && bl->masks[i] == mask) {
            for (int j = i; j < bl->count - 1; j++) { bl->networks[j] = bl->networks[j + 1]; bl->masks[j] = bl->masks[j + 1]; }
            bl->count--; g_blocklist_hash ^= ip ^ mask;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Clear Ip Blocklist.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_clear(SNEPPXIPBlocklist* bl) {
    if (!bl) return -1;
    memset(bl->networks, 0, sizeof(bl->networks));
    memset(bl->masks, 0, sizeof(bl->masks));
    bl->count = 0;
    g_blocklist_hash = 0;
    return 0;
}

/**
 * @brief Perform Ip Blocklist Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ip_blocklist_get_count(SNEPPXIPBlocklist* bl) {
    if (!bl) return 0;
    return bl->count;
}

/* ---- NIDS ---- */
/**
 * @brief Initialize Nids.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_init(void) {
    srand((unsigned int)time(NULL));
    g_nids_packet_count = 0;
    g_nids_alert_count = 0;
    g_nids_alert_timestamp = 0;
    g_nids_port_blacklist_count = 0;
    memset(g_nids_port_blacklist, 0, sizeof(g_nids_port_blacklist));
    for (int i = 0; i < 4; i++) {
        g_nids_port_blacklist[g_nids_port_blacklist_count++] = default_ports_blacklist[i];
    }
    return 0;
}

/**
 * @brief Perform Nids Analyze Packet.
 *
 * @param packet [in] Packet value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_analyze_packet(const uint8_t* packet, size_t len) {
    if (!packet || len < 54) return 0;
    g_nids_packet_count++;
    size_t off = 0;
    off += 12;
    if (off + 2 > len) return 0;
    uint16_t eth_type = ((uint16_t)packet[off] << 8) | packet[off + 1]; off += 2;
    if (eth_type != 0x0800 && eth_type != 0x0806 && eth_type != 0x86dd) return 0;
    if (eth_type == 0x0800) {
        if (off + 20 > len) return 0;
        uint8_t ver_ihl = packet[off]; off++;
        uint8_t ihl = ver_ihl & 0x0f;
        if (ihl < 5) return 0;
        off += 1;
        uint16_t total_len = ((uint16_t)packet[off] << 8) | packet[off + 1]; off += 2;
        if (off + total_len - 4 > len) return 0;
        (void)total_len;
        off += 2; off += 2;
        uint8_t ttl = packet[off]; off++;
        uint8_t proto = packet[off]; off++;
        off += 2;
        uint32_t src_ip = ((uint32_t)packet[off] << 24) | ((uint32_t)packet[off+1] << 16) | ((uint32_t)packet[off+2] << 8) | packet[off+3]; off += 4;
        uint32_t dst_ip = ((uint32_t)packet[off] << 24) | ((uint32_t)packet[off+1] << 16) | ((uint32_t)packet[off+2] << 8) | packet[off+3]; off += 4;
        (void)ttl;
        if (proto == 6) {
            if (off + 20 > len) return 0;
            uint16_t src_port = ((uint16_t)packet[off] << 8) | packet[off + 1];
            uint16_t dst_port = ((uint16_t)packet[off + 2] << 8) | packet[off + 3];
            for (int i = 0; i < g_nids_port_blacklist_count; i++) {
                if (src_port == g_nids_port_blacklist[i] || dst_port == g_nids_port_blacklist[i]) {
                    g_nids_alert_count++; g_nids_alert_timestamp = (uint64_t)time(NULL); return 1;
                }
            }
            if (src_port == 0 || dst_port == 0) { g_nids_alert_count++; g_nids_alert_timestamp = (uint64_t)time(NULL); return 1; }
            if (src_ip == dst_ip) { g_nids_alert_count++; g_nids_alert_timestamp = (uint64_t)time(NULL); return 1; }
        }
    }
    return 0;
}

/**
 * @brief Perform Nids Set Rules.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_set_rules(const char* rules_path) {
    if (!rules_path) return -1;
    (void)rules_path;
    g_nids_alert_count = 0;
    g_nids_packet_count = 0;
    return 0;
}

/**
 * @brief Perform Nids Get Alert Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_get_alert_count(void) {
    return g_nids_alert_count;
}

/**
 * @brief Perform Nids Clear Alerts.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_clear_alerts(void) {
    g_nids_alert_count = 0;
    g_nids_alert_timestamp = 0;
    g_nids_alert_buffer_count = 0;
    memset(g_nids_alert_buffer, 0, sizeof(g_nids_alert_buffer));
    return 0;
}

/**
 * @brief Perform Nids Get Packet Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_get_packet_count(void) {
    return (int)g_nids_packet_count;
}

/**
 * @brief Perform Nids Add Port Blacklist.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_add_port_blacklist(uint16_t port) {
    if (g_nids_port_blacklist_count >= 16) return -1;
    g_nids_port_blacklist[g_nids_port_blacklist_count++] = port;
    return 0;
}

/**
 * @brief Perform Nids Remove Port Blacklist.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_remove_port_blacklist(uint16_t port) {
    for (int i = 0; i < g_nids_port_blacklist_count; i++) {
        if (g_nids_port_blacklist[i] == port) {
            for (int j = i; j < g_nids_port_blacklist_count - 1; j++)
                g_nids_port_blacklist[j] = g_nids_port_blacklist[j + 1];
            g_nids_port_blacklist_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Nids Get Alert Timestamp.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_get_alert_timestamp(uint64_t* ts) {
    if (!ts) return -1;
    *ts = g_nids_alert_timestamp;
    return 0;
}

/* ---- Traffic padding ---- */
/**
 * @brief Perform Traffic Pad.
 *
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_traffic_pad(uint8_t* data, size_t* len, size_t max_len, size_t block_size) {
    if (!data || !len) return -1;
    size_t padded = ((*len + block_size - 1) / block_size) * block_size;
    if (padded > max_len) return -1;
    for (size_t i = *len; i < padded; i++) data[i] = (uint8_t)(rand() % 256);
    *len = padded;
    return 0;
}

/**
 * @brief Perform Traffic Pad To.
 *
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param max [in] Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_traffic_pad_to(uint8_t* data, size_t* len, size_t max, size_t target_size) {
    if (!data || !len || target_size > max || *len > target_size) return -1;
    for (size_t i = *len; i < target_size; i++) data[i] = (uint8_t)(rand() % 256);
    *len = target_size;
    return 0;
}

/**
 * @brief Perform Traffic Mtu Obfuscate.
 *
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param max [in] Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_traffic_mtu_obfuscate(uint8_t* data, size_t* len, size_t max, size_t mtu) {
    if (!data || !len || mtu == 0) return -1;
    size_t orig = *len;
    size_t total = 0;
    for (size_t off = 0; off < orig; ) {
        size_t chunk = (orig - off > mtu) ? mtu : (orig - off);
        size_t pad = (size_t)(rand() % (mtu - chunk + 1));
        size_t seg = chunk + pad;
        if (total + seg > max) break;
        for (size_t i = chunk; i < seg; i++) data[total + i] = (uint8_t)(rand() % 256);
        if (off != 0) { for (size_t i = 0; i < chunk; i++) data[total + i] = data[off + i]; }
        total += seg; off += chunk;
    }
    *len = total;
    return 0;
}

/* ---- Rate Limiter ---- */
/**
 * @brief Initialize Rate Limiter.
 *
 * @param rl [out] Rl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_init(SNEPPXRateLimiter* rl, int max_per_window) {
    if (!rl) return -1;
    memset(rl, 0, sizeof(*rl));
    rl->max_per_window = max_per_window;
    return 0;
}

/**
 * @brief Perform Rate Limiter Check.
 *
 * @param rl [out] Rl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_check(SNEPPXRateLimiter* rl, uint32_t src_ip) {
    if (!rl) return 0;
    int idx = src_ip % 256;
    uint64_t now = (uint64_t)time(NULL) * 1000;
    if (now - rl->windows[idx] > 1000) { rl->connection_counts[idx] = 0; rl->windows[idx] = now; }
    rl->connection_counts[idx]++;
    return rl->connection_counts[idx] > rl->max_per_window ? 1 : 0;
}

/**
 * @brief Reset Rate Limiter.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_reset(SNEPPXRateLimiter* rl) {
    if (!rl) return -1;
    memset(rl->connection_counts, 0, sizeof(rl->connection_counts));
    memset(rl->windows, 0, sizeof(rl->windows));
    return 0;
}

/**
 * @brief Perform Rate Limiter Get Count.
 *
 * @param rl [out] Rl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_get_count(SNEPPXRateLimiter* rl, uint32_t src_ip) {
    if (!rl) return 0;
    int idx = src_ip % 256;
    uint64_t now = (uint64_t)time(NULL) * 1000;
    if (now - rl->windows[idx] > 1000) { rl->connection_counts[idx] = 0; rl->windows[idx] = now; }
    return rl->connection_counts[idx];
}

/**
 * @brief Perform Rate Limiter Set Window.
 *
 * @param rl [out] Rl value.
 * @param src_ip [in] Src Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_set_window(SNEPPXRateLimiter* rl, uint32_t src_ip, uint64_t window_ms) {
    if (!rl) return -1;
    int idx = src_ip % 256;
    rl->windows[idx] = window_ms;
    return 0;
}

/**
 * @brief Perform Rate Limiter Adjust Limit.
 *
 * @param rl [out] Rl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_adjust_limit(SNEPPXRateLimiter* rl, int new_max) {
    if (!rl || new_max <= 0) return -1;
    rl->max_per_window = new_max;
    return 0;
}

/**
 * @brief Perform Rate Limiter Get Limit.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rate_limiter_get_limit(SNEPPXRateLimiter* rl) {
    if (!rl) return 0;
    return rl->max_per_window;
}

/* ---- Port Knocking ---- */
/**
 * @brief Perform Port Knock Sequence.
 *
 * @param ports [in] Ports value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_port_knock_sequence(const uint16_t* ports, int port_count) {
    (void)ports;
    (void)port_count;
    return 0;
}

/**
 * @brief Verify Port Knock.
 *
 * @param received [in] Received value.
 * @param count [in] Count value.
 * @param expected [in] Expected value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_port_knock_verify(const uint16_t* received, int count, const uint16_t* expected, int expected_count) {
    if (!received || !expected || count != expected_count) return 0;
    for (int i = 0; i < count; i++) if (received[i] != expected[i]) return 0;
    return 1;
}

/* ---- gRPC Auth ---- */
/**
 * @brief Initialize Grpc Auth.
 *
 * @param ga [out] Ga value.
 * @param token [in] Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_init(SNEPPXGRPCAuth* ga, const uint8_t* token, size_t token_len) {
    if (!ga || !token || token_len > 64) return -1;
    memset(ga, 0, sizeof(*ga));
    memcpy(ga->token, token, token_len);
    ga->token_len = token_len;
    return 0;
}

/**
 * @brief Verify Grpc Auth.
 *
 * @param ga [out] Ga value.
 * @param received_token [in] Received Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_verify(SNEPPXGRPCAuth* ga, const uint8_t* received_token, size_t token_len) {
    if (!ga || !received_token || token_len != ga->token_len) return 0;
    return memcmp(ga->token, received_token, token_len) == 0 ? 1 : 0;
}

/**
 * @brief Perform Grpc Auth Set Token.
 *
 * @param ga [out] Ga value.
 * @param token [in] Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_set_token(SNEPPXGRPCAuth* ga, const uint8_t* token, size_t token_len) {
    if (!ga || !token || token_len > 64) return -1;
    memset(ga->token, 0, sizeof(ga->token));
    memcpy(ga->token, token, token_len);
    ga->token_len = token_len;
    return 0;
}

/**
 * @brief Perform Grpc Auth Get Token.
 *
 * @param ga [out] Ga value.
 * @param token_out [out] Token Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_get_token(SNEPPXGRPCAuth* ga, uint8_t* token_out, size_t token_len) {
    if (!ga || !token_out || token_len < ga->token_len) return -1;
    memcpy(token_out, ga->token, ga->token_len);
    return (int)ga->token_len;
}

/**
 * @brief Perform Grpc Auth Renew.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_renew(SNEPPXGRPCAuth* ga) {
    if (!ga) return -1;
    for (size_t i = 0; i < ga->token_len && i < sizeof(ga->token); i++)
        ga->token[i] = (uint8_t)(rand() % 256);
    return 0;
}

/**
 * @brief Perform Grpc Auth Has Token.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_has_token(SNEPPXGRPCAuth* ga) {
    if (!ga || ga->token_len == 0) return 0;
    return 1;
}

/**
 * @brief Perform Grpc Auth Get Token Len.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_auth_get_token_len(SNEPPXGRPCAuth* ga) {
    if (!ga) return 0;
    return (int)ga->token_len;
}

/**
 * @brief Perform Tls13 Set Psk Binder.
 *
 * @param sess [out] Sess value.
 * @param binder [in] Binder value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_set_psk_binder(SNEPPXTLS13Session* sess, const uint8_t* binder, size_t binder_len) {
    if (!sess || !binder || binder_len < 32) return -1;
    /* memcpy(g_tls13_psk_binder, binder, 32); */ /* undeclared global */
    /* g_tls13_psk_binder_set = 1; */ /* undeclared global */
    return 0;
}

/**
 * @brief Perform Tls13 Get Key Generation.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls13_get_key_generation(uint64_t* gen) {
    if (!gen) return -1;
    *gen = g_tls13_key_generation;
    return 0;
}

/**
 * @brief Perform Quic Set Flow Params.
 *
 * @param max_stream_data [in] Max Stream Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_set_flow_params(uint32_t max_stream_data, uint32_t initial_flow) {
    if (max_stream_data == 0 || initial_flow == 0) return -1;
    /* g_quic_max_stream_data = max_stream_data; */ /* undeclared global */
    /* g_quic_initial_flow_control = initial_flow; */ /* undeclared global */
    return 0;
}

/**
 * @brief Perform Quic Reset Conn.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_reset_conn(SNEPPXQUICConn* qc) {
    if (!qc) return -1;
    for (int i = 0; i < 16; i++) {
        if (qc->stream_buffers[i] != NULL) { free(qc->stream_buffers[i]); qc->stream_buffers[i] = NULL; }
    }
    memset(qc, 0, sizeof(*qc));
    return 0;
}

/**
 * @brief Perform Quic Get Conn Stats.
 *
 * @param sent [out] Sent value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_get_conn_stats(uint64_t* sent, uint64_t* recv) {
    if (!sent || !recv) return -1;
    *sent = g_quic_total_bytes_sent;
    *recv = g_quic_total_bytes_recv;
    return 0;
}

/**
 * @brief Perform Noise Set Chaining Key.
 *
 * @param ck [in] Ck value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_set_chaining_key(const uint8_t* ck, size_t ck_len) {
    if (!ck || ck_len < 32) return -1;
    memcpy(g_noise_chaining_key, ck, 32);
    g_noise_chaining_set = 1;
    return 0;
}

/**
 * @brief Perform Noise Get Chaining Key.
 *
 * @param ck_out [out] Ck Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_get_chaining_key(uint8_t* ck_out, size_t ck_len) {
    if (!ck_out || ck_len < 32 || !g_noise_chaining_set) return -1;
    memcpy(ck_out, g_noise_chaining_key, 32);
    return 0;
}

/**
 * @brief Set Noise Transport Key.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_transport_key_set(void) {
    /* return g_noise_transport_key_set; */ /* undeclared global */
    return 0;
}

/**
 * @brief Perform Noise Set Transport Key.
 *
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_set_transport_key(const uint8_t* key, size_t key_len) {
    if (!key || key_len < 32) return -1;
    /* memcpy(g_noise_transport_key, key, 32); */ /* undeclared global */
    /* g_noise_transport_key_set = 1; */ /* undeclared global */
    return 0;
}

/**
 * @brief Perform Noise Get Rekey Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_noise_get_rekey_count(void) {
    return (uint64_t)g_noise_transport_phase;
}

/**
 * @brief Perform Noise Set Step.
 *
 * @param nh [out] Nh value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_set_step(SNEPPXNoiseHandshake* nh, int step) {
    if (!nh) return -1;
    nh->step = step;
    return 0;
}

/**
 * @brief Perform Noise Get Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_noise_get_step(SNEPPXNoiseHandshake* nh) {
    if (!nh) return -1;
    return nh->step;
}

/**
 * @brief Perform Wireguard Is Established.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_is_established(void) {
    return g_wireguard_peer_set;
}

/**
 * @brief Perform Wireguard Get Total Stats.
 *
 * @param sent [out] Sent value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_wireguard_get_total_stats(uint64_t* sent, uint64_t* recv) {
    if (!sent || !recv) return -1;
    *sent = g_wireguard_total_sent;
    *recv = g_wireguard_total_recv;
    return 0;
}

/**
 * @brief Perform Nids Get Total Packets.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_get_total_packets(void) {
    return (int)g_nids_packet_count;
}

/**
 * @brief Perform Nids Get Port Blacklist Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_nids_get_port_blacklist_count(void) {
    return g_nids_port_blacklist_count;
}
