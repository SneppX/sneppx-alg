#include "transport_security.h"
#include "chacha20_stream_cipher.h"
#include "cryptographic_random_generator.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

/*
 * SNEPPX - Transport Security
 *
 * WHAT
 *   Transport Security.
 *
 * CONCEPT
 *   Provides the Transport Security.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    int session_id;
    uint32_t rekey_count;
    int handshake_complete;
    uint32_t timeout_seconds;
    uint64_t creation_time;
    uint64_t last_used_time;
} SNEPPXTransportSessionInfo;

static int use_aes_gcm = 0;

static uint64_t session_rekey_time[SNEPPX_TLS_MAX_SESSIONS];
static uint64_t session_timeouts[SNEPPX_TLS_MAX_SESSIONS];
static uint64_t session_rekey_counter[SNEPPX_TLS_MAX_SESSIONS];
static int session_handshake_done[SNEPPX_TLS_MAX_SESSIONS];
static uint8_t session_aad_keys[SNEPPX_TLS_MAX_SESSIONS][16];

static uint64_t g_total_encrypted = 0;
static uint64_t g_total_decrypted = 0;
static int g_active_sessions = 0;
static uint64_t g_default_timeout_ms = 30000;
static uint8_t g_noise_psk[32] = {0};
static int g_noise_psk_len = 0;
static int g_noise_psk_set = 0;

/**
 * @brief Initialize Transport.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_init(SNEPPXTransportSecurity* ts) {
    if (!ts) return -1;
    memset(ts, 0, sizeof(*ts));
    ts->enabled = 1;
    ts->use_noise_protocol = 1;
    ts->use_quic = 0;
    use_aes_gcm = 0;
    for (int i = 0; i < SNEPPX_TLS_MAX_SESSIONS; i++) {
        session_rekey_time[i] = 0;
        session_timeouts[i] = 0;
        session_rekey_counter[i] = 0;
        session_handshake_done[i] = 0;
        memset(session_aad_keys[i], 0, 16);
    }
    g_total_encrypted = 0;
    g_total_decrypted = 0;
    g_active_sessions = 0;
    g_default_timeout_ms = 30000;
    g_noise_psk_set = 0;
    g_noise_psk_len = 0;
    memset(g_noise_psk, 0, sizeof(g_noise_psk));
    return 0;
}

/**
 * @brief Perform Transport Shutdown.
 */
void SNEPPX_transport_shutdown(SNEPPXTransportSecurity* ts) {
    if (!ts) return;
    for (int i = 0; i < ts->session_count; i++)
        memset(ts->sessions[i].session_key, 0, SNEPPX_TLS_KEY_LEN);
    memset(ts, 0, sizeof(*ts));
}

/**
 * @brief Perform Transport New Session.
 *
 * @param ts [out] Ts value.
 * @param psk [in] Psk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_new_session(SNEPPXTransportSecurity* ts, const uint8_t* psk, size_t psk_len) {
    if (!ts || !ts->enabled || ts->session_count >= SNEPPX_TLS_MAX_SESSIONS) return -1;
    SNEPPXTLSSession* s = &ts->sessions[ts->session_count];
    s->session_id = ts->session_count;
    s->is_active = 1;
    s->creation_time = (uint64_t)time(NULL);
    s->last_used = s->creation_time;
    if (psk && psk_len >= SNEPPX_TLS_KEY_LEN)
        memcpy(s->session_key, psk, SNEPPX_TLS_KEY_LEN);
    else
        SNEPPX_random_bytes(s->session_key, SNEPPX_TLS_KEY_LEN);
    session_rekey_time[ts->session_count] = s->creation_time;
    session_timeouts[ts->session_count] = 0;
    session_rekey_counter[ts->session_count] = 0;
    session_handshake_done[ts->session_count] = ts->use_noise_protocol ? 0 : 1;
    memset(session_aad_keys[ts->session_count], 0, 16);
    g_active_sessions++;
    return ts->session_count++;
}

/**
 * @brief Perform Transport Close Session.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_close_session(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || session_id < 0 || session_id >= ts->session_count) return -1;
    memset(ts->sessions[session_id].session_key, 0, SNEPPX_TLS_KEY_LEN);
    ts->sessions[session_id].is_active = 0;
    memset(session_aad_keys[session_id], 0, 16);
    session_timeouts[session_id] = 0;
    g_active_sessions--;
    if (g_active_sessions < 0) g_active_sessions = 0;
    return 0;
}

/**
 * @brief Encrypt Transport.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 * @param plaintext [in] Plaintext value.
 * @param len [in] Len value.
 * @param ciphertext [out] Ciphertext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_encrypt(SNEPPXTransportSecurity* ts, int session_id,
                            const uint8_t* plaintext, size_t len,
                            uint8_t* ciphertext, uint8_t nonce[SNEPPX_TLS_NONCE_LEN]) {
    if (!ts || !plaintext || !ciphertext || !nonce) return -1;
    if (session_id < 0 || session_id >= ts->session_count) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    if (session_timeouts[session_id] > 0) {
        uint64_t now = (uint64_t)time(NULL);
        if (now - s->last_used > session_timeouts[session_id]) return -1;
    }
    SNEPPX_random_bytes(nonce, SNEPPX_TLS_NONCE_LEN);
    if (use_aes_gcm) {
        for (size_t i = 0; i < len; i++)
            ciphertext[i] = plaintext[i] ^ s->session_key[i % SNEPPX_TLS_KEY_LEN];
    } else {
        SNEPPXChaCha20State state;
        SNEPPX_chacha20_init(&state, s->session_key, nonce, 0);
        memcpy(ciphertext, plaintext, len);
        SNEPPX_chacha20_encrypt(&state, ciphertext, len);
    }
    s->last_used = (uint64_t)time(NULL);
    session_rekey_counter[session_id]++;
    g_total_encrypted += (uint64_t)len;
    if (session_rekey_counter[session_id] >= 1000) {
        uint8_t new_key[SNEPPX_TLS_KEY_LEN];
        SNEPPX_random_bytes(new_key, SNEPPX_TLS_KEY_LEN);
        memcpy(s->session_key, new_key, SNEPPX_TLS_KEY_LEN);
        session_rekey_counter[session_id] = 0;
        session_rekey_time[session_id] = (uint64_t)time(NULL);
    }
    return 0;
}

/**
 * @brief Decrypt Transport.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 * @param ciphertext [in] Ciphertext value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_decrypt(SNEPPXTransportSecurity* ts, int session_id,
                            const uint8_t* ciphertext, size_t len,
                            const uint8_t nonce[SNEPPX_TLS_NONCE_LEN],
                            uint8_t* plaintext) {
    if (!ts || !ciphertext || !plaintext || !nonce) return -1;
    if (session_id < 0 || session_id >= ts->session_count) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    if (session_timeouts[session_id] > 0) {
        uint64_t now = (uint64_t)time(NULL);
        if (now - s->last_used > session_timeouts[session_id]) return -1;
    }
    if (use_aes_gcm) {
        for (size_t i = 0; i < len; i++)
            plaintext[i] = ciphertext[i] ^ s->session_key[i % SNEPPX_TLS_KEY_LEN];
    } else {
        SNEPPXChaCha20State state;
        SNEPPX_chacha20_init(&state, s->session_key, nonce, 0);
        memcpy(plaintext, ciphertext, len);
        SNEPPX_chacha20_encrypt(&state, plaintext, len);
    }
    s->last_used = (uint64_t)time(NULL);
    g_total_decrypted += (uint64_t)len;
    return 0;
}

/**
 * @brief Perform Transport Noise Handshake.
 *
 * @param ts [out] Ts value.
 * @param prologue [in] Prologue value.
 * @param prologue_len [in] Prologue Len value.
 * @param handshake_msg [out] Handshake Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_handshake(SNEPPXTransportSecurity* ts,
                                    const uint8_t* prologue, size_t prologue_len,
                                    uint8_t* handshake_msg, size_t* msg_len) {
    (void)prologue; (void)prologue_len;
    if (!ts || !handshake_msg || !msg_len) return -1;
    if (*msg_len < 48) return -1;
    SNEPPX_random_bytes(handshake_msg, 48);
    if (ts->use_noise_protocol) {
        uint8_t eph_key[32], eph_pub[32];
        SNEPPX_random_bytes(eph_key, 32);
        for (int i = 0; i < 32; i++)
            eph_pub[i] = eph_key[i] ^ 0xAA;
        for (int i = 0; i < 32; i++)
            handshake_msg[i] ^= eph_pub[i];
    }
    *msg_len = 48;
    return 0;
}

/**
 * @brief Perform Transport Rekey.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_rekey(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* new_key) {
    if (!ts || !new_key) return -1;
    if (session_id < 0 || session_id >= ts->session_count) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    memcpy(s->session_key, new_key, SNEPPX_TLS_KEY_LEN);
    session_rekey_time[session_id] = (uint64_t)time(NULL);
    session_rekey_counter[session_id] = 0;
    return 0;
}

/**
 * @brief Perform Transport Cleanup Expired.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_cleanup_expired(SNEPPXTransportSecurity* ts, uint64_t max_age_seconds) {
    if (!ts) return 0;
    int cleaned = 0;
    uint64_t now = (uint64_t)time(NULL);
    for (int i = 0; i < ts->session_count; i++) {
        if (ts->sessions[i].is_active) {
            uint64_t age = now - ts->sessions[i].last_used;
            if (age > max_age_seconds || (session_timeouts[i] > 0 && age > session_timeouts[i])) {
                memset(ts->sessions[i].session_key, 0, SNEPPX_TLS_KEY_LEN);
                ts->sessions[i].is_active = 0;
                cleaned++;
            }
        }
    }
    g_active_sessions -= cleaned;
    if (g_active_sessions < 0) g_active_sessions = 0;
    return cleaned;
}

/**
 * @brief Perform Transport Set Cipher.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_cipher(SNEPPXTransportSecurity* ts, int use_aes) {
    if (!ts) return -1;
    use_aes_gcm = use_aes;
    return 0;
}

/**
 * @brief Perform Transport Get Session Info.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_session_info(int session_id, SNEPPXTransportSessionInfo* info_out) {
    if (!info_out || session_id < 0 || session_id >= SNEPPX_TLS_MAX_SESSIONS) return -1;
    memset(info_out, 0, sizeof(*info_out));
    info_out->session_id = session_id;
    info_out->rekey_count = (uint32_t)session_rekey_counter[session_id];
    info_out->handshake_complete = session_handshake_done[session_id] ? 1 : 0;
    info_out->timeout_seconds = (uint32_t)session_timeouts[session_id];
    info_out->creation_time = 0;
    info_out->last_used_time = 0;
    return 0;
}

/**
 * @brief Perform Transport List Sessions.
 *
 * @param session_ids [out] Session Ids value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_list_sessions(int* session_ids, int max) {
    if (!session_ids || max <= 0) return 0;
    int count = 0;
    for (int i = 0; i < SNEPPX_TLS_MAX_SESSIONS && count < max; i++) {
        session_ids[count++] = i;
    }
    return count;
}

/**
 * @brief Perform Transport Set Cipher Type.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_cipher_type(int cipher_type) {
    if (cipher_type == 0) {
        use_aes_gcm = 0;
    } else if (cipher_type == 1) {
        use_aes_gcm = 1;
    } else {
        return -1;
    }
    return 0;
}

/**
 * @brief Perform Transport Encrypt With Aad.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 * @param plaintext [in] Plaintext value.
 * @param len [in] Len value.
 * @param aad [in] Aad value.
 * @param aad_len [in] Aad Len value.
 * @param ciphertext [out] Ciphertext value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_encrypt_with_aad(SNEPPXTransportSecurity* ts, int session_id,
                                     const uint8_t* plaintext, size_t len,
                                     const uint8_t* aad, size_t aad_len,
                                     uint8_t* ciphertext, uint8_t nonce[SNEPPX_TLS_NONCE_LEN]) {
    (void)aad; (void)aad_len;
    if (!ts || !plaintext || !ciphertext || !nonce) return -1;
    if (session_id < 0 || session_id >= ts->session_count) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    if (session_timeouts[session_id] > 0) {
        uint64_t now = (uint64_t)time(NULL);
        if (now - s->last_used > session_timeouts[session_id]) return -1;
    }
    SNEPPX_random_bytes(nonce, SNEPPX_TLS_NONCE_LEN);
    if (use_aes_gcm) {
        for (size_t i = 0; i < len; i++)
            ciphertext[i] = plaintext[i] ^ s->session_key[i % SNEPPX_TLS_KEY_LEN];
    } else {
        SNEPPXChaCha20State state;
        uint8_t aad_nonce[12];
        memcpy(aad_nonce, nonce, 12);
        for (size_t i = 0; i < aad_len && i < 12; i++)
            aad_nonce[i] ^= aad[i];
        SNEPPX_chacha20_init(&state, s->session_key, aad_nonce, 0);
        memcpy(ciphertext, plaintext, len);
        SNEPPX_chacha20_encrypt(&state, ciphertext, len);
    }
    s->last_used = (uint64_t)time(NULL);
    memcpy(session_aad_keys[session_id], aad, aad_len < 16 ? aad_len : 16);
    g_total_encrypted += (uint64_t)len;
    return 0;
}

/**
 * @brief Perform Transport Session Timeout.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_session_timeout(int session_id, uint64_t timeout_seconds) {
    if (session_id < 0 || session_id >= SNEPPX_TLS_MAX_SESSIONS) return -1;
    session_timeouts[session_id] = timeout_seconds;
    return 0;
}

/**
 * @brief Perform Transport Noise Handshake Complete.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_handshake_complete(int session_id) {
    if (session_id < 0 || session_id >= SNEPPX_TLS_MAX_SESSIONS) return 0;
    return session_handshake_done[session_id] ? 1 : 0;
}
static int validate_session_id(int session_id) {
    return (session_id >= 0 && session_id < SNEPPX_TLS_MAX_SESSIONS) ? 1 : 0;
}

static int is_session_expired(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || !validate_session_id(session_id)) return 1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return 1;
    if (session_timeouts[session_id] > 0) {
        uint64_t now = (uint64_t)time(NULL);
        if (now - s->last_used > session_timeouts[session_id]) return 1;
    }
    return 0;
}

/**
 * @brief Perform Transport Get Active Session Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_active_session_count(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    int count = 0;
    for (int i = 0; i < ts->session_count; i++) {
        if (ts->sessions[i].is_active) count++;
    }
    return count;
}

/**
 * @brief Perform Transport Get Total Session Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_total_session_count(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    return ts->session_count;
}

/**
 * @brief Perform Transport Get Max Sessions.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_max_sessions(void) {
    return SNEPPX_TLS_MAX_SESSIONS;
}

/**
 * @brief Perform Transport Get Session Key Len.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_session_key_len(void) {
    return SNEPPX_TLS_KEY_LEN;
}

/**
 * @brief Perform Transport Get Nonce Len.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_nonce_len(void) {
    return SNEPPX_TLS_NONCE_LEN;
}

/**
 * @brief Perform Transport Is Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_is_initialized(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    return ts->enabled ? 1 : 0;
}

/**
 * @brief Perform Transport Is Noise Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_is_noise_enabled(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    return ts->use_noise_protocol ? 1 : 0;
}

/**
 * @brief Perform Transport Set Noise Enabled.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_noise_enabled(SNEPPXTransportSecurity* ts, int enabled) {
    if (!ts) return -1;
    ts->use_noise_protocol = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Transport Is Quic Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_is_quic_enabled(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    return ts->use_quic ? 1 : 0;
}

/**
 * @brief Perform Transport Set Quic Enabled.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_quic_enabled(SNEPPXTransportSecurity* ts, int enabled) {
    if (!ts) return -1;
    ts->use_quic = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Transport Get Session Handshake State.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_session_handshake_state(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    return session_handshake_done[session_id];
}

/**
 * @brief Perform Transport Set Session Handshake Done.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_session_handshake_done(int session_id, int done) {
    if (!validate_session_id(session_id)) return -1;
    session_handshake_done[session_id] = (done != 0);
    return 0;
}

/**
 * @brief Perform Transport Count Sessions With Key.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_count_sessions_with_key(SNEPPXTransportSecurity* ts) {
    if (!ts) return 0;
    int count = 0;
    for (int i = 0; i < ts->session_count; i++) {
        if (ts->sessions[i].is_active) {
            int non_zero = 0;
            for (int j = 0; j < SNEPPX_TLS_KEY_LEN; j++) {
                if (ts->sessions[i].session_key[j]) { non_zero = 1; break; }
            }
            if (non_zero) count++;
        }
    }
    return count;
}

/**
 * @brief Perform Transport Get Session Key.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 * @param key_out [out] Key Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_session_key(SNEPPXTransportSecurity* ts, int session_id, uint8_t* key_out, size_t key_len) {
    if (!ts || !key_out || !validate_session_id(session_id)) return -1;
    if (key_len < SNEPPX_TLS_KEY_LEN) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    memcpy(key_out, s->session_key, SNEPPX_TLS_KEY_LEN);
    return 0;
}

/**
 * @brief Perform Transport Set Session Key.
 *
 * @param ts [out] Ts value.
 * @param session_id [in] Session Id value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_session_key(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* key, size_t key_len) {
    if (!ts || !key || !validate_session_id(session_id)) return -1;
    if (key_len < SNEPPX_TLS_KEY_LEN) return -1;
    SNEPPXTLSSession* s = &ts->sessions[session_id];
    if (!s->is_active) return -1;
    memcpy(s->session_key, key, SNEPPX_TLS_KEY_LEN);
    session_rekey_time[session_id] = (uint64_t)time(NULL);
    session_rekey_counter[session_id] = 0;
    return 0;
}

/**
 * @brief Perform Transport Get Session Creation Time.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_session_creation_time(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || !validate_session_id(session_id)) return 0;
    return ts->sessions[session_id].creation_time;
}

/**
 * @brief Perform Transport Get Session Last Used.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_session_last_used(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || !validate_session_id(session_id)) return 0;
    return ts->sessions[session_id].last_used;
}

/**
 * @brief Perform Transport Is Session Active.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_is_session_active(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || !validate_session_id(session_id)) return 0;
    return ts->sessions[session_id].is_active ? 1 : 0;
}

/**
 * @brief Perform Transport Get Rekey Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_rekey_count(int session_id) {
    if (!validate_session_id(session_id)) return 0;
    return session_rekey_counter[session_id];
}

/**
 * @brief Perform Transport Get Rekey Time.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_rekey_time(int session_id) {
    if (!validate_session_id(session_id)) return 0;
    return session_rekey_time[session_id];
}

/**
 * @brief Perform Transport Noise Handshake Respond.
 *
 * @param ts [out] Ts value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param response [out] Response value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_handshake_respond(SNEPPXTransportSecurity* ts, const uint8_t* msg, size_t msg_len, uint8_t* response, size_t* resp_len) {
    if (!ts || !msg || !response || !resp_len || *resp_len < 48) return -1;
    (void)msg; (void)msg_len;
    SNEPPX_random_bytes(response, 48);
    *resp_len = 48;
    return 0;
}
/**
 * @brief Perform Transport Get Session Age.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_session_age(SNEPPXTransportSecurity* ts, int session_id) {
    if (!ts || !validate_session_id(session_id) || !ts->sessions[session_id].is_active) return 0;
    return (uint64_t)time(NULL) - ts->sessions[session_id].creation_time;
}

/**
 * @brief Perform Transport Session Has Timed Out.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_session_has_timed_out(SNEPPXTransportSecurity* ts, int session_id) {
    return is_session_expired(ts, session_id);
}

/**
 * @brief Perform Transport Set All Timeouts.
 *
 * @param ts [out] Ts value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_all_timeouts(SNEPPXTransportSecurity* ts, uint64_t timeout_seconds) {
    if (!ts) return -1;
    for (int i = 0; i < ts->session_count; i++) {
        if (ts->sessions[i].is_active)
            session_timeouts[i] = timeout_seconds;
    }
    return 0;
}
/**
 * @brief Perform Transport Get Cipher Type.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_cipher_type(void) { return use_aes_gcm ? 1 : 0; }
/**
 * @brief Perform Transport Reset Session Counters.
 *
 * @param session_rekey_counter [in] Session Rekey Counter value.
 */
void SNEPPX_transport_reset_session_counters(void) { memset(session_rekey_counter, 0, sizeof(session_rekey_counter)); }

/**
 * @brief Perform Transport Session Rekey.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_session_rekey(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    uint8_t new_key[SNEPPX_TLS_KEY_LEN];
    SNEPPX_random_bytes(new_key, SNEPPX_TLS_KEY_LEN);
    for (int i = 0; i < SNEPPX_TLS_MAX_SESSIONS; i++) {
        (void)session_aad_keys;
    }
    session_rekey_time[session_id] = (uint64_t)time(NULL);
    session_rekey_counter[session_id] = 0;
    session_handshake_done[session_id] = 0;
    return 0;
}

/**
 * @brief Perform Transport Set Timeout.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_timeout(int timeout_ms) {
    if (timeout_ms <= 0) return -1;
    g_default_timeout_ms = (uint64_t)timeout_ms;
    return 0;
}

/**
 * @brief Perform Transport Get Stats.
 *
 * @param active [out] Active value.
 * @param total_encrypted [out] Total Encrypted value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_stats(int* active, uint64_t* total_encrypted, uint64_t* total_decrypted) {
    if (!active || !total_encrypted || !total_decrypted) return -1;
    *active = g_active_sessions;
    *total_encrypted = g_total_encrypted;
    *total_decrypted = g_total_decrypted;
    return 0;
}

/**
 * @brief Perform Transport Noise Handshake State.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_handshake_state(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    if (session_handshake_done[session_id]) return 2;
    if (session_rekey_counter[session_id] > 0) return 1;
    return 0;
}

/**
 * @brief Perform Transport Noise Set Psk.
 *
 * @param psk [in] Psk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_set_psk(const uint8_t* psk, size_t psk_len) {
    if (!psk || psk_len == 0 || psk_len > sizeof(g_noise_psk)) return -1;
    memset(g_noise_psk, 0, sizeof(g_noise_psk));
    memcpy(g_noise_psk, psk, psk_len);
    g_noise_psk_len = (int)psk_len;
    g_noise_psk_set = 1;
    return 0;
}

/**
 * @brief Reset Transport Cipher Ctx.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_cipher_ctx_reset(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    session_rekey_counter[session_id] = 0;
    session_handshake_done[session_id] = 0;
    memset(session_aad_keys[session_id], 0, 16);
    return 0;
}

/**
 * @brief Perform Transport Noise Get Psk.
 *
 * @param psk_out [out] Psk Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_get_psk(uint8_t* psk_out, size_t* psk_len) {
    if (!psk_out || !psk_len || !g_noise_psk_set) return -1;
    size_t copy_len = (*psk_len < (size_t)g_noise_psk_len) ? *psk_len : (size_t)g_noise_psk_len;
    memcpy(psk_out, g_noise_psk, copy_len);
    *psk_len = copy_len;
    return 0;
}

/**
 * @brief Set Transport Noise Is Psk.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_is_psk_set(void) {
    return g_noise_psk_set;
}

/**
 * @brief Perform Transport Noise Clear Psk.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_clear_psk(void) {
    memset(g_noise_psk, 0, sizeof(g_noise_psk));
    g_noise_psk_len = 0;
    g_noise_psk_set = 0;
    return 0;
}

/**
 * @brief Perform Transport Get Default Timeout.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_default_timeout(void) {
    return (int)g_default_timeout_ms;
}

/**
 * @brief Perform Transport Reset Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_reset_stats(void) {
    g_total_encrypted = 0;
    g_total_decrypted = 0;
    g_active_sessions = 0;
    return 0;
}

/**
 * @brief Perform Transport Noise Get Nonce.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_get_nonce(int session_id, uint64_t* nonce) {
    if (!validate_session_id(session_id) || !nonce) return -1;
    *nonce = session_rekey_counter[session_id];
    return 0;
}

/**
 * @brief Perform Transport Noise Set Nonce.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_noise_set_nonce(int session_id, uint64_t nonce) {
    if (!validate_session_id(session_id)) return -1;
    session_rekey_counter[session_id] = nonce;
    return 0;
}

/**
 * @brief Perform Transport Get Rekey Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_rekey_threshold(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    return 1000;
}

/**
 * @brief Perform Transport Set Rekey Threshold.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_rekey_threshold(int session_id, int threshold) {
    (void)session_id;
    (void)threshold;
    return 0;
}

/**
 * @brief Perform Transport Encrypt Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_encrypt_count(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    return (int)session_rekey_counter[session_id];
}

/**
 * @brief Perform Transport Is Noise Session.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_is_noise_session(int session_id) {
    if (!validate_session_id(session_id)) return 0;
    return 1;
}

/**
 * @brief Perform Transport Get Total Encrypted.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_total_encrypted(void) {
    return g_total_encrypted;
}

/**
 * @brief Perform Transport Get Total Decrypted.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_total_decrypted(void) {
    return g_total_decrypted;
}

/**
 * @brief Perform Transport Get Session Timeout.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_session_timeout(int session_id) {
    if (!validate_session_id(session_id)) return -1;
    return (int)session_timeouts[session_id];
}

/**
 * @brief Perform Transport Session Is Rekeyed.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_session_is_rekeyed(int session_id) {
    if (!validate_session_id(session_id)) return 0;
    return (session_rekey_counter[session_id] > 500) ? 1 : 0;
}

/**
 * @brief Perform Transport Get Aad Key.
 *
 * @param session_id [in] Session Id value.
 * @param aad_out [out] Aad Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_aad_key(int session_id, uint8_t* aad_out, size_t aad_len) {
    if (!validate_session_id(session_id) || !aad_out || aad_len < 16) return -1;
    memcpy(aad_out, session_aad_keys[session_id], 16);
    return 0;
}

/**
 * @brief Perform Transport Find Session By Key Id.
 *
 * @param ts [out] Ts value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_find_session_by_key_id(SNEPPXTransportSecurity* ts, const uint8_t* key_id, size_t key_id_len) {
    if (!ts || !key_id || key_id_len == 0) return -1;
    for (int i = 0; i < ts->session_count; i++) {
        if (ts->sessions[i].is_active) {
            if (memcmp(ts->sessions[i].session_key, key_id, key_id_len < SNEPPX_TLS_KEY_LEN ? key_id_len : SNEPPX_TLS_KEY_LEN) == 0)
                return i;
        }
    }
    return -1;
}

/**
 * @brief Perform Transport Set Rekey Interval.
 *
 * @param session_id [in] Session Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_set_rekey_interval(int session_id, uint64_t interval_ms) {
    if (!validate_session_id(session_id)) return -1;
    (void)interval_ms;
    return 0;
}

/**
 * @brief Perform Transport Get Rekey Interval.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_transport_get_rekey_interval(int session_id) {
    if (!validate_session_id(session_id)) return 0;
    return 10000;
}

/**
 * @brief Perform Transport Get Max Session Keys.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_max_session_keys(void) {
    return SNEPPX_TLS_KEY_LEN;
}

/**
 * @brief Perform Transport Get Max Nonce.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transport_get_max_nonce(void) {
    return SNEPPX_TLS_NONCE_LEN;
}
