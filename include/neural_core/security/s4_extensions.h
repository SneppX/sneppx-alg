#ifndef SNEPPX_S4_EXTENSIONS_H
#define SNEPPX_S4_EXTENSIONS_H
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


/* S4 Network Security extensions: TLS 1.3 full handshake, Noise NK/XX/IK, QUIC,
   mTLS, OCSP stapling, CT, DoH, WireGuard, IP blocklist, NIDS, traffic analysis,
   rate limiting, port knocking, gRPC auth */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TLS13_MAX_EXTENSIONS 16
#define SNEPPX_NOISE_MAX_PATTERNS 8
#define SNEPPX_IP_BLOCKLIST_MAX 256

/* TLS 1.3 full handshake */
typedef struct {
    uint8_t random[32];
    uint8_t session_id[32];
    int session_id_len;
    uint16_t cipher_suites[16];
    int cipher_count;
    uint16_t supported_groups[8];
    int group_count;
    uint8_t* extensions[SNEPPX_TLS13_MAX_EXTENSIONS];
    size_t ext_lens[SNEPPX_TLS13_MAX_EXTENSIONS];
    int ext_count;
} SNEPPXTLS13ClientHello;

typedef struct {
    int handshake_complete;
    uint8_t master_secret[48];
    uint8_t server_random[32];
} SNEPPXTLS13Session;

/**
 * @brief Initialize Tls13 Client Hello.
 *
 * @param ch [out] Ch value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls13_client_hello_init(SNEPPXTLS13ClientHello* ch);
/**
 * @brief Parse Tls13 Server Hello.
 *
 * @param sess [out] Sess value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls13_server_hello_parse(SNEPPXTLS13Session* sess, const uint8_t* data, size_t len);
/**
 * @brief Perform Tls13 Derive Keys.
 *
 * @param sess [out] Sess value.
 * @param psk [in] Psk value.
 * @param psk_len [in] Psk Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls13_derive_keys(SNEPPXTLS13Session* sess, const uint8_t* psk, size_t psk_len);

/* Noise protocol patterns NK, XX, IK */
typedef struct {
    int pattern; /* 0=NK, 1=XX, 2=IK */
    uint8_t s[32], e[32], rs[32], re[32];
    int initiator;
    int step;
} SNEPPXNoiseHandshake;

/**
 * @brief Initialize Noise.
 *
 * @param nh [out] Nh value.
 * @param pattern [in] Pattern value.
 * @param initiator [in] Initiator value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_noise_init(SNEPPXNoiseHandshake* nh, int pattern, int initiator);
/**
 * @brief Perform Noise Write Msg.
 *
 * @param nh [out] Nh value.
 * @param msg [out] Msg value.
 * @param msg_len [out] Msg Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_noise_write_msg(SNEPPXNoiseHandshake* nh, uint8_t* msg, size_t* msg_len);
/**
 * @brief Perform Noise Read Msg.
 *
 * @param nh [out] Nh value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_noise_read_msg(SNEPPXNoiseHandshake* nh, const uint8_t* msg, size_t msg_len);

/* QUIC connection manager */
typedef struct {
    int connection_id;
    uint8_t* stream_buffers[16];
    size_t stream_sizes[16];
    int stream_count;
    int established;
} SNEPPXQUICConn;

/**
 * @brief Initialize Quic Conn.
 *
 * @param qc [out] Qc value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_quic_conn_init(SNEPPXQUICConn* qc);
/**
 * @brief Perform Quic Conn Handshake.
 *
 * @param qc [out] Qc value.
 * @param params [in] Params value.
 * @param params_len [in] Params Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_quic_conn_handshake(SNEPPXQUICConn* qc, const uint8_t* params, size_t params_len);
/**
 * @brief Perform Quic Stream Send.
 *
 * @param qc [out] Qc value.
 * @param stream_id [in] Stream Id value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_quic_stream_send(SNEPPXQUICConn* qc, int stream_id, const uint8_t* data, size_t len);
/**
 * @brief Perform Quic Stream Recv.
 *
 * @param qc [out] Qc value.
 * @param stream_id [in] Stream Id value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_quic_stream_recv(SNEPPXQUICConn* qc, int stream_id, uint8_t* data, size_t* len);

/* mTLS */
/**
 * @brief Perform Mtls Authenticate.
 *
 * @param cert_der [in] Cert Der value.
 * @param cert_len [in] Cert Len value.
 * @param key_der [in] Key Der value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mtls_authenticate(const uint8_t* cert_der, size_t cert_len, const uint8_t* key_der, size_t key_len);

/* OCSP stapling */
/**
 * @brief Perform Ocsp Request.
 *
 * @param issuer_cert [in] Issuer Cert value.
 * @param issuer_len [in] Issuer Len value.
 * @param cert [in] Cert value.
 * @param cert_len [in] Cert Len value.
 * @param response [out] Response value.
 * @param resp_len [out] Resp Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ocsp_request(const uint8_t* issuer_cert, size_t issuer_len, const uint8_t* cert, size_t cert_len, uint8_t* response, size_t* resp_len);
/**
 * @brief Verify Ocsp.
 *
 * @param response [in] Response value.
 * @param resp_len [in] Resp Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ocsp_verify(const uint8_t* response, size_t resp_len);

/* Certificate Transparency */
/**
 * @brief Perform Ct Verify Sct.
 *
 * @param sct [in] Sct value.
 * @param sct_len [in] Sct Len value.
 * @param cert [in] Cert value.
 * @param cert_len [in] Cert Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ct_verify_sct(const uint8_t* sct, size_t sct_len, const uint8_t* cert, size_t cert_len);

/* DNS over HTTPS */
/**
 * @brief Perform Doh Resolve.
 *
 * @param hostname [in] Hostname value.
 * @param ip_out [out] Ip Out value.
 * @param ip_len [out] Ip Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_doh_resolve(const char* hostname, uint8_t* ip_out, size_t* ip_len);

/* WireGuard */
typedef struct {
    uint8_t private_key[32];
    uint8_t public_key[32];
    uint8_t preshared_key[32];
    int established;
} SNEPPXWireGuardSession;

/**
 * @brief Initialize Wireguard.
 *
 * @param wg [out] Wg value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_wireguard_init(SNEPPXWireGuardSession* wg);
/**
 * @brief Perform Wireguard Handshake.
 *
 * @param wg [out] Wg value.
 * @param peer_key [in] Peer Key value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_wireguard_handshake(SNEPPXWireGuardSession* wg, const uint8_t* peer_key, size_t key_len);

/* IP blocklist */
typedef struct {
    uint32_t networks[SNEPPX_IP_BLOCKLIST_MAX];
    uint32_t masks[SNEPPX_IP_BLOCKLIST_MAX];
    int count;
} SNEPPXIPBlocklist;

/**
 * @brief Initialize Ip Blocklist.
 *
 * @param bl [out] Bl value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ip_blocklist_init(SNEPPXIPBlocklist* bl);
/**
 * @brief Add Ip Blocklist.
 *
 * @param bl [out] Bl value.
 * @param cidr [in] Cidr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ip_blocklist_add(SNEPPXIPBlocklist* bl, const char* cidr);
/**
 * @brief Perform Ip Blocklist Check.
 *
 * @param bl [out] Bl value.
 * @param ip [in] Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ip_blocklist_check(SNEPPXIPBlocklist* bl, uint32_t ip);

/* NIDS */
/**
 * @brief Initialize Nids.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_nids_init(void);
/**
 * @brief Perform Nids Analyze Packet.
 *
 * @param packet [in] Packet value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_nids_analyze_packet(const uint8_t* packet, size_t len);

/* Traffic analysis mitigation */
/**
 * @brief Perform Traffic Pad.
 *
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param max_len [in] Max Len value.
 * @param block_size [in] Block Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_traffic_pad(uint8_t* data, size_t* len, size_t max_len, size_t block_size);

/* Connection rate limiting */
typedef struct {
    uint32_t connection_counts[256];
    uint64_t windows[256];
    int max_per_window;
} SNEPPXRateLimiter;

/**
 * @brief Initialize Rate Limiter.
 *
 * @param rl [out] Rl value.
 * @param max_per_window [in] Max Per Window value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_rate_limiter_init(SNEPPXRateLimiter* rl, int max_per_window);
/**
 * @brief Perform Rate Limiter Check.
 *
 * @param rl [out] Rl value.
 * @param src_ip [in] Src Ip value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_rate_limiter_check(SNEPPXRateLimiter* rl, uint32_t src_ip);

/* Port knocking */
/**
 * @brief Perform Port Knock Sequence.
 *
 * @param ports [in] Ports value.
 * @param port_count [in] Port Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_port_knock_sequence(const uint16_t* ports, int port_count);
/**
 * @brief Verify Port Knock.
 *
 * @param received [in] Received value.
 * @param count [in] Count value.
 * @param expected [in] Expected value.
 * @param expected_count [in] Expected Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_port_knock_verify(const uint16_t* received, int count, const uint16_t* expected, int expected_count);

/* gRPC auth interceptor */
typedef struct {
    uint8_t token[64];
    size_t token_len;
    int authenticated;
} SNEPPXGRPCAuth;

/**
 * @brief Initialize Grpc Auth.
 *
 * @param ga [out] Ga value.
 * @param token [in] Token value.
 * @param token_len [in] Token Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grpc_auth_init(SNEPPXGRPCAuth* ga, const uint8_t* token, size_t token_len);
/**
 * @brief Verify Grpc Auth.
 *
 * @param ga [out] Ga value.
 * @param received_token [in] Received Token value.
 * @param token_len [in] Token Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grpc_auth_verify(SNEPPXGRPCAuth* ga, const uint8_t* received_token, size_t token_len);

#ifdef __cplusplus
}
#endif
#endif
