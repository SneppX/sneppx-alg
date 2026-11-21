#ifndef SNEPPX_TLS_EXTENDED_H
#define SNEPPX_TLS_EXTENDED_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TLS_MAX_CERTS 16
#define SNEPPX_TLS_MAX_EXTENSIONS 64
#define SNEPPX_TLS_MAX_CIPHER_SUITES 128

/*
 * SNEPPX - Tls Extended
 *
 * WHAT
 *   Tls Extended.
 *
 * CONCEPT
 *   Provides the Tls Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TLS_VERSION_1_2,
    SNEPPX_TLS_VERSION_1_3,
    SNEPPX_TLS_VERSION_DRAFT_1_4
} SNEPPXTLSVersion;

typedef enum {
    SNEPPX_TLS_CIPHER_AES_128_GCM_SHA256,
    SNEPPX_TLS_CIPHER_AES_256_GCM_SHA384,
    SNEPPX_TLS_CIPHER_CHACHA20_POLY1305_SHA256,
    SNEPPX_TLS_CIPHER_AES_128_CCM_8_SHA256,
    SNEPPX_TLS_CIPHER_AES_128_CCM_SHA256,
    SNEPPX_TLS_CIPHER_AEGIS_256_SHA512,
    SNEPPX_TLS_CIPHER_AEGIS_128L_SHA256
} SNEPPXTlsCipherSuite;

typedef enum {
    SNEPPX_TLS_GROUP_SECP256R1,
    SNEPPX_TLS_GROUP_SECP384R1,
    SNEPPX_TLS_GROUP_SECP521R1,
    SNEPPX_TLS_GROUP_X25519,
    SNEPPX_TLS_GROUP_X448,
    SNEPPX_TLS_GROUP_FFDHE2048,
    SNEPPX_TLS_GROUP_FFDHE3072,
    SNEPPX_TLS_GROUP_FFDHE4096,
    SNEPPX_TLS_GROUP_KYBER_768,
    SNEPPX_TLS_GROUP_KYBER_1024,
    SNEPPX_TLS_GROUP_BIKE_128,
    SNEPPX_TLS_GROUP_HQC_192
} SNEPPXTlsKeyExchangeGroup;

typedef enum {
    SNEPPX_TLS_SIG_RSA_PSS_2048_SHA256,
    SNEPPX_TLS_SIG_RSA_PSS_4096_SHA512,
    SNEPPX_TLS_SIG_ECDSA_SECP256R1_SHA256,
    SNEPPX_TLS_SIG_ECDSA_SECP384R1_SHA384,
    SNEPPX_TLS_SIG_ED25519,
    SNEPPX_TLS_SIG_ED448,
    SNEPPX_TLS_SIG_DILITHIUM_3,
    SNEPPX_TLS_SIG_DILITHIUM_5,
    SNEPPX_TLS_SIG_FALCON_512,
    SNEPPX_TLS_SIG_FALCON_1024,
    SNEPPX_TLS_SIG_SPHINCS_PLUS_SHA256
} SNEPPXTlsSignatureScheme;

typedef struct {
    uint8_t* session_id;
    size_t session_id_len;
    uint8_t* master_secret;
    size_t master_secret_len;
    uint8_t* server_random;
    size_t server_random_len;
    uint8_t* client_random;
    size_t client_random_len;
    SNEPPXTLSVersion version;
    SNEPPXTlsCipherSuite cipher_suite;
    uint8_t* cipher_suites;
    size_t cipher_suites_len;
    SNEPPXTlsKeyExchangeGroup named_group;
    SNEPPXTlsSignatureScheme sig_scheme;
    uint8_t* server_certificate;
    size_t server_certificate_len;
    uint8_t* client_certificate;
    size_t client_certificate_len;
    uint8_t* cert_chain[SNEPPX_TLS_MAX_CERTS];
    size_t cert_chain_lens[SNEPPX_TLS_MAX_CERTS];
    uint32_t num_certs;
    uint8_t* ocsp_response;
    size_t ocsp_response_len;
    uint8_t* sct_list;
    size_t sct_list_len;
    uint8_t* key_share;
    size_t key_share_len;
    uint8_t* psk_binder;
    size_t psk_binder_len;
    uint8_t* early_data;
    size_t early_data_len;
    uint8_t* extensions[SNEPPX_TLS_MAX_EXTENSIONS];
    size_t extensions_lens[SNEPPX_TLS_MAX_EXTENSIONS];
    uint32_t num_extensions;
} SNEPPXTlsSession;

typedef struct {
    uint8_t* certificate;
    size_t certificate_len;
    uint8_t* private_key;
    size_t private_key_len;
    uint8_t* certificate_chain;
    size_t chain_len;
    uint8_t* ocsp_stapling;
    size_t ocsp_len;
} SNEPPXTlsCredentials;

int snepx_tls_client_hello(SNEPPXTlsSession* session);
int snepx_tls_server_hello(SNEPPXTlsSession* session);
int snepx_tls_encrypted_extensions(SNEPPXTlsSession* session);
int snepx_tls_certificate(SNEPPXTlsSession* session, const SNEPPXTlsCredentials* creds);
int snepx_tls_certificate_verify(SNEPPXTlsSession* session, const SNEPPXTlsCredentials* creds);
int snepx_tls_finished(SNEPPXTlsSession* session);
int snepx_tls_handshake(SNEPPXTlsSession* session, const SNEPPXTlsCredentials* creds);

int snepx_tls13_derive_secret(SNEPPXTlsSession* session, const uint8_t* psk, size_t psk_len);
int snepx_tls13_derive_early_secret(SNEPPXTlsSession* session, const uint8_t* salt, size_t salt_len);
int snepx_tls13_derive_handshake_secret(SNEPPXTlsSession* session, const uint8_t* ecdhe, size_t ecdhe_len);
int snepx_tls13_derive_master_secret(SNEPPXTlsSession* session);
int snepx_tls13_derive_application_traffic(SNEPPXTlsSession* session, uint8_t* key, size_t* key_len, uint8_t* iv, size_t* iv_len);

int snepx_tls_encrypt(SNEPPXTlsSession* session, const uint8_t* plaintext, size_t plaintext_len, uint8_t* ciphertext, size_t* ciphertext_len, uint64_t seq_num);
int snepx_tls_decrypt(SNEPPXTlsSession* session, const uint8_t* ciphertext, size_t ciphertext_len, uint8_t* plaintext, size_t* plaintext_len, uint64_t seq_num);

int snepx_tls_key_update(SNEPPXTlsSession* session);
int snepx_tls_new_session_ticket(SNEPPXTlsSession* session, uint32_t ticket_lifetime);
int snepx_tls_session_resume(SNEPPXTlsSession* session, const uint8_t* ticket, size_t ticket_len);
int snepx_tls_session_ticket_decrypt(SNEPPXTlsSession* session, const uint8_t* ticket, size_t ticket_len);

// Certificate compression (RFC 8879)
int snepx_tls_cert_compress_brotli(SNEPPXTlsSession* session, uint8_t* compressed, size_t* compressed_len);
int snepx_tls_cert_decompress_brotli(SNEPPXTlsSession* session, const uint8_t* compressed, size_t compressed_len);

// Encrypted ClientHello (ECH)
int snepx_tls_ech_create(SNEPPXTlsSession* session, const uint8_t* client_hello, size_t ch_len, const uint8_t* config, size_t config_len, uint8_t* ech_out, size_t* ech_len);
int snepx_tls_ech_decrypt(SNEPPXTlsSession* session, const uint8_t* ech, size_t ech_len, uint8_t* client_hello_out, size_t* ch_out_len);

#endif
