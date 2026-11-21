#ifndef SNEPPX_NOISE_PROTOCOL_EXTENDED_H
#define SNEPPX_NOISE_PROTOCOL_EXTENDED_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_NOISE_MAX_PATTERN_LEN 64
#define SNEPPX_NOISE_HANDSHAKE_HASH_SIZE 64
#define SNEPPX_NOISE_MAX_PSK_COUNT 8
#define SNEPPX_NOISE_FALLBACK_PATTERNS 16

/*
 * SNEPPX - Noise Protocol Extended
 *
 * WHAT
 *   Noise Protocol Extended.
 *
 * CONCEPT
 *   Provides the Noise Protocol Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_NOISE_PATTERN_N,
    SNEPPX_NOISE_PATTERN_K,
    SNEPPX_NOISE_PATTERN_X,
    SNEPPX_NOISE_PATTERN_NN,
    SNEPPX_NOISE_PATTERN_NK,
    SNEPPX_NOISE_PATTERN_NX,
    SNEPPX_NOISE_PATTERN_KN,
    SNEPPX_NOISE_PATTERN_KK,
    SNEPPX_NOISE_PATTERN_KX,
    SNEPPX_NOISE_PATTERN_XN,
    SNEPPX_NOISE_PATTERN_XK,
    SNEPPX_NOISE_PATTERN_XX,
    SNEPPX_NOISE_PATTERN_IK,
    SNEPPX_NOISE_PATTERN_IX,
    SNEPPX_NOISE_PATTERN_IN,
    SNEPPX_NOISE_PATTERN_I1K,
    SNEPPX_NOISE_PATTERN_I1N,
    SNEPPX_NOISE_PATTERN_I1X,
    SNEPPX_NOISE_PATTERN_KK1,
    SNEPPX_NOISE_PATTERN_KK2,
    SNEPPX_NOISE_PATTERN_KK3,
    SNEPPX_NOISE_PATTERN_NX1,
    SNEPPX_NOISE_PATTERN_NX2,
    SNEPPX_NOISE_PATTERN_XX1,
    SNEPPX_NOISE_PATTERN_XX2,
    SNEPPX_NOISE_PATTERN_XX3,
    SNEPPX_NOISE_PATTERN_IK1,
    SNEPPX_NOISE_PATTERN_IK2,
    SNEPPX_NOISE_PATTERN_IK3,
    SNEPPX_NOISE_PATTERN_FALLBACK
} SNEPPXNoiseHandshakePattern;

typedef enum {
    SNEPPX_NOISE_CIPHER_AES256GCM,
    SNEPPX_NOISE_CIPHER_CHACHAPOLY,
    SNEPPX_NOISE_CIPHER_AEGIS256,
    SNEPPX_NOISE_CIPHER_DEONIS_256
} SNEPPXNoiseCipherSuite;

typedef struct {
    SNEPPXNoiseHandshakePattern pattern;
    SNEPPXNoiseCipherSuite cipher;
    uint8_t prologue[1024];
    size_t prologue_len;
    uint8_t* s;
    size_t s_len;
    uint8_t* e;
    size_t e_len;
    uint8_t* rs;
    size_t rs_len;
    uint8_t* re;
    size_t re_len;
    uint8_t* psk[SNEPPX_NOISE_MAX_PSK_COUNT];
    size_t psk_lens[SNEPPX_NOISE_MAX_PSK_COUNT];
    uint32_t num_psk;
    uint8_t h[SNEPPX_NOISE_HANDSHAKE_HASH_SIZE];
    uint8_t ck[SNEPPX_NOISE_HANDSHAKE_HASH_SIZE];
    uint8_t* payload_buffer;
    size_t payload_len;
    uint32_t message_pattern_index;
    uint32_t total_messages;
    uint8_t initiator : 1;
    uint8_t handshake_finished : 1;
    uint8_t transport_phase : 1;
} SNEPPXNoiseHandshakeState;

typedef struct {
    uint8_t* sending_key;
    size_t sending_key_len;
    uint8_t* receiving_key;
    size_t receiving_key_len;
    uint64_t sending_nonce;
    uint64_t receiving_nonce;
    uint8_t* chaining_key;
    size_t chaining_key_len;
    uint8_t* handshake_hash;
    size_t handshake_hash_len;
} SNEPPXNoiseTransportState;

int snepx_noise_handshake_init(SNEPPXNoiseHandshakeState* hs, SNEPPXNoiseHandshakePattern pattern, SNEPPXNoiseCipherSuite cipher, uint8_t initiator);
int snepx_noise_handshake_set_prologue(SNEPPXNoiseHandshakeState* hs, const uint8_t* prologue, size_t len);
int snepx_noise_handshake_set_keypair(SNEPPXNoiseHandshakeState* hs, const uint8_t* static_priv, size_t s_priv_len, const uint8_t* static_pub, size_t s_pub_len);
int snepx_noise_handshake_set_ephemeral(SNEPPXNoiseHandshakeState* hs, const uint8_t* ephemeral_priv, size_t e_priv_len, const uint8_t* ephemeral_pub, size_t e_pub_len);
int snepx_noise_handshake_add_psk(SNEPPXNoiseHandshakeState* hs, const uint8_t* psk, size_t psk_len);
int snepx_noise_handshake_write_message(SNEPPXNoiseHandshakeState* hs, const uint8_t* payload, size_t payload_len, uint8_t* message_out, size_t* message_len, SNEPPXNoiseTransportState* transport);
int snepx_noise_handshake_read_message(SNEPPXNoiseHandshakeState* hs, const uint8_t* message, size_t message_len, uint8_t* payload_out, size_t* payload_len, SNEPPXNoiseTransportState* transport);
int snepx_noise_handshake_get_hash(SNEPPXNoiseHandshakeState* hs, uint8_t* hash_out, size_t* hash_len);
int snepx_noise_handshake_destroy(SNEPPXNoiseHandshakeState* hs);

int snepx_noise_transport_encrypt(SNEPPXNoiseTransportState* ts, const uint8_t* plaintext, size_t plaintext_len, uint8_t* ciphertext, size_t* ciphertext_len);
int snepx_noise_transport_decrypt(SNEPPXNoiseTransportState* ts, const uint8_t* ciphertext, size_t ciphertext_len, uint8_t* plaintext, size_t* plaintext_len);
int snepx_noise_transport_rekey(SNEPPXNoiseTransportState* ts);
int snepx_noise_transport_destroy(SNEPPXNoiseTransportState* ts);

// Fallback patterns and PSK modes
int snepx_noise_pattern_supported(SNEPPXNoiseHandshakePattern pattern);
int snepx_noise_pattern_to_string(SNEPPXNoiseHandshakePattern pattern, char* out, size_t out_max);

#endif
