#ifndef SNEPPX_TRANSPORT_SECURITY_H
#define SNEPPX_TRANSPORT_SECURITY_H
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

/*
 * S4 Network Security — Transport Security Layer
 * TLS 1.3 wrappers, Noise protocol handshake, QUIC session management.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define SNEPPX_TLS_MAX_SESSIONS 64
#define SNEPPX_TLS_KEY_LEN 32
#define SNEPPX_TLS_NONCE_LEN 12
#define SNEPPX_TRANSPORT_AAD_TAG_LEN 32

    typedef struct
    {
        int session_id;
        int is_active;
        uint8_t session_key[SNEPPX_TLS_KEY_LEN];
        uint64_t creation_time;
        uint64_t last_used;
    } SNEPPXTLSSession;

    typedef struct
    {
        int enabled;
        SNEPPXTLSSession sessions[SNEPPX_TLS_MAX_SESSIONS];
        int session_count;
        int use_noise_protocol;
        int use_quic;
    } SNEPPXTransportSecurity;

    /**
     * @brief Initialize Transport.
     *
     * @param ts [out] Ts value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_init(SNEPPXTransportSecurity* ts);
    /**
     * @brief Perform Transport Shutdown.
     *
     * @param ts [out] Ts value.
     */
    void SNEPPX_transport_shutdown(SNEPPXTransportSecurity* ts);
    /**
     * @brief Perform Transport New Session.
     *
     * @param ts [out] Ts value.
     * @param psk [in] Psk value.
     * @param psk_len [in] Psk Len value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_new_session(SNEPPXTransportSecurity* ts, const uint8_t* psk, size_t psk_len);
    /**
     * @brief Perform Transport Close Session.
     *
     * @param ts [out] Ts value.
     * @param session_id [in] Session Id value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_close_session(SNEPPXTransportSecurity* ts, int session_id);
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
    int SNEPPX_transport_encrypt(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* plaintext, size_t len,
                                 uint8_t* ciphertext, uint8_t nonce[SNEPPX_TLS_NONCE_LEN]);
    /**
     * @brief Decrypt Transport.
     *
     * @param ts [out] Ts value.
     * @param session_id [in] Session Id value.
     * @param ciphertext [in] Ciphertext value.
     * @param len [in] Len value.
     * @param plaintext [out] Plaintext value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_decrypt(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* ciphertext, size_t len,
                                 const uint8_t nonce[SNEPPX_TLS_NONCE_LEN], uint8_t* plaintext);
    /**
     * @brief Encrypt with associated data (no integrity tag).
     *
     * Pairs with SNEPPX_transport_decrypt_with_aad. AAD is bound to the nonce;
     * use the *_aead variants below when tamper detection is required.
     */
    int SNEPPX_transport_encrypt_with_aad(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* plaintext,
                                          size_t len, const uint8_t* aad, size_t aad_len, uint8_t* ciphertext,
                                          uint8_t nonce[SNEPPX_TLS_NONCE_LEN]);
    int SNEPPX_transport_decrypt_with_aad(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* ciphertext,
                                          size_t len, const uint8_t* aad, size_t aad_len,
                                          const uint8_t nonce[SNEPPX_TLS_NONCE_LEN], uint8_t* plaintext);
    /**
     * @brief Encrypt with associated data and an HMAC-SHA256 integrity tag.
     */
    int SNEPPX_transport_encrypt_aead(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* plaintext, size_t len,
                                      const uint8_t* aad, size_t aad_len, uint8_t* ciphertext,
                                      uint8_t nonce[SNEPPX_TLS_NONCE_LEN], uint8_t tag[SNEPPX_TRANSPORT_AAD_TAG_LEN]);
    /**
     * @brief Decrypt with associated data and verify the HMAC-SHA256 tag.
     */
    int SNEPPX_transport_decrypt_aead(SNEPPXTransportSecurity* ts, int session_id, const uint8_t* ciphertext,
                                      size_t len, const uint8_t* aad, size_t aad_len,
                                      const uint8_t nonce[SNEPPX_TLS_NONCE_LEN],
                                      const uint8_t tag[SNEPPX_TRANSPORT_AAD_TAG_LEN], uint8_t* plaintext);
    /**
     * @brief Perform Transport Noise Handshake.
     *
     * @param ts [out] Ts value.
     * @param prologue [in] Prologue value.
     * @param prologue_len [in] Prologue Len value.
     * @param handshake_msg [out] Handshake Msg value.
     * @param msg_len [out] Msg Len value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_noise_handshake(SNEPPXTransportSecurity* ts, const uint8_t* prologue, size_t prologue_len,
                                         uint8_t* handshake_msg, size_t* msg_len);

    /**
     * @brief Select the transport cipher.
     *
     * Only ChaCha20 (cipher 0 / use_aes 0) is supported. Requests to enable
     * AES-GCM are rejected because this API has no authentication tag path.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_transport_set_cipher(SNEPPXTransportSecurity* ts, int use_aes);
    int SNEPPX_transport_set_cipher_type(int cipher_type);
    int SNEPPX_transport_get_cipher_type(void);

#ifdef __cplusplus
}
#endif
#endif
