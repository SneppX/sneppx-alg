#ifndef SNEPPX_TLS_TRANSPORT_H
#define SNEPPX_TLS_TRANSPORT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Tls Transport
 *
 * WHAT
 *   Tls Transport.
 *
 * CONCEPT
 *   Provides the Tls Transport.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { void* ctx; void* ssl; int fd; int is_server; } SNEPPXTlsSocket;
/**
 * @brief Initialize Tls.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_init(void);
/**
 * @brief Perform Tls Cleanup.
 */
void SNEPPX_tls_cleanup(void);
/**
 * @brief Create Tls Server Ctx.
 *
 * @param cert_path [in] Cert Path value.
 * @param key_path [in] Key Path value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_server_ctx_create(const char* cert_path, const char* key_path, void** ctx);
/**
 * @brief Create Tls Client Ctx.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_client_ctx_create(void** ctx);
/**
 * @brief Destroy Tls Ctx.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_tls_ctx_destroy(void* ctx);
/**
 * @brief Perform Tls Accept.
 *
 * @param ctx [out] Ctx value.
 * @param client_fd [in] Client Fd value.
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_accept(void* ctx, int client_fd, SNEPPXTlsSocket* sock);
/**
 * @brief Perform Tls Connect.
 *
 * @param ctx [out] Ctx value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_connect(void* ctx, const char* host, int port, SNEPPXTlsSocket* sock);
/**
 * @brief Close Tls.
 *
 * @param sock [out] Sock value.
 */
void SNEPPX_tls_close(SNEPPXTlsSocket* sock);
/**
 * @brief Perform Tls Send.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_send(SNEPPXTlsSocket* sock, const unsigned char* data, size_t len);
/**
 * @brief Perform Tls Recv.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_recv(SNEPPXTlsSocket* sock, unsigned char* buf, size_t max_len);
#ifdef __cplusplus
}
#endif
#endif
