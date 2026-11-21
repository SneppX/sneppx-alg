#ifndef SNEPPX_TCP_TRANSPORT_H
#define SNEPPX_TCP_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
/*
 * SNEPPX - Tcp Transport
 *
 * WHAT
 *   Tcp Transport.
 *
 * CONCEPT
 *   Provides the Tcp Transport.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { char host[256]; int port; int fd; int is_listener; } SNEPPXTcpSocket;
/**
 * @brief Perform Tcp Listen.
 *
 * @param port [in] Port value.
 * @param backlog [in] Backlog value.
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_listen(int port, int backlog, SNEPPXTcpSocket* sock);
/**
 * @brief Perform Tcp Accept.
 *
 * @param listener [out] Listener value.
 * @param client [out] Client value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_accept(SNEPPXTcpSocket* listener, SNEPPXTcpSocket* client, int timeout_ms);
/**
 * @brief Perform Tcp Connect.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param timeout_ms [in] Timeout Ms value.
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_connect(const char* host, int port, int timeout_ms, SNEPPXTcpSocket* sock);
/**
 * @brief Close Tcp.
 *
 * @param sock [out] Sock value.
 */
void SNEPPX_tcp_close(SNEPPXTcpSocket* sock);
/**
 * @brief Perform Tcp Send.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_send(SNEPPXTcpSocket* sock, const unsigned char* data, size_t len, int timeout_ms);
/**
 * @brief Perform Tcp Recv.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_recv(SNEPPXTcpSocket* sock, unsigned char* buf, size_t max_len, int timeout_ms);
/**
 * @brief Perform Tcp Set Nodelay.
 *
 * @param sock [out] Sock value.
 * @param enable [in] Enable value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_set_nodelay(SNEPPXTcpSocket* sock, int enable);
/**
 * @brief Perform Tcp Set Keepalive.
 *
 * @param sock [out] Sock value.
 * @param enable [in] Enable value.
 * @param idle_s [in] Idle S value.
 * @param interval_s [in] Interval S value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tcp_set_keepalive(SNEPPXTcpSocket* sock, int enable, int idle_s, int interval_s, int count);
#ifdef __cplusplus
}
#endif
#endif
