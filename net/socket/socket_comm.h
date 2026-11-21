#ifndef SNEPPX_SOCKET_COMM_H
#define SNEPPX_SOCKET_COMM_H
/*
 * SNEPPX - Socket Comm
 *
 * WHAT
 *   Socket Comm.
 *
 * CONCEPT
 *   Provides the Socket Comm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Socket Communication — v1.0 (distributed training transport)
 *
 * PURPOSE: Low-level TCP socket abstraction for point-to-point tensor
 * transfers between training nodes.  Provides blocking and non-blocking
 * send/recv with length-prefixed message framing.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_SOCKET_TCP,
    SNEPPX_SOCKET_UNIX,
} SNEPPXSocketType;

typedef struct {
    int            fd;
    SNEPPXSocketType type;
    int            port;
    char           hostname[256];
    size_t         send_buf_size;
    size_t         recv_buf_size;
    int            is_connected;
    int            is_listening;
    uint64_t       bytes_sent;
    uint64_t       bytes_recv;
    int            timeout_ms;
} SNEPPXSocket;

/* ---------- Lifecycle ---------- */
/**
 * @brief Create Socket.
 *
 * @param type [in] Type value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSocket* SNEPPX_socket_create(SNEPPXSocketType type);
/**
 * @brief Destroy Socket.
 *
 * @param sock [out] Sock value.
 */
void        SNEPPX_socket_destroy(SNEPPXSocket* sock);

/* ---------- Server ---------- */
/**
 * @brief Perform Socket Bind.
 *
 * @param sock [out] Sock value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_bind(SNEPPXSocket* sock, int port);
/**
 * @brief Perform Socket Listen.
 *
 * @param sock [out] Sock value.
 * @param backlog [in] Backlog value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_listen(SNEPPXSocket* sock, int backlog);
/**
 * @brief Perform Socket Accept.
 *
 * @param server_sock [out] Server Sock value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSocket* SNEPPX_socket_accept(SNEPPXSocket* server_sock);

/* ---------- Client ---------- */
/**
 * @brief Perform Socket Connect.
 *
 * @param sock [out] Sock value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_connect(SNEPPXSocket* sock, const char* host, int port);

/* ---------- I/O ---------- */
/**
 * @brief Perform Socket Send.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_socket_send(SNEPPXSocket* sock, const void* data, size_t len);
/**
 * @brief Perform Socket Recv.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_socket_recv(SNEPPXSocket* sock, void* buf, size_t len);
/**
 * @brief Perform Socket Send Message.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_socket_send_message(SNEPPXSocket* sock, const void* data, size_t len);
/**
 * @brief Perform Socket Recv Message.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 * @param len [out] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_socket_recv_message(SNEPPXSocket* sock, void** buf, size_t* len);
/**
 * @brief Close Socket.
 *
 * @param sock [out] Sock value.
 */
void SNEPPX_socket_close(SNEPPXSocket* sock);

/* ---------- Tensor helpers (v1.0) ---------- */
/**
 * @brief Perform Socket Send Tensor.
 *
 * @param sock [out] Sock value.
 * @param tensor_handle [in] Tensor Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_send_tensor(SNEPPXSocket* sock, const void* tensor_handle);
/**
 * @brief Perform Socket Recv Tensor.
 *
 * @param sock [out] Sock value.
 * @param tensor_handle [out] Tensor Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_recv_tensor(SNEPPXSocket* sock, void** tensor_handle);

/* ---------- Utility ---------- */
/**
 * @brief Perform Socket Set Timeout.
 *
 * @param sock [out] Sock value.
 * @param ms [in] Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int         SNEPPX_socket_set_timeout(SNEPPXSocket* sock, int ms);
/**
 * @brief Perform Socket Error String.
 *
 * @param err [in] Err value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_socket_error_string(int err);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_SOCKET_COMM_H */
