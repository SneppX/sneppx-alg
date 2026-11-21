#include "socket_comm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
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


  typedef int socklen_t;
  #define CLOSE_SOCKET(fd) closesocket(fd)
  #define ISVALIDSOCK(fd) ((fd) != INVALID_SOCKET)
  #define SOCKERR WSAGetLastError()
  #define SNEPPX_EWOULDBLOCK WSAEWOULDBLOCK
  #pragma comment(lib, "ws2_32.lib")
#else
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #define CLOSE_SOCKET(fd) close(fd)
  #define ISVALIDSOCK(fd) ((fd) >= 0)
  #define SOCKERR errno
  #define SNEPPX_EWOULDBLOCK EWOULDBLOCK
  #define SOCKET_ERROR (-1)
  #define INVALID_SOCKET (-1)
#endif

static int _wsa_inited = 0;

static int ensure_wsa(void) {
#ifdef _WIN32
    if (!_wsa_inited) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) return -1;
        _wsa_inited = 1;
    }
#endif
    return 0;
}

/**
 * @brief Create Socket.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSocket* SNEPPX_socket_create(SNEPPXSocketType type) {
    if (ensure_wsa() != 0) return NULL;
    SNEPPXSocket* sock = (SNEPPXSocket*)calloc(1, sizeof(SNEPPXSocket));
    if (!sock) return NULL;
    sock->type = type;
    sock->fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (!ISVALIDSOCK(sock->fd)) { free(sock); return NULL; }
    sock->send_buf_size = 65536;
    sock->recv_buf_size = 65536;
    sock->timeout_ms = -1;
    return sock;
}

/**
 * @brief Destroy Socket.
 */
void SNEPPX_socket_destroy(SNEPPXSocket* sock) {
    if (!sock) return;
    if (ISVALIDSOCK(sock->fd)) CLOSE_SOCKET(sock->fd);
    free(sock);
}

/**
 * @brief Perform Socket Bind.
 *
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_bind(SNEPPXSocket* sock, int port) {
    if (!sock || !ISVALIDSOCK(sock->fd)) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port);
    int opt = 1;
    setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    if (bind(sock->fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) return -1;
    sock->port = port;
    return 0;
}

/**
 * @brief Perform Socket Listen.
 *
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_listen(SNEPPXSocket* sock, int backlog) {
    if (!sock || !ISVALIDSOCK(sock->fd)) return -1;
    if (listen(sock->fd, backlog > 0 ? backlog : 5) == SOCKET_ERROR) return -1;
    sock->is_listening = 1;
    return 0;
}

/**
 * @brief Perform Socket Accept.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSocket* SNEPPX_socket_accept(SNEPPXSocket* server_sock) {
    if (!server_sock || !ISVALIDSOCK(server_sock->fd)) return NULL;
    struct sockaddr_in client_addr;
    socklen_t addrlen = sizeof(client_addr);
    int client_fd = (int)accept(server_sock->fd, (struct sockaddr*)&client_addr, &addrlen);
    if (!ISVALIDSOCK(client_fd)) return NULL;
    SNEPPXSocket* client = (SNEPPXSocket*)calloc(1, sizeof(SNEPPXSocket));
    if (!client) { CLOSE_SOCKET(client_fd); return NULL; }
    client->type = server_sock->type;
    client->fd = client_fd;
    client->is_connected = 1;
    client->port = ntohs(client_addr.sin_port);
    inet_ntop(AF_INET, &client_addr.sin_addr, client->hostname, sizeof(client->hostname));
    client->timeout_ms = server_sock->timeout_ms;
    return client;
}

/**
 * @brief Perform Socket Connect.
 *
 * @param sock [out] Sock value.
 * @param host [in] Host value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_connect(SNEPPXSocket* sock, const char* host, int port) {
    if (!sock || !ISVALIDSOCK(sock->fd) || !host) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        struct addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host, NULL, &hints, &res) != 0 || !res) return -1;
        addr.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;
        freeaddrinfo(res);
    }
    if (connect(sock->fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) return -1;
    snprintf(sock->hostname, sizeof(sock->hostname), "%s", host);
    sock->port = port;
    sock->is_connected = 1;
    return 0;
}

static int send_all(int fd, const void* data, size_t len) {
    const char* ptr = (const char*)data;
    size_t remaining = len;
    while (remaining > 0) {
        int sent = (int)send(fd, ptr, (int)remaining, 0);
        if (sent <= 0) return -1;
        ptr += sent;
        remaining -= (size_t)sent;
    }
    return 0;
}

static int recv_all(int fd, void* buf, size_t len) {
    char* ptr = (char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        int n = (int)recv(fd, ptr, (int)remaining, 0);
        if (n <= 0) return -1;
        ptr += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/**
 * @brief Perform Socket Send.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_send(SNEPPXSocket* sock, const void* data, size_t len) {
    if (!sock || !ISVALIDSOCK(sock->fd) || !data) return -1;
    if (send_all(sock->fd, data, len) != 0) return -1;
    sock->bytes_sent += len;
    return (int)len;
}

/**
 * @brief Perform Socket Recv.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_recv(SNEPPXSocket* sock, void* buf, size_t len) {
    if (!sock || !ISVALIDSOCK(sock->fd) || !buf) return -1;
    int n = (int)recv(sock->fd, (char*)buf, (int)len, 0);
    if (n <= 0) return -1;
    sock->bytes_recv += (size_t)n;
    return n;
}

/**
 * @brief Perform Socket Send Message.
 *
 * @param sock [out] Sock value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_send_message(SNEPPXSocket* sock, const void* data, size_t len) {
    if (!sock || !data) return -1;
    uint32_t nlen = (uint32_t)len;
    nlen = htonl(nlen);
    if (send_all(sock->fd, &nlen, sizeof(nlen)) != 0) return -1;
    return SNEPPX_socket_send(sock, data, len);
}

/**
 * @brief Perform Socket Recv Message.
 *
 * @param sock [out] Sock value.
 * @param buf [out] Buf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_recv_message(SNEPPXSocket* sock, void** buf, size_t* len) {
    if (!sock || !buf || !len) return -1;
    uint32_t nlen;
    if (recv_all(sock->fd, &nlen, sizeof(nlen)) != 0) return -1;
    nlen = ntohl(nlen);
    *buf = malloc((size_t)nlen);
    if (!*buf) return -1;
    if (recv_all(sock->fd, *buf, (size_t)nlen) != 0) { free(*buf); *buf = NULL; return -1; }
    *len = (size_t)nlen;
    return 0;
}

/**
 * @brief Close Socket.
 */
void SNEPPX_socket_close(SNEPPXSocket* sock) {
    if (!sock) return;
    if (ISVALIDSOCK(sock->fd)) {
        CLOSE_SOCKET(sock->fd);
        sock->fd = INVALID_SOCKET;
    }
    sock->is_connected = 0;
    sock->is_listening = 0;
}

/**
 * @brief Perform Socket Send Tensor.
 *
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_send_tensor(SNEPPXSocket* sock, const void* tensor_handle) {
    (void)tensor_handle;
    if (!sock || !sock->is_connected) return -1;
    return 0;
}

/**
 * @brief Perform Socket Recv Tensor.
 *
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_recv_tensor(SNEPPXSocket* sock, void** tensor_handle) {
    if (!sock || !tensor_handle) return -1;
    *tensor_handle = NULL;
    return 0;
}

/**
 * @brief Perform Socket Set Timeout.
 *
 * @param sock [out] Sock value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_socket_set_timeout(SNEPPXSocket* sock, int ms) {
    if (!sock || !ISVALIDSOCK(sock->fd)) return -1;
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    if (setsockopt(sock->fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv)) != 0) return -1;
    if (setsockopt(sock->fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv)) != 0) return -1;
    sock->timeout_ms = ms;
    return 0;
}

/**
 * @brief Perform Socket Error String.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_socket_error_string(int err) {
    (void)err;
#ifdef _WIN32
    static char buf[128];
    wchar_t* ws = NULL;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, WSAGetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&ws, 0, NULL);
    if (ws) {
        wcstombs(buf, ws, sizeof(buf) - 1);
        LocalFree(ws);
    } else {
        snprintf(buf, sizeof(buf), "err=%d", WSAGetLastError());
    }
    return buf;
#else
    return strerror(errno);
#endif
}
