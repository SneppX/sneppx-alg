#include "http_transport.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef SNEPPX_BUILD_HTTP

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
/*
 * SNEPPX - Http Transport
 *
 * WHAT
 *   Http Transport.
 *
 * CONCEPT
 *   Provides the Http Transport.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


  typedef SOCKET sock_t;
  #define SNEPPX_INVALID_SOCKET INVALID_SOCKET
  #define snepx_close closesocket
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <fcntl.h>
  typedef int sock_t;
  #define SNEPPX_INVALID_SOCKET (-1)
  #define snepx_close close
#endif

static int g_http_initialized = 0;

/**
 * @brief Initialize Http.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_init(void) {
#ifdef _WIN32
    if (g_http_initialized) return 0;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return SNEPPX_DRIVER_ERROR;
#endif
    g_http_initialized = 1;
    return 0;
}

/**
 * @brief Perform Http Shutdown.
 */
void SNEPPX_http_shutdown(void) {
#ifdef _WIN32
    if (g_http_initialized) { WSACleanup(); g_http_initialized = 0; }
#else
    g_http_initialized = 0;
#endif
}

static sock_t snepx_connect(const char* host, int port) {
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == SNEPPX_INVALID_SOCKET) return SNEPPX_INVALID_SOCKET;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
#ifdef _WIN32
    addr.sin_addr.s_addr = inet_addr(host);
#else
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { snepx_close(s); return SNEPPX_INVALID_SOCKET; }
#endif
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        snepx_close(s);
        return SNEPPX_INVALID_SOCKET;
    }
    return s;
}

static int snepx_recv_all(sock_t s, char* buf, size_t buf_max, size_t* got) {
    size_t total = 0;
    while (total < buf_max - 1) {
        int n = recv(s, buf + total, (int)(buf_max - 1 - total), 0);
        if (n <= 0) break;
        total += (size_t)n;
        /* stop at end of headers/body marker is caller's job; we read until
         * the server closes or buffer is full. */
        if (n < 4096) break;
    }
    buf[total] = '\0';
    if (got) *got = total;
    return 0;
}

static int snepx_request(const char* method, const char* host, int port,
                         const char* path, const char* body, char* out, size_t out_max) {
    if (SNEPPX_http_init() != 0) return SNEPPX_DRIVER_ERROR;
    sock_t s = snepx_connect(host, port);
    if (s == SNEPPX_INVALID_SOCKET) return SNEPPX_DRIVER_ERROR;
    char req[8192];
    int hdr_len = snprintf(req, sizeof(req),
        "%s %s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n", method, path, host, port);
    if (body) {
        hdr_len += snprintf(req + hdr_len, sizeof(req) - (size_t)hdr_len,
            "Content-Type: application/json\r\nContent-Length: %u\r\n\r\n",
            (unsigned)strlen(body));
        strncat(req, body, sizeof(req) - strlen(req) - 1);
    } else {
        strncat(req, "\r\n", sizeof(req) - strlen(req) - 1);
    }
    if (send(s, req, (int)strlen(req), 0) < 0) { snepx_close(s); return SNEPPX_DRIVER_ERROR; }
    size_t got = 0;
    snepx_recv_all(s, out, out_max, &got);
    snepx_close(s);
    return 0;
}

/**
 * @brief Get Http.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_get(const char* host, int port, const char* path, char* out, size_t out_max) {
    return snepx_request("GET", host, port, path, NULL, out, out_max);
}

/**
 * @brief Perform Http Post.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param body [in] Body value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_post(const char* host, int port, const char* path, const char* body, char* out, size_t out_max) {
    return snepx_request("POST", host, port, path, body, out, out_max);
}

/* Legacy simple HTTP server — replaced by net/http/http_server.c.
   To use this transport-level server instead, define SNEPPX_USE_TRANSPORT_SERVER. */
#ifdef SNEPPX_USE_TRANSPORT_SERVER
typedef struct {
    int port;
    SNEPPX_http_handler handler;
    volatile int stop;
    sock_t listen_fd;
} snepx_http_server;

/**
 * @brief Create Http Server.
 *
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_http_server_create(int port, SNEPPX_http_handler handler) {
    if (SNEPPX_http_init() != 0) return NULL;
    snepx_http_server* srv = (snepx_http_server*)calloc(1, sizeof(*srv));
    if (!srv) return NULL;
    srv->port = port;
    srv->handler = handler;
    srv->stop = 0;
    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd == SNEPPX_INVALID_SOCKET) { free(srv); return NULL; }
    int opt = 1;
#ifdef _WIN32
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);
    if (bind(srv->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        snepx_close(srv->listen_fd); free(srv); return NULL;
    }
    if (listen(srv->listen_fd, 8) != 0) {
        snepx_close(srv->listen_fd); free(srv); return NULL;
    }
    return srv;
}

/**
 * @brief Run Http Server.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_run(void* server) {
    snepx_http_server* srv = (snepx_http_server*)server;
    if (!srv || !srv->handler) return -1;
    while (!srv->stop) {
        sock_t c = accept(srv->listen_fd, NULL, NULL);
        if (c == SNEPPX_INVALID_SOCKET) { if (srv->stop) break; continue; }
        char buf[8192];
        int n = recv(c, buf, (int)sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            char method[16] = {0}, path[1024] = {0};
            sscanf(buf, "%15s %1023s", method, path);
            char* body = strstr(buf, "\r\n\r\n");
            const char* body_str = body ? body + 4 : "";
            char resp[16384];
            srv->handler(method, path, body_str, resp, sizeof(resp));
            send(c, resp, (int)strlen(resp), 0);
        }
        snepx_close(c);
    }
    return 0;
}

/**
 * @brief Stop Http Server.
 */
void SNEPPX_http_server_stop(void* server) {
    snepx_http_server* srv = (snepx_http_server*)server;
    if (!srv) return;
    srv->stop = 1;
    snepx_close(srv->listen_fd);
    free(srv);
}
#endif /* SNEPPX_USE_TRANSPORT_SERVER */

#else /* !SNEPPX_BUILD_HTTP — UNSUPPORTED stub */

/**
 * @brief Initialize Http.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_init(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Http Shutdown.
 */
void SNEPPX_http_shutdown(void) {}
/**
 * @brief Get Http.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_get(const char* host, int port, const char* path, char* out, size_t out_max) {
    (void)host; (void)port; (void)path; (void)out; (void)out_max;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
/**
 * @brief Perform Http Post.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param body [in] Body value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_post(const char* host, int port, const char* path, const char* body, char* out, size_t out_max) {
    (void)host; (void)port; (void)path; (void)body; (void)out; (void)out_max;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
#ifdef SNEPPX_USE_TRANSPORT_SERVER
/**
 * @brief Create Http Server.
 *
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_http_server_create(int port, SNEPPX_http_handler handler) {
    (void)port; (void)handler;
    return NULL;
}
/**
 * @brief Run Http Server.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_run(void* server) { (void)server; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Stop Http Server.
 */
void SNEPPX_http_server_stop(void* server) { (void)server; }
#endif /* SNEPPX_USE_TRANSPORT_SERVER */

#endif
