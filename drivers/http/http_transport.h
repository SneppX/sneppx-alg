#ifndef SNEPPX_HTTP_TRANSPORT_H
#define SNEPPX_HTTP_TRANSPORT_H

#include <stddef.h>

#ifdef __cplusplus
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


extern "C" {
#endif

/* HTTP inference transport. When SNEPPX_BUILD_HTTP is defined this provides a
 * real, dependency-free client (GET/POST) and a minimal blocking server built
 * on BSD sockets. Without the flag every entry point reports
 * SNEPPX_DRIVER_UNSUPPORTED. */

/**
 * @brief Initialize Http.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_http_init(void);
/**
 * @brief Perform Http Shutdown.
 */
void SNEPPX_http_shutdown(void);

/* Perform an HTTP GET against host:port/path. Response (status line + headers
 * + body) is written to `out` (at most out_max bytes). Returns 0 on success. */
/**
 * @brief Get Http.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param out [out] Out value.
 * @param out_max [in] Out Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_get(const char* host, int port, const char* path,
                    char* out, size_t out_max);

/* Perform an HTTP POST with `body` (content-type application/json). */
/**
 * @brief Perform Http Post.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param path [in] Path value.
 * @param body [in] Body value.
 * @param out [out] Out value.
 * @param out_max [in] Out Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_post(const char* host, int port, const char* path,
                     const char* body, char* out, size_t out_max);

/* A request handler receives the method, path and body and must write a
 * response (status line + headers + body) into `resp` (resp_max bytes). */
typedef void (*SNEPPX_http_handler)(const char* method, const char* path,
                                    const char* body, char* resp, size_t resp_max);

/* Legacy simple server — use net/http/http_server.h instead for full routing,
 * middleware, and thread pool support. These are only available when
 * SNEPPX_USE_TRANSPORT_SERVER is defined. */
#ifdef SNEPPX_USE_TRANSPORT_SERVER
/**
 * @brief Create Http Server.
 *
 * @param port [in] Port value.
 * @param handler [in] Handler value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_http_server_create(int port, SNEPPX_http_handler handler);
/**
 * @brief Run Http Server.
 *
 * @param server [out] Server value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_http_server_run(void* server);
/**
 * @brief Stop Http Server.
 *
 * @param server [out] Server value.
 */
void  SNEPPX_http_server_stop(void* server);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_TRANSPORT_H */
