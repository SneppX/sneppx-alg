#include "http_server.h"
#include "http_auth.h"
#include "http_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
  #define snepx_sleep_ms(ms) Sleep(ms)
#else
  #include <unistd.h>
  #define snepx_sleep_ms(ms) usleep((ms) * 1000)
#endif

/*
 * SNEPPX - Http Server Demo
 *
 * WHAT
 *   Http Server Demo.
 *
 * CONCEPT
 *   Provides the Http Server Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Demo: run the SneppX C HTTP server with the standard /v1/* REST endpoints.
 *
 * Usage:
 *   http_server_demo [port] [api_key_db] [static_dir]
 *
 * Endpoints:
 *   GET  /v1/health            (public)
 *   GET  /v1/models            (public)
 *   GET  /v1/models/{id}       (public)
 *   POST /v1/generate          (requires Authorization: Bearer <key>)
 *
 * A protected API key must be supplied through SNEPPX_HTTP_API_KEY.
 * No development-prefix or built-in administrator key is accepted.
 */

int main(int argc, char** argv) {
    int port = 8080;
    const char* db_path = NULL;
    if (argc > 1) port = atoi(argv[1]);
    if (argc > 2) db_path = argv[2];

    SNEPPX_HttpServer* srv = SNEPPX_http_server_create(port, 4);
    if (!srv) {
        printf("ERROR: failed to create server\n");
        return 1;
    }

    SNEPPX_HttpApi* api = SNEPPX_http_api_create("1.5.2");
    if (!api) {
        printf("ERROR: failed to create API state\n");
        SNEPPX_http_server_destroy(srv);
        return 1;
    }

    if (SNEPPX_http_api_register(srv, api) != 0) {
        printf("ERROR: failed to register API routes\n");
        SNEPPX_http_api_destroy(api);
        SNEPPX_http_server_destroy(srv);
        return 1;
    }

    if (argc > 3 && argv[3][0] != '\0' &&
        SNEPPX_http_server_add_static_dir(srv, "/static", argv[3]) != 0) {
        fprintf(stderr, "ERROR: failed to register static directory\n");
        SNEPPX_http_api_destroy(api);
        SNEPPX_http_server_destroy(srv);
        return 1;
    }

    SNEPPX_HttpAuth* auth = SNEPPX_http_auth_create(db_path);
    const char* api_key = getenv("SNEPPX_HTTP_API_KEY");
    if (!auth || !api_key || SNEPPX_http_auth_add_key(auth, api_key) != 0 ||
        SNEPPX_http_auth_add_public_path(auth, "/v1/health") != 0 ||
        SNEPPX_http_auth_add_public_path(auth, "/v1/models") != 0 ||
        SNEPPX_http_server_add_middleware(srv, SNEPPX_http_auth_middleware(auth), auth) != 0) {
        fprintf(stderr, "ERROR: HTTP authentication setup failed; set SNEPPX_HTTP_API_KEY to a 16-256 byte key\n");
        SNEPPX_http_auth_destroy(auth);
        SNEPPX_http_api_destroy(api);
        SNEPPX_http_server_destroy(srv);
        return 1;
    }

    printf("SneppX C HTTP server listening on 127.0.0.1:%d (loopback only; no TLS)\n", port);
    printf("For remote access, place a TLS reverse proxy in front of this listener.\n");
    if (argc > 3 && argv[3][0] != '\0')
        printf("Static files: http://127.0.0.1:%d/static/\n", port);
    printf("  GET  /v1/health         (public)\n");
    printf("  GET  /v1/models         (public)\n");
    printf("  GET  /v1/models/{id}    (public)\n");
    printf("  POST /v1/generate       (auth: Bearer <configured key>)\n");
    printf("Press Ctrl+C to stop.\n\n");

    int ret = SNEPPX_http_server_start(srv);

    SNEPPX_http_auth_destroy(auth);
    SNEPPX_http_api_destroy(api);
    SNEPPX_http_server_destroy(srv);
    return ret == 0 ? 0 : 1;
}
