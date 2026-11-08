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
 *   http_server_demo [port] [api_key_db]
 *
 * Endpoints:
 *   GET  /v1/health            (public)
 *   GET  /v1/models            (public)
 *   GET  /v1/models/{id}       (public)
 *   POST /v1/generate          (requires Authorization: Bearer <key>)
 *
 * In auth stub mode (no SQLite), any key with the "sk-sneppx-" prefix
 * (or the admin key "sneppx-admin") is accepted for development.
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

    SNEPPX_HttpAuth* auth = SNEPPX_http_auth_create(db_path);
    if (auth) {
        SNEPPX_http_auth_add_public_path(auth, "/v1/health");
        SNEPPX_http_auth_add_public_path(auth, "/v1/models");
        SNEPPX_http_server_add_middleware(srv, SNEPPX_http_auth_middleware(auth), auth);
    }

    printf("SneppX C HTTP server listening on port %d\n", port);
    printf("  GET  /v1/health         (public)\n");
    printf("  GET  /v1/models         (public)\n");
    printf("  GET  /v1/models/{id}    (public)\n");
    printf("  POST /v1/generate       (auth: Bearer sk-sneppx-...)\n");
    printf("Press Ctrl+C to stop.\n\n");

    int ret = SNEPPX_http_server_start(srv);

    SNEPPX_http_auth_destroy(auth);
    SNEPPX_http_api_destroy(api);
    SNEPPX_http_server_destroy(srv);
    return ret == 0 ? 0 : 1;
}
