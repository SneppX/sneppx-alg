#include "http_server.h"
#include "test_gtest.h"

static int test_http_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata)
{
    (void) req;
    (void) resp;
    (void) userdata;
    return 0;
}

static int test_http_middleware(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata)
{
    (void) req;
    (void) resp;
    (void) userdata;
    return 0;
}

TEST(test_http_server_api, suite)
{
    SX_CHECK(SNEPPX_http_server_create(0, 1) == NULL, "reject invalid server port");
    SX_CHECK(SNEPPX_http_server_create(65536, 1) == NULL, "reject out-of-range server port");
    SX_CHECK(SNEPPX_http_request_method(NULL) == NULL, "null method accessor");
    SX_CHECK(SNEPPX_http_request_path(NULL) == NULL, "null path accessor");
    SX_CHECK(SNEPPX_http_request_header(NULL, "Host") == NULL, "null header accessor");
    SX_CHECK(SNEPPX_http_request_query(NULL, "id") == NULL, "null query accessor");
    SX_CHECK(SNEPPX_http_request_userdata(NULL) == NULL, "null userdata accessor");
    SX_CHECK(SNEPPX_http_request_param(NULL, "id") == NULL, "null parameter accessor");
    size_t body_len = 99;
    SX_CHECK(SNEPPX_http_request_body(NULL, &body_len) == NULL && body_len == 0,
             "null body accessor resets output length");

    SNEPPX_http_response_set_status(NULL, 200);
    SNEPPX_http_response_set_header(NULL, "X-Test", "value");
    SNEPPX_http_response_set_body(NULL, "x", 1);
    SNEPPX_http_response_set_body(NULL, NULL, 0);
    SNEPPX_http_response_set_body_str(NULL, "x");
    SNEPPX_http_response_set_body_str(NULL, NULL);
    SNEPPX_http_response_set_json(NULL, "{}");
    SNEPPX_http_response_set_json(NULL, NULL);

    SNEPPX_HttpServer* srv = SNEPPX_http_server_create(49173, 2);
    SX_CHECK(srv != NULL, "create server state without binding");
    if (!srv)
        return;

    SX_CHECK(SNEPPX_http_server_add_route(NULL, "GET", "/health", test_http_handler, NULL) == -1, "reject null server");
    SX_CHECK(SNEPPX_http_server_add_route(srv, NULL, "/health", test_http_handler, NULL) == -1, "reject null method");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", NULL, test_http_handler, NULL) == -1, "reject null route path");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", "/health", NULL, NULL) == -1, "reject null handler");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "BAD METHOD", "/health", test_http_handler, NULL) == -1,
             "reject invalid method token");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", "/health?x=1", test_http_handler, NULL) == -1,
             "reject query in route pattern");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", "health", test_http_handler, NULL) == -1,
             "require absolute route path");
    char long_path[257];
    memset(long_path, 'a', sizeof(long_path));
    long_path[0] = '/';
    long_path[sizeof(long_path) - 1] = '\0';
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", long_path, test_http_handler, NULL) == -1,
             "reject overlong route path");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", "/health", test_http_handler, NULL) == 0, "register valid route");
    SX_CHECK(SNEPPX_http_server_add_route(srv, "GET", "/health", test_http_handler, NULL) == -1,
             "reject duplicate route");

    SX_CHECK(SNEPPX_http_server_add_middleware(srv, NULL, NULL) == -1, "reject null middleware");
    SX_CHECK(SNEPPX_http_server_add_middleware(srv, test_http_middleware, NULL) == 0, "register valid middleware");

    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, NULL, "/tmp") == -1, "reject null static prefix");
    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, "/", "/tmp") == -1, "reject root static prefix");
    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, "/static/", "/tmp") == -1, "reject trailing static prefix slash");
    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, "/static?x=1", "/tmp") == -1, "reject query in static prefix");
    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, "/static", "/tmp") == 0, "register valid static directory");
    SX_CHECK(SNEPPX_http_server_add_static_dir(srv, "/static", "/tmp/other") == -1, "reject duplicate static prefix");

    SNEPPX_http_server_destroy(srv);
}
