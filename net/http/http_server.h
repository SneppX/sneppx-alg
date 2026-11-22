#ifndef SNEPPX_HTTP_SERVER_H
#define SNEPPX_HTTP_SERVER_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Http Server
 *
 * WHAT
 *   Http Server.
 *
 * CONCEPT
 *   Provides the Http Server.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

extern "C"
{
#endif

    /* HTTP server types */
    typedef struct SNEPPX_HttpServer SNEPPX_HttpServer;
    typedef struct SNEPPX_HttpRequest SNEPPX_HttpRequest;
    typedef struct SNEPPX_HttpResponse SNEPPX_HttpResponse;
    typedef struct SNEPPX_HttpRoute SNEPPX_HttpRoute;
    typedef struct SNEPPX_HttpMiddleware SNEPPX_HttpMiddleware;

    /* Request handler: process a request and populate the response.
     * Return 0 on success, non-zero to abort with an error. */
    typedef int (*SNEPPX_http_handler_fn)(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata);

    /* Middleware function: called before handlers. Return 0 to continue, non-zero to short-circuit. */
    typedef int (*SNEPPX_http_middleware_fn)(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata);

    /* Server lifecycle. The built-in listener binds to loopback only and does not provide TLS.
     * Use a separately configured TLS reverse proxy for remote access. */
    /**
     * @brief Create Http Server.
     *
     * @param port [in] Port value.
     * @param num_threads [in] Num Threads value.
     *
     * @return Pointer on success, NULL on error.
     */
    SNEPPX_HttpServer* SNEPPX_http_server_create(int port, int num_threads);
    /**
     * @brief Start Http Server.
     *
     * @param srv [out] Srv value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_http_server_start(SNEPPX_HttpServer* srv);
    /**
     * @brief Stop Http Server.
     *
     * @param srv [out] Srv value.
     */
    void SNEPPX_http_server_stop(SNEPPX_HttpServer* srv);
    /**
     * @brief Destroy Http Server.
     *
     * @param srv [out] Srv value.
     */
    void SNEPPX_http_server_destroy(SNEPPX_HttpServer* srv);

    /* Route registration */
    /**
     * @brief Perform Http Server Add Route.
     *
     * @param srv [out] Srv value.
     * @param method [in] Method value.
     * @param path [in] Path value.
     * @param handler [in] Handler value.
     * @param userdata [out] Userdata value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_http_server_add_route(SNEPPX_HttpServer* srv, const char* method, const char* path,
                                     SNEPPX_http_handler_fn handler, void* userdata);
    /**
     * @brief Perform Http Server Add Static Dir.
     *
     * @param srv [out] Srv value.
     * @param url_prefix [in] Url Prefix value.
     * @param dir_path [in] Dir Path value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_http_server_add_static_dir(SNEPPX_HttpServer* srv, const char* url_prefix, const char* dir_path);

    /* Middleware */
    /**
     * @brief Perform Http Server Add Middleware.
     *
     * @param srv [out] Srv value.
     * @param mw [in] Mw value.
     * @param userdata [out] Userdata value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_http_server_add_middleware(SNEPPX_HttpServer* srv, SNEPPX_http_middleware_fn mw, void* userdata);

    /* Request framing: one request per connection, Content-Length only, 16 KiB headers and 32 KiB body. */
    /* Request accessors */
    /**
     * @brief Perform Http Request Method.
     *
     * @param req [out] Req value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_method(SNEPPX_HttpRequest* req);
    /**
     * @brief Perform Http Request Path.
     *
     * @param req [out] Req value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_path(SNEPPX_HttpRequest* req);
    /**
     * @brief Perform Http Request Header.
     *
     * @param req [out] Req value.
     * @param name [in] Name value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_header(SNEPPX_HttpRequest* req, const char* name);
    /**
     * @brief Perform Http Request Body.
     *
     * @param req [out] Req value.
     * @param len [out] Len value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_body(SNEPPX_HttpRequest* req, size_t* len);
    /**
     * @brief Perform Http Request Query.
     *
     * @param req [out] Req value.
     * @param key [in] Key value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_query(SNEPPX_HttpRequest* req, const char* key);
    /**
     * @brief Perform Http Request Userdata.
     *
     * @param req [out] Req value.
     *
     * @return Pointer on success, NULL on error.
     */
    void* SNEPPX_http_request_userdata(SNEPPX_HttpRequest* req);
    /**
     * @brief Perform Http Request Param.
     *
     * @param req [out] Req value.
     * @param name [in] Name value.
     *
     * @return Pointer on success, NULL on error.
     */
    const char* SNEPPX_http_request_param(SNEPPX_HttpRequest* req, const char* name);

    /* Response setters */
    /**
     * @brief Perform Http Response Set Status.
     *
     * @param resp [out] Resp value.
     * @param status_code [in] Status Code value.
     */
    void SNEPPX_http_response_set_status(SNEPPX_HttpResponse* resp, int status_code);
    /**
     * @brief Perform Http Response Set Header.
     *
     * @param resp [out] Resp value.
     * @param name [in] Name value.
     * @param value [in] Value value.
     */
    void SNEPPX_http_response_set_header(SNEPPX_HttpResponse* resp, const char* name, const char* value);
    /**
     * @brief Perform Http Response Set Body.
     *
     * @param resp [out] Resp value.
     * @param data [in] Data value.
     * @param len [in] Len value.
     */
    void SNEPPX_http_response_set_body(SNEPPX_HttpResponse* resp, const char* data, size_t len);
    /**
     * @brief Perform Http Response Set Body Str.
     *
     * @param resp [out] Resp value.
     * @param str [in] Str value.
     */
    void SNEPPX_http_response_set_body_str(SNEPPX_HttpResponse* resp, const char* str);
    /**
     * @brief Perform Http Response Set Json.
     *
     * @param resp [out] Resp value.
     * @param json_str [in] Json Str value.
     */
    void SNEPPX_http_response_set_json(SNEPPX_HttpResponse* resp, const char* json_str);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_SERVER_H */
