#ifndef SNEPPX_HTTP_CLIENT_H
#define SNEPPX_HTTP_CLIENT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Http Client
 *
 * WHAT
 *   Http Client.
 *
 * CONCEPT
 *   Provides the Http Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { char host[256]; int port; int use_ssl; void* conn; } SNEPPXHttpClient;
typedef struct { int status_code; char* headers; char* body; size_t body_len; } SNEPPXHttpResponse;
/**
 * @brief Initialize Http Client.
 *
 * @param client [out] Client value.
 * @param base_url [in] Base Url value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_client_init(SNEPPXHttpClient* client, const char* base_url, int timeout_ms);
/**
 * @brief Perform Http Client Cleanup.
 *
 * @param client [out] Client value.
 */
void SNEPPX_http_client_cleanup(SNEPPXHttpClient* client);
/**
 * @brief Get Http.
 *
 * @param client [out] Client value.
 * @param path [in] Path value.
 * @param resp [out] Resp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_get(SNEPPXHttpClient* client, const char* path, SNEPPXHttpResponse* resp);
/**
 * @brief Perform Http Post.
 *
 * @param client [out] Client value.
 * @param path [in] Path value.
 * @param body [in] Body value.
 * @param body_len [in] Body Len value.
 * @param content_type [in] Content Type value.
 * @param resp [out] Resp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_post(SNEPPXHttpClient* client, const char* path, const char* body, size_t body_len, const char* content_type, SNEPPXHttpResponse* resp);
/**
 * @brief Perform Http Put.
 *
 * @param client [out] Client value.
 * @param path [in] Path value.
 * @param body [in] Body value.
 * @param body_len [in] Body Len value.
 * @param resp [out] Resp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_put(SNEPPXHttpClient* client, const char* path, const char* body, size_t body_len, SNEPPXHttpResponse* resp);
/**
 * @brief Perform Http Delete.
 *
 * @param client [out] Client value.
 * @param path [in] Path value.
 * @param resp [out] Resp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_delete(SNEPPXHttpClient* client, const char* path, SNEPPXHttpResponse* resp);
/**
 * @brief Free Http Response.
 *
 * @param resp [out] Resp value.
 */
void SNEPPX_http_response_free(SNEPPXHttpResponse* resp);
#ifdef __cplusplus
}
#endif
#endif
