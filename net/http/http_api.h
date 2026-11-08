#ifndef SNEPPX_HTTP_API_H
#define SNEPPX_HTTP_API_H

#include "http_server.h"

#ifdef __cplusplus
/*
 * SNEPPX - Http Api
 *
 * WHAT
 *   Http Api.
 *
 * CONCEPT
 *   Provides the Http Api.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Higher-level REST API layer on top of SNEPPX_HttpServer.
 *
 * Registers the standard SneppX serving endpoints:
 *   GET  /v1/health          - liveness + version + uptime
 *   GET  /v1/models          - list known model presets
 *   GET  /v1/models/{id}     - single model configuration
 *   POST /v1/generate        - generate text from a prompt (reference scaffold)
 *
 * The generate handler parses the JSON request body and produces a
 * deterministic continuation (char-code/LCG reference path). It does NOT run
 * a trained model forward pass or load weights; responses include mode/warning
 * metadata so callers can detect scaffold behavior.
 */

typedef struct SNEPPX_HttpApi SNEPPX_HttpApi;

/* Create API state. version is copied into the state (may be NULL -> "dev").
 * start time is captured for uptime reporting. */
/**
 * @brief Create Http Api.
 *
 * @param version [in] Version value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_HttpApi* SNEPPX_http_api_create(const char* version);

/* Destroy API state. */
/**
 * @brief Destroy Http Api.
 *
 * @param api [out] Api value.
 */
void SNEPPX_http_api_destroy(SNEPPX_HttpApi* api);

/* Register all /v1/* endpoints on an existing server.
 * Returns 0 on success, -1 if a route could not be registered. */
/**
 * @brief Perform Http Api Register.
 *
 * @param srv [out] Srv value.
 * @param api [out] Api value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_api_register(SNEPPX_HttpServer* srv, SNEPPX_HttpApi* api);

/* Number of known model presets reported by /v1/models. */
/**
 * @brief Perform Http Api Model Count.
 *
 * @param api [out] Api value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_api_model_count(SNEPPX_HttpApi* api);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HTTP_API_H */
