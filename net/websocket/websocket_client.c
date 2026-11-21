#include "websocket_client.h"
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Websocket Client
 *
 * WHAT
 *   Websocket Client.
 *
 * CONCEPT
 *   Provides the Websocket Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Ws Connect.
 *
 * @param url [in] Url value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_ws_connect(const char* url, int timeout_ms) { (void)url; (void)timeout_ms; return NULL; }
/**
 * @brief Perform Ws Disconnect.
 */
void SNEPPX_ws_disconnect(void* ws) { free(ws); }
/**
 * @brief Perform Ws Send Text.
 *
 * @param ws [out] Ws value.
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_text(void* ws, const char* text, size_t len) { (void)ws; (void)text; (void)len; return 0; }
/**
 * @brief Perform Ws Send Binary.
 *
 * @param ws [out] Ws value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_binary(void* ws, const unsigned char* data, size_t len) { (void)ws; (void)data; (void)len; return 0; }
/**
 * @brief Perform Ws Send Ping.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_ping(void* ws) { (void)ws; return 0; }
/**
 * @brief Perform Ws Recv.
 *
 * @param ws [out] Ws value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param opcode [out] Opcode value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_recv(void* ws, unsigned char** data, size_t* len, int* opcode, int timeout_ms) { (void)ws; (void)data; (void)len; (void)opcode; (void)timeout_ms; return 0; }
int SNEPPX_ws_set_on_message(void* ws, void (*cb)(void* ws, int opcode, const unsigned char* data, size_t len)) { (void)ws; (void)cb; return 0; }
int SNEPPX_ws_set_on_error(void* ws, void (*cb)(void* ws, int error_code)) { (void)ws; (void)cb; return 0; }
/**
 * @brief Perform Ws Is Connected.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_is_connected(void* ws) { (void)ws; return 0; }
