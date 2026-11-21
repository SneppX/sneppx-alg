#ifndef SNEPPX_WEBSOCKET_CLIENT_H
#define SNEPPX_WEBSOCKET_CLIENT_H
#include <stddef.h>
#ifdef __cplusplus
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


extern "C" {
#endif
/**
 * @brief Perform Ws Connect.
 *
 * @param url [in] Url value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_ws_connect(const char* url, int timeout_ms);
/**
 * @brief Perform Ws Disconnect.
 *
 * @param ws [out] Ws value.
 */
void SNEPPX_ws_disconnect(void* ws);
/**
 * @brief Perform Ws Send Text.
 *
 * @param ws [out] Ws value.
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_text(void* ws, const char* text, size_t len);
/**
 * @brief Perform Ws Send Binary.
 *
 * @param ws [out] Ws value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_binary(void* ws, const unsigned char* data, size_t len);
/**
 * @brief Perform Ws Send Ping.
 *
 * @param ws [out] Ws value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send_ping(void* ws);
/**
 * @brief Perform Ws Recv.
 *
 * @param ws [out] Ws value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param opcode [out] Opcode value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_recv(void* ws, unsigned char** data, size_t* len, int* opcode, int timeout_ms);
int SNEPPX_ws_set_on_message(void* ws, void (*cb)(void* ws, int opcode, const unsigned char* data, size_t len));
int SNEPPX_ws_set_on_error(void* ws, void (*cb)(void* ws, int error_code));
/**
 * @brief Perform Ws Is Connected.
 *
 * @param ws [out] Ws value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_is_connected(void* ws);
#ifdef __cplusplus
}
#endif
#endif
