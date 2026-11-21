#ifndef SNEPPX_NETWORK_WEBSOCKET_H
#define SNEPPX_NETWORK_WEBSOCKET_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
/*
 * SNEPPX - Websocket
 *
 * WHAT
 *   Websocket.
 *
 * CONCEPT
 *   Provides the Websocket.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { void* handle; char url[512]; int connected; } SNEPPXWebSocket;
/**
 * @brief Open Ws.
 *
 * @param ws [out] Ws value.
 * @param url [in] Url value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_open(SNEPPXWebSocket* ws, const char* url);
/**
 * @brief Close Ws.
 *
 * @param ws [out] Ws value.
 */
void SNEPPX_ws_close(SNEPPXWebSocket* ws);
/**
 * @brief Perform Ws Send.
 *
 * @param ws [out] Ws value.
 * @param opcode [in] Opcode value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_send(SNEPPXWebSocket* ws, int opcode, const unsigned char* data, size_t len);
/**
 * @brief Perform Ws Recv.
 *
 * @param ws [out] Ws value.
 * @param opcode [out] Opcode value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_recv(SNEPPXWebSocket* ws, int* opcode, unsigned char** data, size_t* len, int timeout_ms);
/**
 * @brief Perform Ws Ping.
 *
 * @param ws [out] Ws value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ws_ping(SNEPPXWebSocket* ws);
typedef void (*SNEPPXWsOnMessage)(SNEPPXWebSocket* ws, int opcode, const unsigned char* data, size_t len, void* userdata);
/**
 * @brief Perform Ws Set Callback.
 *
 * @param ws [out] Ws value.
 * @param cb [in] Cb value.
 * @param userdata [out] Userdata value.
 */
void SNEPPX_ws_set_callback(SNEPPXWebSocket* ws, SNEPPXWsOnMessage cb, void* userdata);
#ifdef __cplusplus
}
#endif
#endif
