#ifndef SNEPPX_QUIC_TRANSPORT_H
#define SNEPPX_QUIC_TRANSPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
/*
 * SNEPPX - Quic Transport
 *
 * WHAT
 *   Quic Transport.
 *
 * CONCEPT
 *   Provides the Quic Transport.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Quic Create Listener.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param cert_path [in] Cert Path value.
 * @param key_path [in] Key Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_create_listener(const char* host, int port, const char* cert_path, const char* key_path);
/**
 * @brief Perform Quic Destroy Listener.
 *
 * @param listener [out] Listener value.
 */
void SNEPPX_quic_destroy_listener(void* listener);
/**
 * @brief Perform Quic Listen.
 *
 * @param listener [out] Listener value.
 * @param backlog [in] Backlog value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_listen(void* listener, int backlog);
/**
 * @brief Perform Quic Accept.
 *
 * @param listener [out] Listener value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_accept(void* listener, int timeout_ms);
/**
 * @brief Perform Quic Dial.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_dial(const char* host, int port, int timeout_ms);
/**
 * @brief Close Quic.
 *
 * @param conn [out] Conn value.
 */
void SNEPPX_quic_close(void* conn);
/**
 * @brief Perform Quic Send.
 *
 * @param conn [out] Conn value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_send(void* conn, const unsigned char* data, size_t len);
/**
 * @brief Perform Quic Recv.
 *
 * @param conn [out] Conn value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_recv(void* conn, unsigned char* buf, size_t max_len, int timeout_ms);
/**
 * @brief Open Quic Stream.
 *
 * @param conn [out] Conn value.
 * @param stream_id [out] Stream Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_open(void* conn, uint64_t* stream_id);
/**
 * @brief Close Quic Stream.
 *
 * @param conn [out] Conn value.
 * @param stream_id [in] Stream Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_close(void* conn, uint64_t stream_id);
/**
 * @brief Perform Quic Stream Send.
 *
 * @param conn [out] Conn value.
 * @param stream_id [in] Stream Id value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_send(void* conn, uint64_t stream_id, const unsigned char* data, size_t len);
/**
 * @brief Perform Quic Stream Recv.
 *
 * @param conn [out] Conn value.
 * @param stream_id [in] Stream Id value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_recv(void* conn, uint64_t stream_id, unsigned char* buf, size_t max_len, int timeout_ms);
#ifdef __cplusplus
}
#endif
#endif
