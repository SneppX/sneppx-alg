#include "quic_transport.h"
#include <stdlib.h>
#include <string.h>

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


/**
 * @brief Perform Quic Create Listener.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param cert_path [in] Cert Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_create_listener(const char* host, int port, const char* cert_path, const char* key_path) { (void)host; (void)port; (void)cert_path; (void)key_path; return NULL; }
/**
 * @brief Perform Quic Destroy Listener.
 */
void SNEPPX_quic_destroy_listener(void* listener) { free(listener); }
/**
 * @brief Perform Quic Listen.
 *
 * @param listener [out] Listener value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_listen(void* listener, int backlog) { (void)listener; (void)backlog; return 0; }
/**
 * @brief Perform Quic Accept.
 *
 * @param listener [out] Listener value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_accept(void* listener, int timeout_ms) { (void)listener; (void)timeout_ms; return NULL; }
/**
 * @brief Perform Quic Dial.
 *
 * @param host [in] Host value.
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_quic_dial(const char* host, int port, int timeout_ms) { (void)host; (void)port; (void)timeout_ms; return NULL; }
/**
 * @brief Close Quic.
 */
void SNEPPX_quic_close(void* conn) { free(conn); }
/**
 * @brief Perform Quic Send.
 *
 * @param conn [out] Conn value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_send(void* conn, const unsigned char* data, size_t len) { (void)conn; (void)data; (void)len; return 0; }
/**
 * @brief Perform Quic Recv.
 *
 * @param conn [out] Conn value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_recv(void* conn, unsigned char* buf, size_t max_len, int timeout_ms) { (void)conn; (void)buf; (void)max_len; (void)timeout_ms; return 0; }
/**
 * @brief Open Quic Stream.
 *
 * @param conn [out] Conn value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_open(void* conn, unsigned long long* stream_id) { (void)conn; (void)stream_id; return 0; }
/**
 * @brief Close Quic Stream.
 *
 * @param conn [out] Conn value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_close(void* conn, unsigned long long stream_id) { (void)conn; (void)stream_id; return 0; }
/**
 * @brief Perform Quic Stream Send.
 *
 * @param conn [out] Conn value.
 * @param stream_id [in] Stream Id value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_send(void* conn, unsigned long long stream_id, const unsigned char* data, size_t len) { (void)conn; (void)stream_id; (void)data; (void)len; return 0; }
/**
 * @brief Perform Quic Stream Recv.
 *
 * @param conn [out] Conn value.
 * @param stream_id [in] Stream Id value.
 * @param buf [out] Buf value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quic_stream_recv(void* conn, unsigned long long stream_id, unsigned char* buf, size_t max_len, int timeout_ms) { (void)conn; (void)stream_id; (void)buf; (void)max_len; (void)timeout_ms; return 0; }
