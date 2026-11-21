#ifndef SNEPPX_SD_CLIENT_H
#define SNEPPX_SD_CLIENT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Sd Client
 *
 * WHAT
 *   Sd Client.
 *
 * CONCEPT
 *   Provides the Sd Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Initialize Sd.
 *
 * @param service_name [in] Service Name value.
 * @param service_type [in] Service Type value.
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_sd_init(const char* service_name, const char* service_type, int port);
/**
 * @brief Destroy Sd.
 *
 * @param sd [out] Sd value.
 */
void SNEPPX_sd_destroy(void* sd);
/**
 * @brief Perform Sd Register.
 *
 * @param sd [out] Sd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_register(void* sd);
/**
 * @brief Perform Sd Unregister.
 *
 * @param sd [out] Sd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_unregister(void* sd);
/**
 * @brief Perform Sd Discover.
 *
 * @param sd [out] Sd value.
 * @param hosts [out] Hosts value.
 * @param ports [out] Ports value.
 * @param count [out] Count value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_discover(void* sd, char*** hosts, int** ports, size_t* count, int timeout_ms);
/**
 * @brief Perform Sd Resolve.
 *
 * @param name [in] Name value.
 * @param host [out] Host value.
 * @param host_max [in] Host Max value.
 * @param port [out] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_resolve(const char* name, char* host, size_t host_max, int* port);
/**
 * @brief Perform Sd Set Txt Record.
 *
 * @param sd [out] Sd value.
 * @param key [in] Key value.
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_set_txt_record(void* sd, const char* key, const char* value);
/**
 * @brief Perform Sd Get Txt Record.
 *
 * @param sd [out] Sd value.
 * @param key [in] Key value.
 * @param value [out] Value value.
 * @param value_max [in] Value Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_get_txt_record(void* sd, const char* key, char* value, size_t value_max);
#ifdef __cplusplus
}
#endif
#endif
