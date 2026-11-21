#ifndef SNEPPX_AZURE_BLOB_H
#define SNEPPX_AZURE_BLOB_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Azure Blob
 *
 * WHAT
 *   Azure Blob.
 *
 * CONCEPT
 *   Provides the Azure Blob.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Azure Blob.
 *
 * @param connection_string [in] Connection String value.
 * @param container_name [in] Container Name value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_azure_blob_create(const char* connection_string, const char* container_name);
/**
 * @brief Destroy Azure Blob.
 *
 * @param client [out] Client value.
 */
void SNEPPX_azure_blob_destroy(void* client);
/**
 * @brief Load Azure Blob Up.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param content_type [in] Content Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_upload(void* client, const char* blob_name, const unsigned char* data, size_t len, const char* content_type);
/**
 * @brief Load Azure Blob Down.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_download(void* client, const char* blob_name, unsigned char** data, size_t* len);
/**
 * @brief Perform Azure Blob Delete.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_delete(void* client, const char* blob_name);
/**
 * @brief Perform Azure Blob List.
 *
 * @param client [out] Client value.
 * @param prefix [in] Prefix value.
 * @param blobs [out] Blobs value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_list(void* client, const char* prefix, char*** blobs, size_t* count);
/**
 * @brief Perform Azure Blob Exists.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_exists(void* client, const char* blob_name);
/**
 * @brief Perform Azure Blob Get Url.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param url [out] Url value.
 * @param url_max [in] Url Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_get_url(void* client, const char* blob_name, char* url, size_t url_max);
#ifdef __cplusplus
}
#endif
#endif
