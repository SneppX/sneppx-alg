#include "azure_blob.h"
#include <stdlib.h>
#include <string.h>

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


/**
 * @brief Create Azure Blob.
 *
 * @param connection_string [in] Connection String value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_azure_blob_create(const char* connection_string, const char* container_name) { (void)connection_string; (void)container_name; return NULL; }
/**
 * @brief Destroy Azure Blob.
 */
void SNEPPX_azure_blob_destroy(void* client) { free(client); }
/**
 * @brief Load Azure Blob Up.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_upload(void* client, const char* blob_name, const unsigned char* data, size_t len, const char* content_type) { (void)client; (void)blob_name; (void)data; (void)len; (void)content_type; return 0; }
/**
 * @brief Load Azure Blob Down.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_download(void* client, const char* blob_name, unsigned char** data, size_t* len) { (void)client; (void)blob_name; (void)data; (void)len; return 0; }
/**
 * @brief Perform Azure Blob Delete.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_delete(void* client, const char* blob_name) { (void)client; (void)blob_name; return 0; }
/**
 * @brief Perform Azure Blob List.
 *
 * @param client [out] Client value.
 * @param prefix [in] Prefix value.
 * @param blobs [out] Blobs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_list(void* client, const char* prefix, char*** blobs, size_t* count) { (void)client; (void)prefix; (void)blobs; (void)count; return 0; }
/**
 * @brief Perform Azure Blob Exists.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_exists(void* client, const char* blob_name) { (void)client; (void)blob_name; return 0; }
/**
 * @brief Perform Azure Blob Get Url.
 *
 * @param client [out] Client value.
 * @param blob_name [in] Blob Name value.
 * @param url [out] Url value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_azure_blob_get_url(void* client, const char* blob_name, char* url, size_t url_max) { (void)client; (void)blob_name; (void)url; (void)url_max; return 0; }
