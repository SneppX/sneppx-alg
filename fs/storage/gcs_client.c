#include "gcs_client.h"
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Gcs Client
 *
 * WHAT
 *   Gcs Client.
 *
 * CONCEPT
 *   Provides the Gcs Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Create Gcs.
 *
 * @param project_id [in] Project Id value.
 * @param bucket_name [in] Bucket Name value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_gcs_create(const char* project_id, const char* bucket_name, const char* credentials_json) { (void)project_id; (void)bucket_name; (void)credentials_json; return NULL; }
/**
 * @brief Destroy Gcs.
 */
void SNEPPX_gcs_destroy(void* client) { free(client); }
/**
 * @brief Load Gcs Up.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_upload(void* client, const char* object_path, const unsigned char* data, size_t len, const char* content_type) { (void)client; (void)object_path; (void)data; (void)len; (void)content_type; return 0; }
/**
 * @brief Load Gcs Down.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_download(void* client, const char* object_path, unsigned char** data, size_t* len) { (void)client; (void)object_path; (void)data; (void)len; return 0; }
/**
 * @brief Perform Gcs Delete.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_delete(void* client, const char* object_path) { (void)client; (void)object_path; return 0; }
/**
 * @brief Perform Gcs List.
 *
 * @param client [out] Client value.
 * @param prefix [in] Prefix value.
 * @param objects [out] Objects value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_list(void* client, const char* prefix, char*** objects, size_t* count) { (void)client; (void)prefix; (void)objects; (void)count; return 0; }
/**
 * @brief Perform Gcs Exists.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_exists(void* client, const char* object_path) { (void)client; (void)object_path; return 0; }
