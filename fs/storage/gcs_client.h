#ifndef SNEPPX_GCS_CLIENT_H
#define SNEPPX_GCS_CLIENT_H
#include <stddef.h>
#ifdef __cplusplus
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


extern "C" {
#endif
/**
 * @brief Create Gcs.
 *
 * @param project_id [in] Project Id value.
 * @param bucket_name [in] Bucket Name value.
 * @param credentials_json [in] Credentials Json value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_gcs_create(const char* project_id, const char* bucket_name, const char* credentials_json);
/**
 * @brief Destroy Gcs.
 *
 * @param client [out] Client value.
 */
void SNEPPX_gcs_destroy(void* client);
/**
 * @brief Load Gcs Up.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param content_type [in] Content Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_upload(void* client, const char* object_path, const unsigned char* data, size_t len, const char* content_type);
/**
 * @brief Load Gcs Down.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_download(void* client, const char* object_path, unsigned char** data, size_t* len);
/**
 * @brief Perform Gcs Delete.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_delete(void* client, const char* object_path);
/**
 * @brief Perform Gcs List.
 *
 * @param client [out] Client value.
 * @param prefix [in] Prefix value.
 * @param objects [out] Objects value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_list(void* client, const char* prefix, char*** objects, size_t* count);
/**
 * @brief Perform Gcs Exists.
 *
 * @param client [out] Client value.
 * @param object_path [in] Object Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcs_exists(void* client, const char* object_path);
#ifdef __cplusplus
}
#endif
#endif
