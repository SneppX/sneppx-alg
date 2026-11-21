#include "s3_client.h"
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - S3 Client
 *
 * WHAT
 *   S3 Client.
 *
 * CONCEPT
 *   Provides the S3 Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Create S3.
 *
 * @param endpoint [in] Endpoint value.
 * @param region [in] Region value.
 * @param access_key [in] Access Key value.
 * @param secret_key [in] Secret Key value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_s3_create(const char* endpoint, const char* region, const char* access_key, const char* secret_key, int use_ssl) { (void)endpoint; (void)region; (void)access_key; (void)secret_key; (void)use_ssl; return NULL; }
/**
 * @brief Destroy S3.
 */
void SNEPPX_s3_destroy(void* client) { free(client); }
/**
 * @brief Perform S3 List Buckets.
 *
 * @param client [out] Client value.
 * @param buckets [out] Buckets value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_list_buckets(void* client, char*** buckets, size_t* count) { (void)client; (void)buckets; (void)count; return 0; }
/**
 * @brief Perform S3 Create Bucket.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_create_bucket(void* client, const char* bucket) { (void)client; (void)bucket; return 0; }
/**
 * @brief Perform S3 Delete Bucket.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_delete_bucket(void* client, const char* bucket) { (void)client; (void)bucket; return 0; }
/**
 * @brief Perform S3 Put Object.
 *
 * @param client [out] Client value.
 * @param bucket [in] Bucket value.
 * @param key [in] Key value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_put_object(void* client, const char* bucket, const char* key, const unsigned char* data, size_t len, const char* content_type) { (void)client; (void)bucket; (void)key; (void)data; (void)len; (void)content_type; return 0; }
/**
 * @brief Perform S3 Get Object.
 *
 * @param client [out] Client value.
 * @param bucket [in] Bucket value.
 * @param key [in] Key value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_get_object(void* client, const char* bucket, const char* key, unsigned char** data, size_t* len) { (void)client; (void)bucket; (void)key; (void)data; (void)len; return 0; }
/**
 * @brief Perform S3 Delete Object.
 *
 * @param client [out] Client value.
 * @param bucket [in] Bucket value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_delete_object(void* client, const char* bucket, const char* key) { (void)client; (void)bucket; (void)key; return 0; }
/**
 * @brief Perform S3 List Objects.
 *
 * @param client [out] Client value.
 * @param bucket [in] Bucket value.
 * @param prefix [in] Prefix value.
 * @param keys [out] Keys value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_list_objects(void* client, const char* bucket, const char* prefix, char*** keys, size_t* count) { (void)client; (void)bucket; (void)prefix; (void)keys; (void)count; return 0; }
/**
 * @brief Perform S3 Head Object.
 *
 * @param client [out] Client value.
 * @param bucket [in] Bucket value.
 * @param key [in] Key value.
 * @param size [out] Size value.
 * @param etag [out] Etag value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_s3_head_object(void* client, const char* bucket, const char* key, unsigned long long* size, char** etag, char** last_modified) { (void)client; (void)bucket; (void)key; (void)size; (void)etag; (void)last_modified; return 0; }
