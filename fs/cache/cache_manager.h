#ifndef SNEPPX_CACHE_MANAGER_H
#define SNEPPX_CACHE_MANAGER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Cache Manager
 *
 * WHAT
 *   Cache Manager.
 *
 * CONCEPT
 *   Provides cache timing protection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Cache.
 *
 * @param max_size_bytes [in] Max Size Bytes value.
 * @param ttl_secs [in] Ttl Secs value.
 * @param policy [in] Policy value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_cache_create(size_t max_size_bytes, int ttl_secs, const char* policy);
/**
 * @brief Destroy Cache.
 *
 * @param cache [out] Cache value.
 */
void SNEPPX_cache_destroy(void* cache);
/**
 * @brief Perform Cache Put.
 *
 * @param cache [out] Cache value.
 * @param key [in] Key value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cache_put(void* cache, const char* key, const unsigned char* data, size_t len);
/**
 * @brief Get Cache.
 *
 * @param cache [out] Cache value.
 * @param key [in] Key value.
 * @param data [out] Data value.
 * @param len [out] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cache_get(void* cache, const char* key, unsigned char** data, size_t* len);
/**
 * @brief Remove Cache.
 *
 * @param cache [out] Cache value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cache_remove(void* cache, const char* key);
/**
 * @brief Clear Cache.
 *
 * @param cache [out] Cache value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cache_clear(void* cache);
/**
 * @brief Perform Cache Contains.
 *
 * @param cache [out] Cache value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cache_contains(void* cache, const char* key);
/**
 * @brief Perform Cache Size.
 *
 * @param cache [out] Cache value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_cache_size(void* cache);
/**
 * @brief Perform Cache Count.
 *
 * @param cache [out] Cache value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_cache_count(void* cache);
/**
 * @brief Perform Cache Hit Rate.
 *
 * @param cache [out] Cache value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_cache_hit_rate(void* cache);
#ifdef __cplusplus
}
#endif
#endif
