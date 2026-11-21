#ifndef SNEPPX_OBJECT_POOL_H
#define SNEPPX_OBJECT_POOL_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Object Pool
 *
 * WHAT
 *   Object Pool.
 *
 * CONCEPT
 *   Provides the Object Pool.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Objpool.
 *
 * @param object_size [in] Object Size value.
 * @param capacity [in] Capacity value.
 * @param thread_safe [in] Thread Safe value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_objpool_create(size_t object_size, size_t capacity, int thread_safe);
/**
 * @brief Destroy Objpool.
 *
 * @param pool [out] Pool value.
 */
void SNEPPX_objpool_destroy(void* pool);
/**
 * @brief Perform Objpool Acquire.
 *
 * @param pool [out] Pool value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_objpool_acquire(void* pool);
/**
 * @brief Perform Objpool Release.
 *
 * @param pool [out] Pool value.
 * @param obj [out] Obj value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_objpool_release(void* pool, void* obj);
/**
 * @brief Perform Objpool Available.
 *
 * @param pool [out] Pool value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_objpool_available(void* pool);
/**
 * @brief Perform Objpool Capacity.
 *
 * @param pool [out] Pool value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_objpool_capacity(void* pool);
/**
 * @brief Perform Objpool Prealloc.
 *
 * @param pool [out] Pool value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_objpool_prealloc(void* pool, size_t count);
/**
 * @brief Clear Objpool.
 *
 * @param pool [out] Pool value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_objpool_clear(void* pool);
#ifdef __cplusplus
}
#endif
#endif
