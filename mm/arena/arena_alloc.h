#ifndef SNEPPX_ARENA_ALLOC_H
#define SNEPPX_ARENA_ALLOC_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Arena Alloc
 *
 * WHAT
 *   Arena Alloc.
 *
 * CONCEPT
 *   Provides the Arena Alloc.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Arena.
 *
 * @param block_size [in] Block Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_create(size_t block_size, size_t alignment);
/**
 * @brief Destroy Arena.
 *
 * @param arena [out] Arena value.
 */
void SNEPPX_arena_destroy(void* arena);
/**
 * @brief Perform Arena Alloc.
 *
 * @param arena [out] Arena value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_alloc(void* arena, size_t size);
/**
 * @brief Perform Arena Alloc Aligned.
 *
 * @param arena [out] Arena value.
 * @param size [in] Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_alloc_aligned(void* arena, size_t size, size_t alignment);
/**
 * @brief Reset Arena.
 *
 * @param arena [out] Arena value.
 */
void SNEPPX_arena_reset(void* arena);
/**
 * @brief Perform Arena Used.
 *
 * @param arena [out] Arena value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arena_used(void* arena);
/**
 * @brief Perform Arena Capacity.
 *
 * @param arena [out] Arena value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arena_capacity(void* arena);
/**
 * @brief Perform Arena Stats.
 *
 * @param arena [out] Arena value.
 * @param total_allocated [out] Total Allocated value.
 * @param total_used [out] Total Used value.
 * @param wasted [out] Wasted value.
 */
void SNEPPX_arena_stats(void* arena, size_t* total_allocated, size_t* total_used, size_t* wasted);
#ifdef __cplusplus
}
#endif
#endif
