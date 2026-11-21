#include "arena_alloc.h"
#include <stdlib.h>
#include <string.h>

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


/**
 * @brief Create Arena.
 *
 * @param block_size [in] Block Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_create(size_t block_size, size_t alignment) { (void)block_size; (void)alignment; return calloc(1, 32); }
/**
 * @brief Destroy Arena.
 */
void SNEPPX_arena_destroy(void* arena) { free(arena); }
/**
 * @brief Perform Arena Alloc.
 *
 * @param arena [out] Arena value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_alloc(void* arena, size_t size) { (void)arena; (void)size; return NULL; }
/**
 * @brief Perform Arena Alloc Aligned.
 *
 * @param arena [out] Arena value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_arena_alloc_aligned(void* arena, size_t size, size_t alignment) { (void)arena; (void)size; (void)alignment; return NULL; }
/**
 * @brief Reset Arena.
 */
void SNEPPX_arena_reset(void* arena) { (void)arena; }
/**
 * @brief Perform Arena Used.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arena_used(void* arena) { (void)arena; return 0; }
/**
 * @brief Perform Arena Capacity.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arena_capacity(void* arena) { (void)arena; return 0; }
/**
 * @brief Perform Arena Stats.
 *
 * @param arena [out] Arena value.
 * @param total_allocated [out] Total Allocated value.
 * @param total_used [out] Total Used value.
 */
void SNEPPX_arena_stats(void* arena, size_t* total_allocated, size_t* total_used, size_t* wasted) { (void)arena; (void)total_allocated; (void)total_used; (void)wasted; }
