#ifndef SNEPPX_MEMORY_INTERNAL_H
#define SNEPPX_MEMORY_INTERNAL_H
/*
 * SNEPPX - Pool Impl
 *
 * WHAT
 *   Pool Impl.
 *
 * CONCEPT
 *   Provides the Pool Impl.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Memory Allocator Internal — v0.5
 *
 * PURPOSE: Internal pool management, chunk allocation, and TLS cache
 * operations for the SNEPPX_memory pool allocator.  Implements the
 * Treiber-stack free lists, chunk carving, and per-thread caching.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*  mem;             /* mmap'd chunk */
    size_t size;
    int    size_class_idx;
    int    slab_count;
} SNEPPXMemChunk;

/**
 * @brief Create Mem Chunk.
 *
 * @param chunk [out] Chunk value.
 * @param min_size [in] Min Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_mem_chunk_create(SNEPPXMemChunk** chunk, size_t min_size);
/**
 * @brief Destroy Mem Chunk.
 *
 * @param chunk [out] Chunk value.
 */
void SNEPPX_mem_chunk_destroy(SNEPPXMemChunk* chunk);

/**
 * @brief Perform Mem Chunk Carve.
 *
 * @param chunk [out] Chunk value.
 * @param block_size [in] Block Size value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mem_chunk_carve(SNEPPXMemChunk* chunk, size_t block_size, size_t alignment);
/**
 * @brief Perform Mem Chunk Has Space.
 *
 * @param chunk [in] Chunk value.
 * @param block_size [in] Block Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_mem_chunk_has_space(const SNEPPXMemChunk* chunk, size_t block_size);

/* ---------- Treiber stack operations ---------- */
/**
 * @brief Perform Lockfree Stack Push.
 *
 * @param stack_ptr [out] Stack Ptr value.
 * @param node_ptr [out] Node Ptr value.
 */
void SNEPPX_lockfree_stack_push(void* stack_ptr, void* node_ptr);
/**
 * @brief Perform Lockfree Stack Pop.
 *
 * @param stack_ptr [out] Stack Ptr value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_lockfree_stack_pop(void* stack_ptr);
/**
 * @brief Perform Lockfree Stack Count.
 *
 * @param stack_ptr [in] Stack Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_lockfree_stack_count(const void* stack_ptr);

/* ---------- TLS cache ---------- */
/**
 * @brief Initialize Mem Tls.
 */
void SNEPPX_mem_tls_init(void);
/**
 * @brief Perform Mem Tls Cleanup.
 */
void SNEPPX_mem_tls_cleanup(void);
/**
 * @brief Get Mem Tls.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mem_tls_get(void);
/**
 * @brief Set Mem Tls.
 *
 * @param cache [out] Cache value.
 */
void  SNEPPX_mem_tls_set(void* cache);
/**
 * @brief Perform Mem Tls Flush.
 */
void  SNEPPX_mem_tls_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_MEMORY_INTERNAL_H */
