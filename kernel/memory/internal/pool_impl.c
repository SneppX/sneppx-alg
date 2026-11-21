#include "pool_impl.h"
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_InterlockedCompareExchangePointer)
#else
#include <stdatomic.h>
#endif

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


typedef struct TreiberNode { struct TreiberNode* next; } TreiberNode;
typedef struct { TreiberNode* top; } TreiberStack;

#ifdef _MSC_VER
#define CAS_PTR(dest, comp, exch) _InterlockedCompareExchangePointer((void*volatile*)(dest), (exch), (comp))
#else
#define CAS_PTR(dest, comp, exch) __sync_val_compare_and_swap((void*volatile*)(dest), (comp), (exch))
#endif

/**
 * @brief Create Mem Chunk.
 *
 * @param chunk [out] Chunk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_chunk_create(SNEPPXMemChunk** chunk, size_t min_size) {
    if (!chunk) return -1;
    size_t sz = min_size < 65536 ? 65536 : (min_size + 4095) & ~(size_t)4095;
    *chunk = (SNEPPXMemChunk*)calloc(1, sizeof(SNEPPXMemChunk));
    if (!*chunk) return -1;
    (*chunk)->mem = calloc(1, sz);
    if (!(*chunk)->mem) { free(*chunk); *chunk = NULL; return -1; }
    (*chunk)->size = sz;
    (*chunk)->size_class_idx = 0;
    (*chunk)->slab_count = 0;
    return 0;
}

/**
 * @brief Destroy Mem Chunk.
 */
void SNEPPX_mem_chunk_destroy(SNEPPXMemChunk* chunk) {
    if (chunk) { free(chunk->mem); free(chunk); }
}

/**
 * @brief Perform Mem Chunk Carve.
 *
 * @param chunk [out] Chunk value.
 * @param block_size [in] Block Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mem_chunk_carve(SNEPPXMemChunk* chunk, size_t block_size, size_t alignment) {
    if (!chunk || !chunk->mem || block_size == 0) return NULL;
    if (alignment == 0) alignment = 16;
    if (alignment & (alignment - 1)) alignment = 16;
    uintptr_t base = (uintptr_t)chunk->mem;
    size_t used = 0;
    for (int i = 0; i < chunk->slab_count; i++) {
        size_t slab_sz = chunk->size_class_idx > 0 ? (1 << (chunk->size_class_idx + 4)) : 64;
        used += slab_sz;
    }
    uintptr_t candidate = base + used;
    uintptr_t aligned = (candidate + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t needed = (aligned - candidate) + block_size;
    if (used + needed > chunk->size) return NULL;
    chunk->slab_count++;
    return (void*)aligned;
}

/**
 * @brief Perform Mem Chunk Has Space.
 *
 * @param chunk [in] Chunk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mem_chunk_has_space(const SNEPPXMemChunk* chunk, size_t block_size) {
    if (!chunk) return 0;
    size_t used = 0;
    size_t slab_sz = chunk->size_class_idx > 0 ? (1 << (chunk->size_class_idx + 4)) : 64;
    used = (size_t)chunk->slab_count * slab_sz;
    return (used + slab_sz <= chunk->size) ? 1 : 0;
}

/**
 * @brief Perform Lockfree Stack Push.
 *
 * @param stack_ptr [out] Stack Ptr value.
 */
void SNEPPX_lockfree_stack_push(void* stack_ptr, void* node_ptr) {
    if (!stack_ptr || !node_ptr) return;
    TreiberStack* stack = (TreiberStack*)stack_ptr;
    TreiberNode* node = (TreiberNode*)node_ptr;
    TreiberNode* old_top;
    do {
        old_top = stack->top;
        node->next = old_top;
    } while (CAS_PTR(&stack->top, old_top, node) != old_top);
}

/**
 * @brief Perform Lockfree Stack Pop.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_lockfree_stack_pop(void* stack_ptr) {
    if (!stack_ptr) return NULL;
    TreiberStack* stack = (TreiberStack*)stack_ptr;
    TreiberNode* old_top;
    TreiberNode* new_top;
    do {
        old_top = stack->top;
        if (!old_top) return NULL;
        new_top = old_top->next;
    } while (CAS_PTR(&stack->top, old_top, new_top) != old_top);
    return old_top;
}

/**
 * @brief Perform Lockfree Stack Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lockfree_stack_count(const void* stack_ptr) {
    if (!stack_ptr) return 0;
    const TreiberStack* stack = (const TreiberStack*)stack_ptr;
    int cnt = 0;
    TreiberNode* cur = stack->top;
    while (cur) { cnt++; cur = cur->next; }
    return cnt;
}

#ifdef _MSC_VER
static __declspec(thread) void* tls_cache = NULL;
#else
static __thread void* tls_cache = NULL;
#endif

/**
 * @brief Initialize Mem Tls.
 */
void SNEPPX_mem_tls_init(void) { tls_cache = NULL; }
/**
 * @brief Perform Mem Tls Cleanup.
 */
void SNEPPX_mem_tls_cleanup(void) { tls_cache = NULL; }
/**
 * @brief Get Mem Tls.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mem_tls_get(void) { return tls_cache; }
/**
 * @brief Set Mem Tls.
 */
void SNEPPX_mem_tls_set(void* cache) { tls_cache = cache; }
/**
 * @brief Perform Mem Tls Flush.
 */
void SNEPPX_mem_tls_flush(void) { tls_cache = NULL; }
