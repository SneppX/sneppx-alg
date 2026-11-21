#ifndef SNEPPX_PQUEUE_H
#define SNEPPX_PQUEUE_H
/*
 * SNEPPX - Pqueue
 *
 * WHAT
 *   Pqueue.
 *
 * CONCEPT
 *   Provides the Pqueue.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Priority Queue — v0.5 (generic library)
 *
 * PURPOSE: Binary max-heap for task scheduling (thread pool),
 * gradient compression priority, and timer management.
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
    uint64_t priority;
    void*    data;
} SNEPPXPQElement;

typedef struct {
    SNEPPXPQElement* heap;
    size_t         capacity;
    size_t         size;
    int            (*compare)(uint64_t a, uint64_t b);
} SNEPPXPriorityQueue;

/**
 * @brief Create Pq.
 *
 * @param initial_capacity [in] Initial Capacity value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXPriorityQueue* SNEPPX_pq_create(size_t initial_capacity);
/**
 * @brief Destroy Pq.
 *
 * @param pq [out] Pq value.
 */
void               SNEPPX_pq_destroy(SNEPPXPriorityQueue* pq);

/**
 * @brief Perform Pq Push.
 *
 * @param pq [out] Pq value.
 * @param priority [in] Priority value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_pq_push(SNEPPXPriorityQueue* pq, uint64_t priority, void* data);
/**
 * @brief Perform Pq Pop.
 *
 * @param pq [out] Pq value.
 * @param priority [out] Priority value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_pq_pop(SNEPPXPriorityQueue* pq, uint64_t* priority, void** data);
/**
 * @brief Perform Pq Peek.
 *
 * @param pq [in] Pq value.
 * @param priority [out] Priority value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_pq_peek(const SNEPPXPriorityQueue* pq, uint64_t* priority, void** data);
/**
 * @brief Perform Pq Is Empty.
 *
 * @param pq [in] Pq value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_pq_is_empty(const SNEPPXPriorityQueue* pq);
/**
 * @brief Perform Pq Size.
 *
 * @param pq [in] Pq value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_pq_size(const SNEPPXPriorityQueue* pq);

/**
 * @brief Perform Pq Heapify.
 *
 * @param pq [out] Pq value.
 * @param idx [in] Idx value.
 */
void  SNEPPX_pq_heapify(SNEPPXPriorityQueue* pq, size_t idx);
/**
 * @brief Clear Pq.
 *
 * @param pq [out] Pq value.
 */
void  SNEPPX_pq_clear(SNEPPXPriorityQueue* pq);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_PQUEUE_H */
