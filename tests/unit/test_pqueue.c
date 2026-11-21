#include "pqueue.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Pqueue
 *
 * WHAT
 *   Test Pqueue.
 *
 * CONCEPT
 *   Provides the Test Pqueue.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_pqueue_create_destroy(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    SX_ASSERT(pq != NULL, "pqueue created");
    SX_ASSERT(SNEPPX_pq_size(pq) == 0, "empty pqueue");
    SNEPPX_pq_destroy(pq);
}

static void test_pqueue_push_pop(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    SNEPPX_pq_push(pq, 3, (void*)3);
    SNEPPX_pq_push(pq, 1, (void*)1);
    SNEPPX_pq_push(pq, 2, (void*)2);
    SX_ASSERT(SNEPPX_pq_size(pq) == 3, "three items");

    uint64_t priority;
    void* val;
    int ret = SNEPPX_pq_pop(pq, &priority, &val);
    SX_ASSERT(ret == 0, "pop returns 0");
    SX_ASSERT(priority == 1 && val == (void*)1, "pop lowest priority");
    ret = SNEPPX_pq_pop(pq, &priority, &val);
    SX_ASSERT(ret == 0 && priority == 2 && val == (void*)2, "pop second lowest");
    ret = SNEPPX_pq_pop(pq, &priority, &val);
    SX_ASSERT(ret == 0 && priority == 3 && val == (void*)3, "pop highest priority");
    SX_ASSERT(SNEPPX_pq_size(pq) == 0, "empty after pops");
    SNEPPX_pq_destroy(pq);
}

static void test_pqueue_peek(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    SNEPPX_pq_push(pq, 5, (void*)42);
    uint64_t priority;
    void* val;
    int ret = SNEPPX_pq_peek(pq, &priority, &val);
    SX_ASSERT(ret == 0, "peek returns 0");
    SX_ASSERT(priority == 5 && val == (void*)42, "peek returns top");
    SX_ASSERT(SNEPPX_pq_size(pq) == 1, "size unchanged after peek");
    SNEPPX_pq_destroy(pq);
}


TEST(test_pqueue, pqueue_create_destroy) { test_pqueue_create_destroy(); }
TEST(test_pqueue, pqueue_push_pop) { test_pqueue_push_pop(); }
TEST(test_pqueue, pqueue_peek) { test_pqueue_peek(); }
