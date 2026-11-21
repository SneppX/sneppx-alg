#include "rbtree.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Rbtree
 *
 * WHAT
 *   Test Rbtree.
 *
 * CONCEPT
 *   Provides the Test Rbtree.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_rbtree_create_destroy(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SX_ASSERT(tree != NULL, "rbtree created");
    SX_ASSERT(tree->size == 0, "empty tree");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_insert_search(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 5, (void*)50);
    SNEPPX_rbtree_insert(tree, 15, (void*)150);
    SX_ASSERT(tree->size == 3, "three nodes");

    void* val = SNEPPX_rbtree_search(tree, 10);
    SX_ASSERT(val == (void*)100, "search key 10");
    val = SNEPPX_rbtree_search(tree, 5);
    SX_ASSERT(val == (void*)50, "search key 5");
    val = SNEPPX_rbtree_search(tree, 99);
    SX_ASSERT(val == NULL, "search missing key");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_delete(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    SNEPPX_rbtree_delete(tree, 10);
    SX_ASSERT(tree->size == 2, "size 2 after delete");
    void* val = SNEPPX_rbtree_search(tree, 10);
    SX_ASSERT(val == NULL, "deleted key absent");
    SNEPPX_rbtree_destroy(tree);
}

static void foreach_collect(void* ctx) {
    // We can't use this easily with the current API, skip inorder test
    (void)ctx;
}

static void test_rbtree_min_max(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    
    uint64_t min = SNEPPX_rbtree_min(tree);
    SX_ASSERT(min == 10, "min is 10");
    
    uint64_t max = SNEPPX_rbtree_max(tree);
    SX_ASSERT(max == 30, "max is 30");
    
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_successor_predecessor(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    SNEPPX_rbtree_insert(tree, 15, (void*)150);
    
    uint64_t succ = SNEPPX_rbtree_successor(tree, 15);
    SX_ASSERT(succ == 20, "successor of 15 is 20");
    
    uint64_t pred = SNEPPX_rbtree_predecessor(tree, 15);
    SX_ASSERT(pred == 10, "predecessor of 15 is 10");
    
    SNEPPX_rbtree_destroy(tree);
}


TEST(test_rbtree, rbtree_create_destroy) { test_rbtree_create_destroy(); }
TEST(test_rbtree, rbtree_insert_search) { test_rbtree_insert_search(); }
TEST(test_rbtree, rbtree_delete) { test_rbtree_delete(); }
TEST(test_rbtree, rbtree_min_max) { test_rbtree_min_max(); }
TEST(test_rbtree, rbtree_successor_predecessor) { test_rbtree_successor_predecessor(); }
