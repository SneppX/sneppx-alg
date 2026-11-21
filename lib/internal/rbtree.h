#ifndef SNEPPX_RBTREE_H
#define SNEPPX_RBTREE_H
/*
 * SNEPPX - Rbtree
 *
 * WHAT
 *   Rbtree.
 *
 * CONCEPT
 *   Provides the Rbtree.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Red-Black Tree — v0.5 (generic library)
 *
 * PURPOSE: Balanced binary search tree for memory region tracking
 * (secure allocator), timer scheduling, and ordered maps.
 *
 * Invariants:
 *   1. Every node is red or black.
 *   2. Root is black.
 *   3. Leaves (NULL) are black.
 *   4. Red nodes have black children.
 *   5. Every path from root to leaf has the same black height.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { SNEPPX_RBTREE_RED, SNEPPX_RBTREE_BLACK } SNEPPXRBColor;

typedef struct SNEPPXRBNode {
    struct SNEPPXRBNode* parent;
    struct SNEPPXRBNode* left;
    struct SNEPPXRBNode* right;
    SNEPPXRBColor        color;
    uint64_t           key;
    void*              value;
} SNEPPXRBNode;

typedef struct {
    SNEPPXRBNode* root;
    size_t      size;
    int         (*compare)(uint64_t a, uint64_t b);
} SNEPPXRBTree;

/**
 * @brief Create Rbtree.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRBTree* SNEPPX_rbtree_create(void);
/**
 * @brief Destroy Rbtree.
 *
 * @param tree [out] Tree value.
 */
void        SNEPPX_rbtree_destroy(SNEPPXRBTree* tree);

/**
 * @brief Perform Rbtree Insert.
 *
 * @param tree [out] Tree value.
 * @param key [in] Key value.
 * @param value [out] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_rbtree_insert(SNEPPXRBTree* tree, uint64_t key, void* value);
/**
 * @brief Perform Rbtree Search.
 *
 * @param tree [in] Tree value.
 * @param key [in] Key value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_rbtree_search(const SNEPPXRBTree* tree, uint64_t key);
/**
 * @brief Perform Rbtree Delete.
 *
 * @param tree [out] Tree value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_rbtree_delete(SNEPPXRBTree* tree, uint64_t key);

/**
 * @brief Perform Rbtree Min.
 *
 * @param tree [in] Tree value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_min(const SNEPPXRBTree* tree);
/**
 * @brief Perform Rbtree Max.
 *
 * @param tree [in] Tree value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_max(const SNEPPXRBTree* tree);
/**
 * @brief Perform Rbtree Successor.
 *
 * @param tree [in] Tree value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_successor(const SNEPPXRBTree* tree, uint64_t key);
/**
 * @brief Perform Rbtree Predecessor.
 *
 * @param tree [in] Tree value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_predecessor(const SNEPPXRBTree* tree, uint64_t key);

void SNEPPX_rbtree_foreach(const SNEPPXRBTree* tree, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_RBTREE_H */
