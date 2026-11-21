#include "rbtree.h"
#include "polymorphic_memory_allocator.h"
#include <stdlib.h>

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


static int default_compare(uint64_t a, uint64_t b) {
    return (a > b) - (a < b);
}

static SNEPPXRBNode* node_create(uint64_t key, void* value) {
    SNEPPXRBNode* node = (SNEPPXRBNode*)SNEPPX_malloc(sizeof(SNEPPXRBNode), 16);
    if (!node) return NULL;
    node->parent = NULL;
    node->left = NULL;
    node->right = NULL;
    node->color = SNEPPX_RBTREE_RED;
    node->key = key;
    node->value = value;
    return node;
}

static void node_free(SNEPPXRBNode* node) {
    if (node) SNEPPX_free(node, sizeof(SNEPPXRBNode));
}

static void rotate_left(SNEPPXRBTree* tree, SNEPPXRBNode* x) {
    SNEPPXRBNode* y = x->right;
    x->right = y->left;
    if (y->left) y->left->parent = x;
    y->parent = x->parent;
    if (!x->parent) tree->root = y;
    else if (x == x->parent->left) x->parent->left = y;
    else x->parent->right = y;
    y->left = x;
    x->parent = y;
}

static void rotate_right(SNEPPXRBTree* tree, SNEPPXRBNode* y) {
    SNEPPXRBNode* x = y->left;
    y->left = x->right;
    if (x->right) x->right->parent = y;
    x->parent = y->parent;
    if (!y->parent) tree->root = x;
    else if (y == y->parent->left) y->parent->left = x;
    else y->parent->right = x;
    x->right = y;
    y->parent = x;
}

static void insert_fixup(SNEPPXRBTree* tree, SNEPPXRBNode* z) {
    while (z->parent && z->parent->color == SNEPPX_RBTREE_RED) {
        if (z->parent == z->parent->parent->left) {
            SNEPPXRBNode* y = z->parent->parent->right;
            if (y && y->color == SNEPPX_RBTREE_RED) {
                z->parent->color = SNEPPX_RBTREE_BLACK;
                y->color = SNEPPX_RBTREE_BLACK;
                z->parent->parent->color = SNEPPX_RBTREE_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    rotate_left(tree, z);
                }
                z->parent->color = SNEPPX_RBTREE_BLACK;
                z->parent->parent->color = SNEPPX_RBTREE_RED;
                rotate_right(tree, z->parent->parent);
            }
        } else {
            SNEPPXRBNode* y = z->parent->parent->left;
            if (y && y->color == SNEPPX_RBTREE_RED) {
                z->parent->color = SNEPPX_RBTREE_BLACK;
                y->color = SNEPPX_RBTREE_BLACK;
                z->parent->parent->color = SNEPPX_RBTREE_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rotate_right(tree, z);
                }
                z->parent->color = SNEPPX_RBTREE_BLACK;
                z->parent->parent->color = SNEPPX_RBTREE_RED;
                rotate_left(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = SNEPPX_RBTREE_BLACK;
}

/**
 * @brief Create Rbtree.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRBTree* SNEPPX_rbtree_create(void) {
    SNEPPXRBTree* tree = (SNEPPXRBTree*)SNEPPX_malloc(sizeof(SNEPPXRBTree), 16);
    if (!tree) return NULL;
    tree->root = NULL;
    tree->size = 0;
    tree->compare = default_compare;
    return tree;
}

/**
 * @brief Destroy Rbtree.
 */
void SNEPPX_rbtree_destroy(SNEPPXRBTree* tree) {
    if (!tree) return;
    SNEPPXRBNode* stack[1024];
    size_t top = 0;
    SNEPPXRBNode* node = tree->root;
    
    while (node || top > 0) {
        while (node) {
            stack[top++] = node;
            node = node->left;
        }
        node = stack[--top];
        SNEPPXRBNode* right = node->right;
        node_free(node);
        node = right;
    }
    SNEPPX_free(tree, sizeof(SNEPPXRBTree));
}

/**
 * @brief Perform Rbtree Insert.
 *
 * @param tree [out] Tree value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rbtree_insert(SNEPPXRBTree* tree, uint64_t key, void* value) {
    if (!tree) return -1;
    if (!tree->compare) tree->compare = default_compare;
    
    SNEPPXRBNode* y = NULL;
    SNEPPXRBNode* x = tree->root;
    
    while (x) {
        y = x;
        int cmp = tree->compare(key, x->key);
        if (cmp == 0) {
            x->value = value;
            return 0;
        }
        x = cmp < 0 ? x->left : x->right;
    }
    
    SNEPPXRBNode* z = node_create(key, value);
    if (!z) return -1;
    z->parent = y;
    
    if (!y) tree->root = z;
    else if (tree->compare(key, y->key) < 0) y->left = z;
    else y->right = z;
    
    tree->size++;
    insert_fixup(tree, z);
    return 0;
}

static SNEPPXRBNode* tree_search(SNEPPXRBNode* root, uint64_t key, int (*compare)(uint64_t, uint64_t)) {
    while (root) {
        int cmp = compare(key, root->key);
        if (cmp == 0) return root;
        root = cmp < 0 ? root->left : root->right;
    }
    return NULL;
}

/**
 * @brief Perform Rbtree Search.
 *
 * @param tree [in] Tree value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_rbtree_search(const SNEPPXRBTree* tree, uint64_t key) {
    if (!tree) return NULL;
    if (!tree->compare) return NULL;
    SNEPPXRBNode* node = tree_search(tree->root, key, tree->compare);
    return node ? node->value : NULL;
}

static SNEPPXRBNode* tree_minimum(SNEPPXRBNode* node) {
    while (node->left) node = node->left;
    return node;
}

static void transplant(SNEPPXRBTree* tree, SNEPPXRBNode* u, SNEPPXRBNode* v) {
    if (!u->parent) tree->root = v;
    else if (u == u->parent->left) u->parent->left = v;
    else u->parent->right = v;
    if (v) v->parent = u->parent;
}

static void delete_fixup(SNEPPXRBTree* tree, SNEPPXRBNode* x) {
    while (x != tree->root && (!x || x->color == SNEPPX_RBTREE_BLACK)) {
        if (x == x->parent->left) {
            SNEPPXRBNode* w = x->parent->right;
            if (w && w->color == SNEPPX_RBTREE_RED) {
                w->color = SNEPPX_RBTREE_BLACK;
                x->parent->color = SNEPPX_RBTREE_RED;
                rotate_left(tree, x->parent);
                w = x->parent->right;
            }
            if (w && (!w->left || w->left->color == SNEPPX_RBTREE_BLACK) &&
                (!w->right || w->right->color == SNEPPX_RBTREE_BLACK)) {
                w->color = SNEPPX_RBTREE_RED;
                x = x->parent;
            } else {
                if (!w->right || w->right->color == SNEPPX_RBTREE_BLACK) {
                    if (w->left) w->left->color = SNEPPX_RBTREE_BLACK;
                    w->color = SNEPPX_RBTREE_RED;
                    rotate_right(tree, w);
                    w = x->parent->right;
                }
                w->color = x->parent->color;
                x->parent->color = SNEPPX_RBTREE_BLACK;
                if (w->right) w->right->color = SNEPPX_RBTREE_BLACK;
                rotate_left(tree, x->parent);
                x = tree->root;
            }
        } else {
            SNEPPXRBNode* w = x->parent->left;
            if (w && w->color == SNEPPX_RBTREE_RED) {
                w->color = SNEPPX_RBTREE_BLACK;
                x->parent->color = SNEPPX_RBTREE_RED;
                rotate_right(tree, x->parent);
                w = x->parent->left;
            }
            if (w && (!w->right || w->right->color == SNEPPX_RBTREE_BLACK) &&
                (!w->left || w->left->color == SNEPPX_RBTREE_BLACK)) {
                w->color = SNEPPX_RBTREE_RED;
                x = x->parent;
            } else {
                if (!w->left || w->left->color == SNEPPX_RBTREE_BLACK) {
                    if (w->right) w->right->color = SNEPPX_RBTREE_BLACK;
                    w->color = SNEPPX_RBTREE_RED;
                    rotate_left(tree, w);
                    w = x->parent->left;
                }
                w->color = x->parent->color;
                x->parent->color = SNEPPX_RBTREE_BLACK;
                if (w->left) w->left->color = SNEPPX_RBTREE_BLACK;
                rotate_right(tree, x->parent);
                x = tree->root;
            }
        }
    }
    if (x) x->color = SNEPPX_RBTREE_BLACK;
}

/**
 * @brief Perform Rbtree Delete.
 *
 * @param tree [out] Tree value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rbtree_delete(SNEPPXRBTree* tree, uint64_t key) {
    if (!tree || !tree->root) return -1;
    
    SNEPPXRBNode* z = tree_search(tree->root, key, tree->compare ? tree->compare : default_compare);
    if (!z) return -1;
    
    SNEPPXRBNode* y = z;
    SNEPPXRBColor y_orig_color = y->color;
    SNEPPXRBNode* x;
    
    if (!z->left) {
        x = z->right;
        transplant(tree, z, z->right);
    } else if (!z->right) {
        x = z->left;
        transplant(tree, z, z->left);
    } else {
        y = tree_minimum(z->right);
        y_orig_color = y->color;
        x = y->right;
        if (y->parent == z) {
            if (x) x->parent = y;
        } else {
            transplant(tree, y, y->right);
            y->right = z->right;
            if (y->right) y->right->parent = y;
        }
        transplant(tree, z, y);
        y->left = z->left;
        y->left->parent = y;
        y->color = z->color;
    }
    
    if (y_orig_color == SNEPPX_RBTREE_BLACK) delete_fixup(tree, x);
    node_free(z);
    tree->size--;
    return 0;
}

/**
 * @brief Perform Rbtree Min.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_min(const SNEPPXRBTree* tree) {
    if (!tree || !tree->root) return 0;
    SNEPPXRBNode* node = tree->root;
    while (node->left) node = node->left;
    return node->key;
}

/**
 * @brief Perform Rbtree Max.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_max(const SNEPPXRBTree* tree) {
    if (!tree || !tree->root) return 0;
    SNEPPXRBNode* node = tree->root;
    while (node->right) node = node->right;
    return node->key;
}

/**
 * @brief Perform Rbtree Successor.
 *
 * @param tree [in] Tree value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_successor(const SNEPPXRBTree* tree, uint64_t key) {
    if (!tree || !tree->root) return 0;
    SNEPPXRBNode* node = tree_search(tree->root, key, tree->compare ? tree->compare : default_compare);
    if (!node) return 0;
    if (node->right) return tree_minimum(node->right)->key;
    
    SNEPPXRBNode* p = node->parent;
    while (p && node == p->right) {
        node = p;
        p = p->parent;
    }
    return p ? p->key : 0;
}

/**
 * @brief Perform Rbtree Predecessor.
 *
 * @param tree [in] Tree value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_rbtree_predecessor(const SNEPPXRBTree* tree, uint64_t key) {
    if (!tree || !tree->root) return 0;
    SNEPPXRBNode* node = tree_search(tree->root, key, tree->compare ? tree->compare : default_compare);
    if (!node) return 0;
    if (node->left) {
        SNEPPXRBNode* n = node->left;
        while (n->right) n = n->right;
        return n->key;
    }
    SNEPPXRBNode* p = node->parent;
    while (p && node == p->left) {
        node = p;
        p = p->parent;
    }
    return p ? p->key : 0;
}

static void inorder_collect(const SNEPPXRBNode* node, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx) {
    if (!node) return;
    inorder_collect(node->left, fn, ctx);
    fn(node->key, node->value, ctx);
    inorder_collect(node->right, fn, ctx);
}

void SNEPPX_rbtree_foreach(const SNEPPXRBTree* tree, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx) {
    if (tree && fn) inorder_collect(tree->root, fn, ctx);
}
