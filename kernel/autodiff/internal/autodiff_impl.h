#ifndef SNEPPX_AUTODIFF_INTERNAL_H
#define SNEPPX_AUTODIFF_INTERNAL_H
/*
 * SNEPPX - Autodiff Impl
 *
 * WHAT
 *   Autodiff Impl.
 *
 * CONCEPT
 *   Provides the Autodiff Impl.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Autodiff Internal Implementation — v0.5
 *
 * PURPOSE: Internal gradient graph representation, tape node structures,
 * and backward-pass scheduler.  The external API (automatic_differentiation_framework.h) exposes
 * a high-level Variable/GradFn interface; this header defines the internal
 * graph topology, reference-counted tensors, and topological sort utilities.
 *
 * DEPENDENCIES: automatic_differentiation_framework.h, multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_GRAD_NODE_MUL,
    SNEPPX_GRAD_NODE_ADD,
    SNEPPX_GRAD_NODE_SUB,
    SNEPPX_GRAD_NODE_MATMUL,
    SNEPPX_GRAD_NODE_CONV2D,
    SNEPPX_GRAD_NODE_RELU,
    SNEPPX_GRAD_NODE_SIGMOID,
    SNEPPX_GRAD_NODE_TANH,
    SNEPPX_GRAD_NODE_SOFTMAX,
    SNEPPX_GRAD_NODE_CROSS_ENTROPY,
    SNEPPX_GRAD_NODE_RESHAPE,
    SNEPPX_GRAD_NODE_CUSTOM,
} SNEPPXGradNodeType;

typedef struct SNEPPXGradNode {
    SNEPPXGradNodeType   type;
    void*              grad_fn;
    struct SNEPPXGradNode** inputs;
    int                num_inputs;
    struct SNEPPXGradNode** consumers;
    int                num_consumers;
    void*              saved_tensors;      /* for backward */
    int                ref_count;
    uint64_t           id;
} SNEPPXGradNode;

typedef struct {
    SNEPPXGradNode**    nodes;
    size_t            num_nodes;
    size_t            capacity;
    size_t            num_leafs;
    SNEPPXGradNode**    topological_order;
    int               needs_rebuild;
} SNEPPXGradGraph;

/**
 * @brief Initialize Grad Graph.
 *
 * @param graph [out] Graph value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grad_graph_init(SNEPPXGradGraph* graph);
/**
 * @brief Destroy Grad Graph.
 *
 * @param graph [out] Graph value.
 */
void SNEPPX_grad_graph_destroy(SNEPPXGradGraph* graph);

/**
 * @brief Create Grad Node.
 *
 * @param type [in] Type value.
 * @param node [out] Node value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grad_node_create(SNEPPXGradNodeType type, SNEPPXGradNode** node);
/**
 * @brief Destroy Grad Node.
 *
 * @param node [out] Node value.
 */
void SNEPPX_grad_node_destroy(SNEPPXGradNode* node);
/**
 * @brief Perform Grad Node Add Input.
 *
 * @param node [out] Node value.
 * @param input [out] Input value.
 */
void SNEPPX_grad_node_add_input(SNEPPXGradNode* node, SNEPPXGradNode* input);
/**
 * @brief Perform Grad Node Add Consumer.
 *
 * @param node [out] Node value.
 * @param consumer [out] Consumer value.
 */
void SNEPPX_grad_node_add_consumer(SNEPPXGradNode* node, SNEPPXGradNode* consumer);

/**
 * @brief Perform Grad Graph Toposort.
 *
 * @param graph [out] Graph value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grad_graph_toposort(SNEPPXGradGraph* graph);
/**
 * @brief Run the backward pass for Grad Graph.
 *
 * @param graph [out] Graph value.
 * @param root [out] Root value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grad_graph_backward(SNEPPXGradGraph* graph, SNEPPXGradNode* root);

/* ---------- Saved tensor helpers ---------- */
/**
 * @brief Perform Grad Save Tensor.
 *
 * @param node [out] Node value.
 * @param tensor [out] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grad_save_tensor(SNEPPXGradNode* node, void* tensor);
/**
 * @brief Perform Grad Restore Tensor.
 *
 * @param node [out] Node value.
 * @param idx [in] Idx value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_grad_restore_tensor(SNEPPXGradNode* node, int idx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_AUTODIFF_INTERNAL_H */
