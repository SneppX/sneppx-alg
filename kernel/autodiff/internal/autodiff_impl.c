#include "autodiff_impl.h"
#include <stdlib.h>
#include <string.h>

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


/**
 * @brief Initialize Grad Graph.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_graph_init(SNEPPXGradGraph* graph) {
    if (!graph) return -1;
    memset(graph, 0, sizeof(*graph));
    graph->capacity = 64;
    graph->nodes = (SNEPPXGradNode**)calloc(graph->capacity, sizeof(SNEPPXGradNode*));
    if (!graph->nodes) return -1;
    graph->topological_order = NULL;
    graph->needs_rebuild = 1;
    return 0;
}

/**
 * @brief Destroy Grad Graph.
 */
void SNEPPX_grad_graph_destroy(SNEPPXGradGraph* graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->num_nodes; i++)
        SNEPPX_grad_node_destroy(graph->nodes[i]);
    free(graph->nodes);
    free(graph->topological_order);
    memset(graph, 0, sizeof(*graph));
}

static uint64_t g_next_id = 1;

/**
 * @brief Create Grad Node.
 *
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_node_create(SNEPPXGradNodeType type, SNEPPXGradNode** node) {
    if (!node) return -1;
    *node = (SNEPPXGradNode*)calloc(1, sizeof(SNEPPXGradNode));
    if (!*node) return -1;
    (*node)->type = type;
    (*node)->ref_count = 1;
    (*node)->id = g_next_id++;
    return 0;
}

/**
 * @brief Destroy Grad Node.
 */
void SNEPPX_grad_node_destroy(SNEPPXGradNode* node) {
    if (!node) return;
    free(node->inputs);
    free(node->consumers);
    free(node->saved_tensors);
    free(node);
}

/**
 * @brief Perform Grad Node Add Input.
 *
 * @param node [out] Node value.
 */
void SNEPPX_grad_node_add_input(SNEPPXGradNode* node, SNEPPXGradNode* input) {
    if (!node || !input) return;
    int n = node->num_inputs;
    SNEPPXGradNode** new_inputs = (SNEPPXGradNode**)realloc(node->inputs, (n + 1) * sizeof(SNEPPXGradNode*));
    if (!new_inputs) return;
    node->inputs = new_inputs;
    node->inputs[n] = input;
    node->num_inputs = n + 1;
    if (input) input->ref_count++;
}

/**
 * @brief Perform Grad Node Add Consumer.
 *
 * @param node [out] Node value.
 */
void SNEPPX_grad_node_add_consumer(SNEPPXGradNode* node, SNEPPXGradNode* consumer) {
    if (!node || !consumer) return;
    int n = node->num_consumers;
    SNEPPXGradNode** new_consumers = (SNEPPXGradNode**)realloc(node->consumers, (n + 1) * sizeof(SNEPPXGradNode*));
    if (!new_consumers) return;
    node->consumers = new_consumers;
    node->consumers[n] = consumer;
    node->num_consumers = n + 1;
}

static void toposort_dfs(SNEPPXGradNode* node, SNEPPXGradNode*** order, size_t* count, size_t* capacity, int* visited) {
    if (!node || visited[node->id % 65536]) return;
    visited[node->id % 65536] = 1;
    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs[i])
            toposort_dfs(node->inputs[i], order, count, capacity, visited);
    }
    if (*count >= *capacity) {
        *capacity *= 2;
        *order = (SNEPPXGradNode**)realloc(*order, *capacity * sizeof(SNEPPXGradNode*));
    }
    if (*order)
        (*order)[(*count)++] = node;
}

/**
 * @brief Perform Grad Graph Toposort.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_graph_toposort(SNEPPXGradGraph* graph) {
    if (!graph || graph->num_nodes == 0) return 0;
    free(graph->topological_order);
    graph->topological_order = NULL;
    size_t capacity = graph->num_nodes > 0 ? graph->num_nodes : 1;
    graph->topological_order = (SNEPPXGradNode**)malloc(capacity * sizeof(SNEPPXGradNode*));
    if (!graph->topological_order) return -1;
    size_t count = 0;
    int visited[65536] = {0};
    for (size_t i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i])
            toposort_dfs(graph->nodes[i], &graph->topological_order, &count, &capacity, visited);
    }
    graph->needs_rebuild = 0;
    return 0;
}

/**
 * @brief Run the backward pass for Grad Graph.
 *
 * @param graph [out] Graph value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_graph_backward(SNEPPXGradGraph* graph, SNEPPXGradNode* root) {
    if (!graph || !root) return -1;
    if (graph->needs_rebuild) SNEPPX_grad_graph_toposort(graph);
    if (!graph->topological_order) return -1;

    for (size_t i = graph->num_nodes; i > 0; i--) {
        SNEPPXGradNode* node = graph->topological_order[i - 1];
        if (!node || !node->grad_fn) continue;
        ((void (*)(void*, void*))node->grad_fn)(node->saved_tensors, NULL);
    }
    return 0;
}

/**
 * @brief Perform Grad Save Tensor.
 *
 * @param node [out] Node value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_save_tensor(SNEPPXGradNode* node, void* tensor) {
    if (!node) return -1;
    node->saved_tensors = tensor;
    return 0;
}

/**
 * @brief Perform Grad Restore Tensor.
 *
 * @param node [out] Node value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_grad_restore_tensor(SNEPPXGradNode* node, int idx) {
    (void)idx;
    if (!node) return NULL;
    return node->saved_tensors;
}
