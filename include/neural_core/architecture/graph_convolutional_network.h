#ifndef SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#define SNEPPX_GRAPH_CONVOLUTIONAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Graph Convolutional Network
 *
 * WHAT
 *   Graph Convolutional Network.
 *
 * CONCEPT
 *   GCN configuration and forward API for graph-structured data.
 *
 * ROLE
 *   Declares the GCN API for graph neural network operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXGCN SNEPPXGCN;

/**
 * @brief Create Gcn.
 *
 * @param in_features [in] In Features value.
 * @param out_features [in] Out Features value.
 * @param hidden_features [in] Hidden Features value.
 * @param num_layers [in] Num Layers value.
 * @param dropout [in] Dropout value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGCN* SNEPPX_gcn_create(size_t in_features, size_t out_features, size_t hidden_features,
    int num_layers, float dropout);
/**
 * @brief Destroy Gcn.
 *
 * @param gcn [out] Gcn value.
 */
void SNEPPX_gcn_destroy(void* gcn);
/**
 * @brief Run the forward pass for Gcn.
 *
 * @param gcn [out] Gcn value.
 * @param adj [in] Adj value.
 * @param features [in] Features value.
 * @param num_nodes [in] Num Nodes value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcn_forward(void* gcn, const float* adj, const float* features, size_t num_nodes, float* output);

#ifdef __cplusplus
}
#endif
#endif
