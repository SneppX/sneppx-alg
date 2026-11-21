#include "topology.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Topology
 *
 * WHAT
 *   Topology.
 *
 * CONCEPT
 *   Provides the Topology.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Topology Create Ring.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_ring(int world_size) {
    if (world_size < 1) return NULL;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (!t) return NULL;
    t->type = SNEPPX_TOPOLOGY_RING;
    t->world_size = world_size;
    t->nodes = (SNEPPXTopologyNode*)calloc((size_t)world_size, sizeof(SNEPPXTopologyNode));
    if (!t->nodes) { free(t); return NULL; }
    for (int i = 0; i < world_size; i++) {
        t->nodes[i].rank = i;
        t->nodes[i].prev_rank = (i - 1 + world_size) % world_size;
        t->nodes[i].next_rank = (i + 1) % world_size;
        t->nodes[i].parent_rank = -1;
        t->nodes[i].num_children = 0;
        snprintf(t->nodes[i].host, sizeof(t->nodes[i].host), "node_%d", i);
        t->nodes[i].data_port = 9876 + i;
    }
    return t;
}

/**
 * @brief Perform Topology Create Tree.
 *
 * @param world_size [in] World Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_tree(int world_size, int branching_factor) {
    if (world_size < 1 || branching_factor < 1) return NULL;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (!t) return NULL;
    t->type = SNEPPX_TOPOLOGY_TREE;
    t->world_size = world_size;
    t->nodes = (SNEPPXTopologyNode*)calloc((size_t)world_size, sizeof(SNEPPXTopologyNode));
    if (!t->nodes) { free(t); return NULL; }
    for (int i = 0; i < world_size; i++) {
        t->nodes[i].rank = i;
        t->nodes[i].prev_rank = -1;
        t->nodes[i].next_rank = -1;
        t->nodes[i].parent_rank = (i == 0) ? -1 : (i - 1) / branching_factor;
        t->nodes[i].num_children = 0;
        snprintf(t->nodes[i].host, sizeof(t->nodes[i].host), "node_%d", i);
        t->nodes[i].data_port = 9876 + i;
    }
    for (int i = 0; i < world_size; i++) {
        int base = i * branching_factor + 1;
        for (int j = 0; j < branching_factor; j++) {
            int child = base + j;
            if (child < world_size && t->nodes[i].num_children < 16) {
                t->nodes[i].children[t->nodes[i].num_children++] = child;
            }
        }
    }
    return t;
}

/**
 * @brief Perform Topology Create Graph.
 *
 * @param world_size [in] World Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_graph(int world_size, const int* adjacency_matrix) {
    if (world_size < 1 || !adjacency_matrix) return NULL;
    SNEPPXTopology* t = (SNEPPXTopology*)calloc(1, sizeof(SNEPPXTopology));
    if (!t) return NULL;
    t->type = SNEPPX_TOPOLOGY_GRAPH;
    t->world_size = world_size;
    t->nodes = (SNEPPXTopologyNode*)calloc((size_t)world_size, sizeof(SNEPPXTopologyNode));
    if (!t->nodes) { free(t); return NULL; }
    for (int i = 0; i < world_size; i++) {
        t->nodes[i].rank = i;
        t->nodes[i].prev_rank = -1;
        t->nodes[i].next_rank = -1;
        t->nodes[i].parent_rank = -1;
        t->nodes[i].num_children = 0;
        snprintf(t->nodes[i].host, sizeof(t->nodes[i].host), "node_%d", i);
        t->nodes[i].data_port = 9876 + i;
        for (int j = 0; j < world_size; j++) {
            if (adjacency_matrix[i * world_size + j] && t->nodes[i].num_children < 16) {
                t->nodes[i].children[t->nodes[i].num_children++] = j;
            }
        }
    }
    return t;
}

/**
 * @brief Destroy Topology.
 */
void SNEPPX_topology_destroy(SNEPPXTopology* topo) {
    if (topo) {
        free(topo->nodes);
        free(topo);
    }
}

/**
 * @brief Perform Topology Get Prev.
 *
 * @param topo [in] Topo value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_prev(const SNEPPXTopology* topo, int rank) {
    if (!topo || rank < 0 || rank >= topo->world_size) return -1;
    if (topo->type == SNEPPX_TOPOLOGY_RING) {
        return topo->nodes[rank].prev_rank;
    }
    return (rank > 0) ? rank - 1 : -1;
}

/**
 * @brief Perform Topology Get Next.
 *
 * @param topo [in] Topo value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_next(const SNEPPXTopology* topo, int rank) {
    if (!topo || rank < 0 || rank >= topo->world_size) return -1;
    if (topo->type == SNEPPX_TOPOLOGY_RING) {
        return topo->nodes[rank].next_rank;
    }
    return (rank < topo->world_size - 1) ? rank + 1 : -1;
}

/**
 * @brief Perform Topology Get Parent.
 *
 * @param topo [in] Topo value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_parent(const SNEPPXTopology* topo, int rank) {
    if (!topo || rank < 0 || rank >= topo->world_size) return -1;
    return topo->nodes[rank].parent_rank;
}

/**
 * @brief Perform Topology Get Children.
 *
 * @param topo [in] Topo value.
 * @param rank [in] Rank value.
 * @param children [out] Children value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_children(const SNEPPXTopology* topo, int rank, int** children, int* count) {
    if (!topo || rank < 0 || rank >= topo->world_size) return -1;
    if (children) {
        *children = topo->nodes[rank].children;
    }
    if (count) {
        *count = topo->nodes[rank].num_children;
    }
    return 0;
}

static int find_route_ring(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len) {
    int fwd = 0, rev = 0;
    int cur = src;
    while (cur != dst) { cur = topo->nodes[cur].next_rank; fwd++; }
    cur = src;
    while (cur != dst) { cur = topo->nodes[cur].prev_rank; rev++; }
    int use_fwd = (fwd <= rev);
    int len = use_fwd ? fwd : rev;
    *path = (int*)malloc((size_t)(len + 1) * sizeof(int));
    if (!*path) return -1;
    cur = src;
    for (int i = 0; i <= len; i++) {
        (*path)[i] = cur;
        cur = use_fwd ? topo->nodes[cur].next_rank : topo->nodes[cur].prev_rank;
    }
    *path_len = len + 1;
    return 0;
}

static int find_route_tree(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len) {
    int src_anc[1024], dst_anc[1024];
    int src_n = 0, dst_n = 0;
    int cur = src;
    while (cur >= 0 && src_n < 1024) { src_anc[src_n++] = cur; cur = topo->nodes[cur].parent_rank; }
    cur = dst;
    while (cur >= 0 && dst_n < 1024) { dst_anc[dst_n++] = cur; cur = topo->nodes[cur].parent_rank; }
    int lca = -1;
    for (int i = 0; i < src_n && lca < 0; i++) {
        for (int j = 0; j < dst_n; j++) {
            if (src_anc[i] == dst_anc[j]) { lca = src_anc[i]; break; }
        }
    }
    if (lca < 0) { *path = NULL; *path_len = 0; return 0; }
    int src_to_lca = 0;
    while (src_n - 1 - src_to_lca >= 0 && src_anc[src_n - 1 - src_to_lca] != lca) src_to_lca++;
    int lca_to_dst = 0;
    while (dst_n - 1 - lca_to_dst >= 0 && dst_anc[dst_n - 1 - lca_to_dst] != lca) lca_to_dst++;
    int total = src_to_lca + 1 + lca_to_dst;
    *path = (int*)malloc((size_t)total * sizeof(int));
    if (!*path) return -1;
    int idx = 0;
    for (int i = 0; i < src_to_lca; i++) (*path)[idx++] = src_anc[src_n - 1 - i];
    (*path)[idx++] = lca;
    for (int i = lca_to_dst - 1; i >= 0; i--) (*path)[idx++] = dst_anc[dst_n - 1 - i];
    *path_len = total;
    return 0;
}

/**
 * @brief Perform Topology Compute Route.
 *
 * @param topo [in] Topo value.
 * @param src [in] Src value.
 * @param dst [in] Dst value.
 * @param path [out] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_compute_route(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len) {
    if (!topo || !path || !path_len) return -1;
    if (src < 0 || src >= topo->world_size || dst < 0 || dst >= topo->world_size) return -1;
    if (src == dst) {
        *path = (int*)malloc(sizeof(int));
        if (!*path) return -1;
        (*path)[0] = src;
        *path_len = 1;
        return 0;
    }
    switch (topo->type) {
        case SNEPPX_TOPOLOGY_RING: return find_route_ring(topo, src, dst, path, path_len);
        case SNEPPX_TOPOLOGY_TREE: return find_route_tree(topo, src, dst, path, path_len);
        case SNEPPX_TOPOLOGY_GRAPH: {
            *path = (int*)malloc(2 * sizeof(int));
            if (!*path) return -1;
            (*path)[0] = src;
            (*path)[1] = dst;
            *path_len = 2;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Topology Free Route.
 */
void SNEPPX_topology_free_route(int* path) { free(path); }
