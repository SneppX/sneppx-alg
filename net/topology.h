#ifndef SNEPPX_NET_TOPOLOGY_H
#define SNEPPX_NET_TOPOLOGY_H
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


/*
 * Network Topology Abstraction — v1.0 (distributed communication patterns)
 *
 * PURPOSE: Maps nodes to logical network positions for collective
 * communication.  Supports ring (all-reduce), tree (broadcast), and
 * arbitrary graph (gossip) topologies.  Used by the distributed
 * training coordinator to choose communication schedules.
 *
 * DEPENDENCIES: SNEPPX_grpc_service.h, SNEPPX_socket_comm.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_TOPOLOGY_RING,
    SNEPPX_TOPOLOGY_TREE,
    SNEPPX_TOPOLOGY_GRAPH,
} SNEPPXTopologyType;

typedef struct {
    int    rank;
    int    prev_rank;
    int    next_rank;
    int    parent_rank;
    int    children[16];
    int    num_children;
    char   host[256];
    int    data_port;
} SNEPPXTopologyNode;

typedef struct {
    SNEPPXTopologyType  type;
    int               world_size;
    SNEPPXTopologyNode* nodes;
} SNEPPXTopology;

/* ---------- Topology construction ---------- */
/**
 * @brief Perform Topology Create Ring.
 *
 * @param world_size [in] World Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_ring(int world_size);
/**
 * @brief Perform Topology Create Tree.
 *
 * @param world_size [in] World Size value.
 * @param branching_factor [in] Branching Factor value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_tree(int world_size, int branching_factor);
/**
 * @brief Perform Topology Create Graph.
 *
 * @param world_size [in] World Size value.
 * @param adjacency_matrix [in] Adjacency Matrix value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTopology* SNEPPX_topology_create_graph(int world_size, const int* adjacency_matrix);
/**
 * @brief Destroy Topology.
 *
 * @param topo [out] Topo value.
 */
void          SNEPPX_topology_destroy(SNEPPXTopology* topo);

/* ---------- Neighbor queries ---------- */
/**
 * @brief Perform Topology Get Prev.
 *
 * @param topo [in] Topo value.
 * @param rank [in] Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_prev(const SNEPPXTopology* topo, int rank);
/**
 * @brief Perform Topology Get Next.
 *
 * @param topo [in] Topo value.
 * @param rank [in] Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_next(const SNEPPXTopology* topo, int rank);
/**
 * @brief Perform Topology Get Parent.
 *
 * @param topo [in] Topo value.
 * @param rank [in] Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_parent(const SNEPPXTopology* topo, int rank);
/**
 * @brief Perform Topology Get Children.
 *
 * @param topo [in] Topo value.
 * @param rank [in] Rank value.
 * @param children [out] Children value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_topology_get_children(const SNEPPXTopology* topo, int rank, int** children, int* count);

/* ---------- Route calculation (v1.0) ---------- */
/**
 * @brief Perform Topology Compute Route.
 *
 * @param topo [in] Topo value.
 * @param src [in] Src value.
 * @param dst [in] Dst value.
 * @param path [out] Path value.
 * @param path_len [out] Path Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_topology_compute_route(const SNEPPXTopology* topo, int src, int dst, int** path, int* path_len);
/**
 * @brief Perform Topology Free Route.
 *
 * @param path [out] Path value.
 */
void SNEPPX_topology_free_route(int* path);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_NET_TOPOLOGY_H */
