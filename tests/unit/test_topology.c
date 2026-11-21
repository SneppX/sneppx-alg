#include "../../net/topology.h"
#include "test_gtest.h"
#include "../../net/topology.c"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Topology
 *
 * WHAT
 *   Test Topology.
 *
 * CONCEPT
 *   Provides the Test Topology.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_topology_create_destroy(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_ring(4);
    SX_ASSERT(topo != NULL, "topology created");
    SX_ASSERT(topo->type == SNEPPX_TOPOLOGY_RING, "ring type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_ring(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_ring(8);
    SX_ASSERT(topo != NULL, "ring topology");
    SX_ASSERT(topo->type == SNEPPX_TOPOLOGY_RING, "ring type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_tree(void) {
    SNEPPXTopology* topo = SNEPPX_topology_create_tree(8, 2);
    SX_ASSERT(topo != NULL, "tree topology");
    SX_ASSERT(topo->type == SNEPPX_TOPOLOGY_TREE, "tree type");
    SNEPPX_topology_destroy(topo);
}

static void test_topology_graph(void) {
    int adj[9 * 9];
    memset(adj, 0, sizeof(adj));
    adj[0 * 9 + 1] = 1;
    adj[1 * 9 + 2] = 1;
    adj[2 * 9 + 0] = 1;
    SNEPPXTopology* topo = SNEPPX_topology_create_graph(9, adj);
    SX_ASSERT(topo != NULL, "graph topology");
    SX_ASSERT(topo->type == SNEPPX_TOPOLOGY_GRAPH, "graph type");
    SNEPPX_topology_destroy(topo);
}


TEST(test_topology, topology_create_destroy) { test_topology_create_destroy(); }
TEST(test_topology, topology_ring) { test_topology_ring(); }
TEST(test_topology, topology_tree) { test_topology_tree(); }
TEST(test_topology, topology_graph) { test_topology_graph(); }
