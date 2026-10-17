from SneppX_ALG import graph as gr


def test_graph_basics():
    g = gr.Graph.from_edges([(0, 1, 2.0), (1, 2, 3.0), (0, 2, 10.0)])
    assert len(g) == 3
    assert g.number_of_edges() == 3
    assert g.has_edge(0, 1) and g.has_edge(1, 0)
    assert sorted(g.nodes()) == [0, 1, 2]
    A = g.adjacency_matrix()
    assert A.shape == (3, 3)
    assert A[0, 1] == 2.0


def test_traversal():
    g = gr.Graph.from_edges([(0, 1), (0, 2), (1, 3), (2, 3)])
    assert gr.bfs(g, 0)[0] == 0
    assert set(gr.bfs(g, 0)) == {0, 1, 2, 3}
    assert set(gr.dfs(g, 0)) == {0, 1, 2, 3}
    comps = gr.connected_components(g)
    assert len(comps) == 1 and set(comps[0]) == {0, 1, 2, 3}


def test_topological_sort():
    d = gr.Graph.from_edges([(0, 1), (1, 2), (0, 2)], directed=True)
    order = gr.topological_sort(d)
    assert order.index(0) < order.index(1) < order.index(2)
    cyc = gr.Graph.from_edges([(0, 1), (1, 0)], directed=True)
    assert gr.has_cycle(cyc)


def test_dijkstra_bellman_floyd():
    g = gr.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)])
    dist, prev = gr.dijkstra(g, 0)
    assert dist[2] == 3.0
    assert gr.reconstruct_path(prev, 2) == [0, 1, 2]
    d2, _, neg = gr.bellman_ford(g, 0)
    assert d2[2] == 3.0 and not neg
    D, idx = gr.floyd_warshall(g)
    assert D[idx[0], idx[2]] == 3.0


def test_spanning_tree():
    g = gr.Graph.from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])
    mst, total = gr.kruskal(g)
    assert len(mst) == 2 and abs(total - 3.0) < 1e-9
    mst2, total2 = gr.prim(g, 0)
    assert len(mst2) == 2 and abs(total2 - 3.0) < 1e-9


def test_disjoint_set():
    ds = gr.DisjointSet([0, 1, 2])
    assert ds.union(0, 1)
    assert not ds.union(0, 1)
    assert ds.connected(0, 1)
    assert not ds.connected(0, 2)


def test_centrality():
    g = gr.Graph.from_edges([(0, 1), (1, 2), (2, 0)])
    dc = gr.degree_centrality(g)
    assert all(abs(v - 1.0) < 1e-9 for v in dc.values())
    cc = gr.closeness_centrality(g)
    assert all(abs(v - 1.0) < 1e-9 for v in cc.values())
    bc = gr.betweenness_centrality(g)
    assert all(abs(v) < 1e-9 for v in bc.values())


def test_pagerank():
    g = gr.Graph.from_edges([(0, 1), (1, 2), (2, 0)], directed=True)
    pr = gr.pagerank(g)
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    assert all(abs(v - 1.0 / 3) < 1e-3 for v in pr.values())
