"""Shortest-path algorithms (pure Python)."""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Graph


def dijkstra(
    graph: Graph, source
) -> Tuple[Dict[object, float], Dict[object, Optional[object]]]:
    """Single-source shortest paths (non-negative weights).

    Returns ``(dist, prev)``.
    """
    dist: Dict[object, float] = {n: float("inf") for n in graph.nodes()}
    prev: Dict[object, Optional[object]] = {n: None for n in graph.nodes()}
    if not graph.has_node(source):
        return dist, prev
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.neighbors(u):
            if w < 0:
                raise ValueError("dijkstra requires non-negative weights")
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[object, Optional[object]], target) -> List[object]:
    """Rebuild a path from a ``prev`` map."""
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path[::-1]


def bellman_ford(
    graph: Graph, source
) -> Tuple[Dict[object, float], Dict[object, Optional[object]], bool]:
    """Bellman-Ford; supports negative weights.

    Returns ``(dist, prev, has_negative_cycle)``.
    """
    nodes = graph.nodes()
    dist: Dict[object, float] = {n: float("inf") for n in nodes}
    prev: Dict[object, Optional[object]] = {n: None for n in nodes}
    if source not in dist:
        return dist, prev, False
    dist[source] = 0.0
    edges = graph.edges()
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break
    negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            negative_cycle = True
            break
    return dist, prev, negative_cycle


def floyd_warshall(
    graph: Graph, num_nodes: Optional[int] = None, node_index: Optional[dict] = None
):
    """All-pairs shortest paths over the adjacency matrix.

    Returns ``(dist_matrix, index)`` where ``index`` maps node → row.
    """
    nodes = graph.nodes()
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    INF = float("inf")
    D = np.full((n, n), INF, dtype=np.float64)
    for i in range(n):
        D[i, i] = 0.0
    for u, v, w in graph.edges():
        D[index[u], index[v]] = min(D[index[u], index[v]], w)
        if not graph.directed:
            D[index[v], index[u]] = min(D[index[v], index[u]], w)
    for k in range(n):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D, index
