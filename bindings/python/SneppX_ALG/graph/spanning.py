"""Minimum spanning trees and union-find (pure Python)."""

import heapq
from typing import List, Tuple

from .base import Graph


class DisjointSet:
    """Union-find with path compression and union by rank."""

    def __init__(self, nodes=None):
        self.parent = {}
        self.rank = {}
        if nodes is not None:
            for n in nodes:
                self.make_set(n)

    def make_set(self, x) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> bool:
        """Merge the sets of ``a`` and ``b``; return False if already joined."""
        self.make_set(a)
        self.make_set(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def connected(self, a, b) -> bool:
        return self.find(a) == self.find(b)


def kruskal(graph: Graph) -> Tuple[List[Tuple[object, object, float]], float]:
    """Kruskal MST; returns ``(edges, total_weight)``."""
    ds = DisjointSet(graph.nodes())
    edges = sorted(graph.edges(), key=lambda e: e[2])
    mst = []
    total = 0.0
    for u, v, w in edges:
        if ds.union(u, v):
            mst.append((u, v, w))
            total += w
    return mst, total


def prim(graph: Graph, start=None) -> Tuple[List[Tuple[object, object, float]], float]:
    """Prim MST; returns ``(edges, total_weight)``."""
    nodes = graph.nodes()
    if not nodes:
        return [], 0.0
    start = start if start is not None else nodes[0]
    visited = {start}
    heap = [(w, start, v) for v, w in graph.neighbors(start)]
    heapq.heapify(heap)
    mst = []
    total = 0.0
    while heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, w))
        total += w
        for nbr, nw in graph.neighbors(v):
            if nbr not in visited:
                heapq.heappush(heap, (nw, v, nbr))
    return mst, total
