"""Centrality measures and PageRank (pure Python)."""

from collections import deque
from typing import Dict

from .base import Graph


def degree_centrality(graph: Graph) -> Dict[object, float]:
    """Degree normalized by ``n - 1``."""
    n = len(graph.nodes())
    if n <= 1:
        return {u: 0.0 for u in graph.nodes()}
    return {u: len(graph.neighbors(u)) / (n - 1) for u in graph.nodes()}


def closeness_centrality(graph: Graph) -> Dict[object, float]:
    """Closeness = ``(reachable - 1) / sum(distances)`` (BFS, unweighted)."""
    nodes = graph.nodes()
    out = {}
    for s in nodes:
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, _ in graph.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        reachable = len(dist) - 1
        total = sum(dist.values())
        out[s] = (reachable / total) if total > 0 else 0.0
    return out


def betweenness_centrality(
    graph: Graph, normalized: bool = True
) -> Dict[object, float]:
    """Brandes betweenness centrality (unweighted)."""
    nodes = graph.nodes()
    cb: Dict[object, float] = {v: 0.0 for v in nodes}
    for s in nodes:
        S = []
        P: Dict[object, list] = {v: [] for v in nodes}
        sigma: Dict[object, float] = {v: 0.0 for v in nodes}
        sigma[s] = 1.0
        d: Dict[object, int] = {v: -1 for v in nodes}
        d[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            S.append(v)
            for w, _ in graph.neighbors(v):
                if d[w] < 0:
                    q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta: Dict[object, float] = {v: 0.0 for v in nodes}
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    if normalized:
        n = len(nodes)
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            if not graph.directed:
                scale *= 2.0
            cb = {k: v * scale for k, v in cb.items()}
    return cb


def pagerank(
    graph: Graph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[object, float]:
    """PageRank via the power iteration."""
    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}
    rank = {u: 1.0 / n for u in nodes}
    out_degree = {u: max(1, len(graph.neighbors(u))) for u in nodes}
    for _ in range(max_iter):
        new = {u: (1.0 - alpha) / n for u in nodes}
        dangling = sum(rank[u] for u in nodes if len(graph.neighbors(u)) == 0)
        base = alpha * dangling / n
        for u in nodes:
            new[u] += base
            if len(graph.neighbors(u)) == 0:
                continue
            share = alpha * rank[u] / out_degree[u]
            for v, _ in graph.neighbors(u):
                new[v] += share
        diff = sum(abs(new[u] - rank[u]) for u in nodes)
        rank = new
        if diff < tol:
            break
    return rank
