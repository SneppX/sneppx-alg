"""Graph traversal and connectivity (pure Python)."""

from collections import deque
from typing import Dict, List

from .base import Graph


def bfs(graph: Graph, start) -> List:
    """Breadth-first traversal order from ``start``."""
    if not graph.has_node(start):
        return []
    visited = {start}
    order = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nbr, _ in graph.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return order


def dfs(graph: Graph, start=None) -> List:
    """Depth-first traversal order (iterative)."""
    if start is None:
        starts = graph.nodes()
    else:
        starts = [start]
    visited = set()
    order = []
    for s in starts:
        if s in visited:
            continue
        stack = [s]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for nbr, _ in reversed(graph.neighbors(node)):
                if nbr not in visited:
                    stack.append(nbr)
    return order


def connected_components(graph: Graph) -> List[List[object]]:
    """Connected components (treats a directed graph as undirected/weakly)."""
    adj = {n: set() for n in graph.nodes()}
    for u in graph.nodes():
        for v, _ in graph.neighbors(u):
            adj[u].add(v)
            adj[v].add(u)
    seen = set()
    comps = []
    for node in graph.nodes():
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        comps.append(comp)
    return comps


def topological_sort(graph: Graph) -> List[object]:
    """Kahn's algorithm; raises ``ValueError`` if the graph has a cycle."""
    indeg: Dict[object, int] = {n: 0 for n in graph.nodes()}
    for u in graph.nodes():
        for v, _ in graph.neighbors(u):
            indeg[v] = indeg.get(v, 0) + 1
    queue = [n for n in graph.nodes() if indeg[n] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v, _ in graph.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != len(graph.nodes()):
        raise ValueError("graph contains a cycle")
    return order


def has_cycle(graph: Graph) -> bool:
    """True if the directed graph contains a cycle."""
    try:
        topological_sort(graph)
        return False
    except ValueError:
        return True
