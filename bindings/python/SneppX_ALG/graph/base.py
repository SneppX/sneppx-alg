"""A small adjacency-based graph structure (pure Python)."""

from typing import Dict, Iterable, List, Optional, Set, Tuple


class Graph:
    """Directed or undirected weighted graph.

    Nodes may be any hashable value; edge weights default to 1.0.
    """

    def __init__(self, directed: bool = False):
        self.directed = bool(directed)
        self._adj: Dict[object, List[Tuple[object, float]]] = {}
        self._nodes: Set[object] = set()

    def add_node(self, node) -> None:
        self._nodes.add(node)
        self._adj.setdefault(node, [])

    def add_nodes(self, nodes: Iterable) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, u, v, weight: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        self._adj[u].append((v, float(weight)))
        if not self.directed:
            self._adj[v].append((u, float(weight)))

    def add_edges(self, edges: Iterable) -> None:
        for e in edges:
            if len(e) == 3:
                self.add_edge(e[0], e[1], e[2])
            else:
                self.add_edge(e[0], e[1])

    def has_node(self, node) -> bool:
        return node in self._nodes

    def has_edge(self, u, v) -> bool:
        return any(nbr == v for nbr, _ in self._adj.get(u, []))

    def neighbors(self, node) -> List[Tuple[object, float]]:
        return list(self._adj.get(node, []))

    def nodes(self) -> List[object]:
        return list(self._nodes)

    def edges(self) -> List[Tuple[object, object, float]]:
        seen = set()
        out = []
        for u, nbrs in self._adj.items():
            for v, w in nbrs:
                if self.directed or (v, u) not in seen:
                    out.append((u, v, w))
                    seen.add((u, v))
        return out

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        total = sum(len(v) for v in self._adj.values())
        return total if self.directed else total // 2

    def degree(self, node) -> int:
        return len(self._adj.get(node, []))

    def degree_sequence(self) -> List[int]:
        return [len(self._adj.get(n, [])) for n in self._nodes]

    def adjacency_matrix(self, nodes: Optional[List] = None):
        import numpy as np

        nodes = list(self._nodes) if nodes is None else list(nodes)
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        A = np.zeros((n, n), dtype=np.float64)
        for u, nbrs in self._adj.items():
            for v, w in nbrs:
                if u in index and v in index:
                    A[index[u], index[v]] = w
        return A

    @classmethod
    def from_edges(
        cls, edges: Iterable, directed: bool = False, num_nodes: Optional[int] = None
    ) -> "Graph":
        g = cls(directed=directed)
        if num_nodes is not None:
            g.add_nodes(range(num_nodes))
        g.add_edges(edges)
        return g

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        kind = "DiGraph" if self.directed else "Graph"
        return f"{kind}(nodes={len(self._nodes)}, edges={self.number_of_edges()})"
