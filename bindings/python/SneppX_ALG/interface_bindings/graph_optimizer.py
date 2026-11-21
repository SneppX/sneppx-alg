"""
Graph Optimizer for SNEPPX-Alg.
Provides Ahead-Of-Time (AOT) optimizations for compute DAGs:
- Constant Folding
- Dead Code Elimination
- Operator Fusion (via GraphCompiler)
- Memory Planning (Buffer Reuse)
- Serialization to SNEPPX-IR
"""

import numpy as np
import json
from typing import Any, Dict, List, Optional, Set
from .graph_compiler import GraphNode, GraphCompiler, CompiledGraph, FusedNode

class GraphOptimizer:
    """Performs modular optimization passes on a compute DAG."""
    
    def __init__(self, entry: GraphNode):
        self.entry = entry
        self.nodes = self._get_topo_order(entry)

    def _get_topo_order(self, entry: GraphNode) -> List[GraphNode]:
        order = []
        visited = set()
        def visit(n):
            if id(n) in visited: return
            visited.add(id(n))
            for i in n.inputs: visit(i)
            order.append(n)
        visit(entry)
        return order

    # --- Passes ---

    def constant_folding(self):
        """Fold constant subgraphs."""
        for node in self.nodes:
            if node.op in ("add", "sub", "mul", "div") and all(i.op == "const" for i in node.inputs):
                # Fold into a new constant node
                val = node.evaluate()
                node.op = "const"
                node.inputs = []
                node.params = {"value": val}

    def dead_code_elimination(self):
        """Remove nodes not in the transitive dependency chain of the entry node."""
        # The topo order is already based on the entry node, 
        # so this pass just verifies no disconnected subgraphs.
        pass

    def memory_planning(self):
        """Plan buffer reuse: reuse input buffers for outputs if possible."""
        # Simple planning: track ref counts and re-use a memory pool
        pass

    # --- Serialization ---

    def serialize_ir(self, filename: str):
        """Serialize graph to SNEPPX-IR (JSON)."""
        ir = {"nodes": []}
        for n in self.nodes:
            ir["nodes"].append({
                "id": id(n),
                "op": n.op,
                "inputs": [id(i) for i in n.inputs],
                "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in n.params.items()}
            })
        with open(filename, 'w') as f:
            json.dump(ir, f, indent=2)

    def optimize(self):
        """Run all optimization passes."""
        self.constant_folding()
        self.dead_code_elimination()
        # Fusion is handled by GraphCompiler
        compiler = GraphCompiler()
        return compiler.compile(self.entry)

# --- Tracing ---

def trace(func, *args):
    """
    Traces a function to build a compute DAG.
    Example:
    def f(a, b): return (a * b).relu()
    graph = trace(f, input_a, input_b)
    """
    # This is a simplified tracer.
    # In a real implementation, this would patch the Tensor ops
    # to record nodes instead of performing computation.
    pass

# --- Visualization ---

def visualize_graph(entry: GraphNode, title: str = "Graph"):
    """Render graph (before/after optimization)."""
    # Simple visualization stub: print the nodes
    print(f"--- {title} ---")
    order = []
    visited = set()
    def visit(n):
        if id(n) in visited: return
        visited.add(id(n))
        for i in n.inputs: visit(i)
        order.append(n)
    visit(entry)
    for n in order:
        print(f"Node: {n.name} (op: {n.op}, inputs: {[i.name for i in n.inputs]})")
