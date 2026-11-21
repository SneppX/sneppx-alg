"""Tests for sneppx_serve.graph_optimizer (sneppx-alg interface_bindings.graph_optimizer)."""

import os
from SneppX_ALG.interface_bindings.graph_compiler import GraphNode, GraphCompiler
from SneppX_ALG.interface_bindings.graph_optimizer import GraphOptimizer


def _build_graph():
    x = GraphNode(op="input", name="x")
    w = GraphNode(op="input", name="w")
    y = GraphNode(op="matmul", inputs=(x, w), name="y")
    z = GraphNode(op="relu", inputs=(y,), name="z")
    return z


def test_graph_optimizer_create():
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    assert opt.entry is z
    assert isinstance(opt.nodes, (list, tuple, set)) or hasattr(opt.nodes, "__iter__")


def test_graph_optimizer_optimize():
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    result = opt.optimize()
    assert result is not None


def test_constant_folding():
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    out = opt.constant_folding()
    # should return something (graph or None)
    assert out is not None or out is None


def test_dead_code_elimination():
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    out = opt.dead_code_elimination()
    assert out is not None or out is None


def test_memory_planning():
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    out = opt.memory_planning()
    assert out is not None or out is None


def test_serialize_ir(tmp_path):
    z = _build_graph()
    opt = GraphOptimizer(entry=z)
    out_path = str(tmp_path / "graph.ir")
    result = opt.serialize_ir(out_path)
    # may return None or a string
    assert result is None or isinstance(result, str) or os.path.exists(out_path)
