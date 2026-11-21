import math

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.graph_compiler import (
    CompiledGraph,
    FusedNode,
    FUSABLE_OPS,
    GraphCompiler,
    GraphNode,
)


def np_silu(x):
    return x / (1.0 + np.exp(-x))


def np_gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


class TestGraphNode:
    def test_factories(self):
        a = GraphNode.input_("a")
        c = GraphNode.constant(3.0)
        p = GraphNode.parameter(np.array([1.0, 2.0]))
        assert a.op == "input"
        assert c.params["value"] == 3.0
        assert p.params["value"].tolist() == [1.0, 2.0]

    def test_arithmetic_builds_dag(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        c = (a * b) + b
        assert c.op == "add"
        assert c.inputs[0].op == "mul"
        assert c.inputs[1] is b

    def test_scalar_coerced_to_const(self):
        a = GraphNode.input_("a")
        c = a * 2.0
        assert c.inputs[1].op == "const"
        assert c.inputs[1].params["value"] == 2.0

    def test_unary_helpers(self):
        a = GraphNode.input_("a")
        for node, op in [
            (a.relu(), "relu"),
            (a.sigmoid(), "sigmoid"),
            (a.tanh(), "tanh"),
            (a.gelu(), "gelu"),
            (a.silu(), "silu"),
        ]:
            assert node.op == op
            assert node.inputs == [a]

    def test_clip_helper(self):
        a = GraphNode.input_("a")
        c = GraphNode.clip(a, -1.0, 1.0)
        assert c.op == "clip"
        assert c.params["min"] == -1.0
        assert c.params["max"] == 1.0

    def test_evaluate_basic(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        assert np.allclose((a + b).evaluate({"a": x, "b": y}), x + y)
        assert np.allclose((a * 2.0).evaluate({"a": x}), x * 2.0)

    def test_evaluate_chain(self):
        a = GraphNode.input_("a")
        x = np.array([-2.0, 0.0, 3.0])
        out = GraphNode.clip(a.relu(), 0.5, 2.0).evaluate({"a": x})
        assert np.allclose(out, [0.5, 0.5, 2.0])

    def test_evaluate_matmul(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        x = np.random.randn(4, 3)
        y = np.random.randn(3, 5)
        assert np.allclose(a.matmul(b).evaluate({"a": x, "b": y}), x @ y)

    def test_evaluate_unknown_op_raises(self):
        n = GraphNode("bogus", [GraphNode.input_("a")])
        with pytest.raises(ValueError):
            n.evaluate({"a": np.ones(3)})

    def test_const_evaluate_by_name_binding(self):
        a = GraphNode.input_("x")
        out = (a * 3.0).evaluate({"x": np.array([1.0, 2.0])})
        assert np.allclose(out, [3.0, 6.0])


class TestFusion:
    def test_compile_chain_fuses_into_single_node(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        c = (a * b) + b
        out = c.relu()
        compiler = GraphCompiler()
        compiled = compiler.compile(out)
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        assert len(fused) == 1
        assert {m.op for m in fused[0].members} == {"mul", "add", "relu"}

    def test_fused_forward_matches_unfused(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = ((a * b) + b).relu()
        compiler = GraphCompiler()
        compiled = compiler.compile(out)
        x = np.random.randn(8, 16)
        y = np.random.randn(8, 16)
        expected = np.maximum(x * y + y, 0.0)
        assert np.allclose(compiled.forward({"a": x, "b": y}), expected)

    def test_multiple_clusters_stay_separate(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        c = GraphNode.input_("c")
        left = (a * b).relu()
        right = c.tanh()
        out = left.matmul(right)
        compiled = GraphCompiler().compile(out)
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        assert len(fused) == 2
        member_ops = {frozenset(m.op for m in f.members) for f in fused}
        assert frozenset({"mul", "relu"}) in member_ops
        assert frozenset({"tanh"}) in member_ops
        x = np.random.randn(4, 6)
        y = np.random.randn(4, 6)
        z = np.random.randn(6, 4)
        expected = np.maximum(x * y, 0.0) @ np.tanh(z)
        assert np.allclose(compiled.forward({"a": x, "b": y, "c": z}), expected)

    def test_matmul_breaks_fusion(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        w = GraphNode.input_("w")
        out = (a + b).matmul(w).relu()
        compiler = GraphCompiler()
        compiled = compiler.compile(out)
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        # matmul is a non-fusable boundary: add and relu stay in two clusters.
        assert len(fused) == 2
        assert frozenset(m.op for m in fused[0].members) == {"add"}
        assert frozenset(m.op for m in fused[1].members) == {"relu"}
        x = np.random.randn(4, 6)
        y = np.random.randn(4, 6)
        wv = np.random.randn(6, 3)
        expected = np.maximum((x + y) @ wv, 0.0)
        assert np.allclose(compiled.forward({"a": x, "b": y, "w": wv}), expected)

    def test_shared_subexpression_only_computed_once(self):
        a = GraphNode.input_("a")
        c = a * 2.0
        out = c + c
        compiled = GraphCompiler().compile(out)
        # The whole chain collapses into a single fused kernel.
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        assert len(fused) == 1
        assert frozenset(m.op for m in fused[0].members) == {"mul", "add"}
        x = np.random.randn(10)
        assert np.allclose(compiled.forward({"a": x}), x * 4.0)

    def test_unfused_matmul_standalone_graph(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        compiled = GraphCompiler().compile(a.matmul(b))
        x = np.random.randn(3, 4)
        y = np.random.randn(4, 5)
        assert np.allclose(compiled.forward({"a": x, "b": y}), x @ y)

    def test_compile_chain_broadcast_shapes(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = (a * b).sigmoid()
        compiled = GraphCompiler().compile(out)
        x = np.random.randn(8, 1)
        y = np.random.randn(1, 8)
        expected = 1.0 / (1.0 + np.exp(-(x * y)))
        assert np.allclose(compiled.forward({"a": x, "b": y}), expected)

    def test_compile_gelu_silu_parity(self):
        a = GraphNode.input_("a")
        for op in ("gelu", "silu"):
            out = getattr(a, op)()
            compiled = GraphCompiler().compile(out)
            x = np.random.randn(12)
            expected = np_gelu(x) if op == "gelu" else np_silu(x)
            assert np.allclose(compiled.forward({"a": x}), expected, atol=1e-6)

    def test_compile_clip_is_fused(self):
        a = GraphNode.input_("a")
        out = GraphNode.clip(a.silu(), -0.5, 1.5)
        compiled = GraphCompiler().compile(out)
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        assert len(fused) == 1
        x = np.random.randn(20)
        expected = np.clip(np_silu(x), -0.5, 1.5)
        assert np.allclose(compiled.forward({"a": x}), expected)

    def test_constants_inlined_into_fused_kernel(self):
        a = GraphNode.input_("a")
        out = (a * 2.0 + 1.0).relu()
        compiled = GraphCompiler().compile(out)
        fused = [n for n in compiled.nodes if isinstance(n, FusedNode)]
        assert len(fused) == 1
        x = np.random.randn(9)
        assert np.allclose(compiled.forward({"a": x}), np.maximum(x * 2.0 + 1.0, 0.0))

    def test_compile_chain_single_float_precision(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = (a + b).relu()
        compiled = GraphCompiler().compile(out)
        x = np.random.randn(6).astype(np.float32)
        y = np.random.randn(6).astype(np.float32)
        res = compiled.forward({"a": x, "b": y})
        assert res.dtype == np.float32
        assert np.allclose(res, np.maximum(x + y, 0.0))

    def test_fused_node_tiling_path(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = (a * b).relu().sigmoid()
        compiled = GraphCompiler().compile(out)
        x = np.random.randn(100, 4)
        y = np.random.randn(100, 4)
        full = compiled.forward({"a": x, "b": y})
        tiled = compiled.forward({"a": x, "b": y}, tile_size=17)
        assert np.allclose(full, tiled)

    def test_generator_version_number(self):
        assert isinstance(FUSABLE_OPS, set)
        for op in ("add", "sub", "mul", "div", "relu", "sigmoid", "tanh",
                   "gelu", "silu", "clip", "neg", "abs", "exp", "log"):
            assert op in FUSABLE_OPS


class TestCodegen:
    def _gen(self, out):
        return GraphCompiler().generate_c(GraphCompiler().compile(out))

    def test_emits_math_include(self):
        a = GraphNode.input_("a")
        code = self._gen(a.relu())
        assert "#include <math.h>" in code
        assert "#include <stddef.h>" in code

    def test_fused_kernel_has_loop_and_inline_ops(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = (a * b + b).relu()
        code = self._gen(out)
        assert "for (size_t i = 0; i < n; ++i) {" in code
        assert "((t1) > 0.0f ? (t1) : 0.0f)" in code
        assert "((a[i]) * (b[i]))" in code
        assert "((t0) + (b[i]))" in code

    def test_scalar_const_inlined_as_literal(self):
        a = GraphNode.input_("a")
        out = (a * 2.0 + 1.0).relu()
        code = self._gen(out)
        assert "((a[i]) * (2f))" in code
        assert "((t0) + (1f))" in code
        # kernel takes only one array param (a) plus out/n
        assert "static void fused_" in code
        assert "const float* a, float* out, size_t n" in code

    def test_gelu_silu_use_helpers(self):
        a = GraphNode.input_("a")
        code = self._gen(a.gelu())
        assert "sneppx_gelu_f32" in code
        code = self._gen(a.silu())
        assert "sneppx_silu_f32" in code

    def test_clip_emits_min_max_macro(self):
        a = GraphNode.input_("a")
        out = GraphNode.clip(a.sigmoid(), -0.25, 0.75)
        code = self._gen(out)
        assert "SNEPPX_MIN(SNEPPX_MAX" in code
        assert "0.25f" in code
        assert "0.75f" in code

    def test_matmul_graph_emits_helper_and_driver_dims(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        out = a.matmul(b)
        code = self._gen(out)
        assert "sneppx_matmul_f32" in code
        assert "size_t m, size_t k, size_t nn" in code
        assert "sneppx_matmul_f32(inputs[0], inputs[1], out, m, k, nn);" in code

    def test_driver_wires_scratch_buffers_in_order(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        w = GraphNode.input_("w")
        out = ((a + b).relu()).matmul(w)
        code = self._gen(out)
        assert "static void sneppx_graph_forward(" in code
        # fused kernel feeds scratch[0], then matmul consumes it
        assert "fused_" in code
        assert "scratch[0], n" in code
        assert "sneppx_matmul_f32(scratch[0], inputs[2], out, m, k, nn);" in code

    def test_driver_missing_matmul_omits_dims(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        code = self._gen((a + b).sigmoid())
        assert "size_t m, size_t k, size_t nn" not in code
        assert "fused_" in code
        assert "inputs[0], inputs[1], out, n);" in code

    def test_multi_input_kernel_param_order(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        c = GraphNode.input_("c")
        out = (a * b + c).tanh()
        code = self._gen(out)
        assert "const float* a, const float* b, const float* c, float* out, size_t n" in code
