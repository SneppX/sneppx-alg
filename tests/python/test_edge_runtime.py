import numpy as np
import pytest

from SneppX_ALG.interface_bindings.edge_runtime import (
    EdgeBackend,
    EdgeConfig,
    EdgeDevice,
    EdgeRuntime,
    CPUBackend,
    benchmark,
    detect_cpu,
    int8_matmul,
    quantize_graph_weights,
    register_backend,
)
from SneppX_ALG.interface_bindings.graph_compiler import GraphCompiler, GraphNode


def _mlp_graph(in_features=16, hidden=32, out_features=8):
    x = GraphNode.input_("x")
    w1 = GraphNode.parameter(np.random.randn(in_features, hidden) * 0.1)
    b1 = GraphNode.parameter(np.random.randn(hidden) * 0.1)
    h = x.matmul(w1) + b1
    h = h.relu()
    w2 = GraphNode.parameter(np.random.randn(hidden, out_features) * 0.1)
    y = h.matmul(w2)
    return GraphCompiler().compile(y), x


class TestDeviceDetection:
    def test_detect_cpu_shape(self):
        dev = detect_cpu()
        assert isinstance(dev, EdgeDevice)
        assert dev.cores >= 1
        assert dev.architecture
        assert dev.supports_int8 is True
        assert dev.vector_width_bytes >= 8

    def test_isa_has_baseline(self):
        dev = detect_cpu()
        assert len(dev.isa) >= 1

    def test_device_props(self):
        dev = EdgeDevice("x86_64", isa={"sse2", "avx2"}, cores=4)
        assert dev.vector_width_bytes == 16
        assert dev.supports_int8
        assert "EdgeDevice" in repr(dev)
        assert "x86_64" in repr(dev)

    def test_arm_fp16_detection(self):
        dev = EdgeDevice("aarch64", isa={"neon", "neon_fp16"}, cores=8)
        assert dev.supports_fp16
        dev2 = EdgeDevice("aarch64", isa={"neon"}, cores=8)
        assert not dev2.supports_fp16


class TestInt8Matmul:
    def test_matches_fp32_within_tolerance(self):
        a = np.random.randn(8, 16).astype(np.float32) * 2.0
        w = np.random.randn(16, 6).astype(np.float32) * 1.5
        peak = float(np.abs(w).max())
        scale = peak / 127.0
        wq = np.round(w / scale).astype(np.int8)
        expected = a @ w
        got = int8_matmul(a, wq, scale)
        err = np.abs(got - expected).max() / max(1.0, np.abs(expected).max())
        assert err < 0.05

    def test_int32_accumulator_agrees(self):
        a = (np.random.randn(4, 8) * 1.0).astype(np.float32)
        w = (np.random.randn(8, 5) * 1.0).astype(np.float32)
        peak = float(np.abs(w).max())
        scale = peak / 127.0
        wq = np.round(w / scale).astype(np.int8)
        r1 = int8_matmul(a, wq, scale, accumulator="fp32")
        r2 = int8_matmul(a, wq, scale, accumulator="int32")
        assert np.allclose(r1, r2, atol=1e-3)

    def test_zero_weights_safe(self):
        w = np.zeros((4, 4), dtype=np.int8)
        a = np.random.randn(3, 4).astype(np.float32)
        assert np.allclose(int8_matmul(a, w, 1.0), np.zeros((3, 4)))


class TestWeightQuantization:
    def test_quantizes_matmul_rhs_only(self):
        compiled, _ = _mlp_graph()
        q = quantize_graph_weights(compiled, mode="int8")
        assert len(q) == 2
        for (wq, scale) in q.values():
            assert wq.dtype == np.int8
            assert scale > 0

    def test_dequant_reproduces_weights(self):
        compiled, _ = _mlp_graph()
        q = quantize_graph_weights(compiled, mode="int8")
        weights = [n for n in compiled.nodes if n.op == "matmul"]
        for node in weights:
            wq, scale = q[id(node)]
            orig = np.asarray(node.inputs[1].params["value"])
            rel_err = np.abs(wq.astype(np.float32) * scale - orig).max() / \
                max(1.0, np.abs(orig).max())
            assert rel_err < 0.05

    def test_unsupported_mode_raises(self):
        compiled, _ = _mlp_graph()
        with pytest.raises(ValueError):
            quantize_graph_weights(compiled, mode="fp16")


class TestEdgeRuntime:
    def test_fp32_parity_with_compiled_graph(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled, EdgeConfig(quant_mode="none"))
        xv = np.random.randn(4, 16)
        expected = compiled.forward({"x": xv})
        assert np.allclose(rt({"x": xv}), expected)
        assert np.allclose(rt.predict({"x": xv}), expected)

    def test_int8_inference_close_to_fp32(self):
        compiled, x = _mlp_graph()
        xv = np.random.randn(16, 16)
        fp32 = compiled.forward({"x": xv})
        rt = EdgeRuntime(compiled, EdgeConfig(quant_mode="int8"))
        got = rt({"x": xv})
        denom = max(1.0, np.abs(fp32).max())
        assert np.abs(got - fp32).max() / denom < 0.1

    def test_tiling_path_matches(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled)
        xv = np.random.randn(64, 16)
        full = rt({"x": xv})
        tiled = rt.forward({"x": xv}, tile_size=13)
        assert np.allclose(full, tiled)

    def test_latency_positive(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled)
        xv = np.random.randn(4, 16)
        ms = rt.latency_ms({"x": xv}, iterations=5)
        assert ms > 0
        assert rt.throughput({"x": xv}, iterations=5) > 0

    def test_summary_fields(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled, EdgeConfig(quant_mode="int8"))
        s = rt.summary()
        for key in ("device", "isa", "cores", "backend", "quant_mode",
                    "quantized_matmul_layers", "estimated_flops"):
            assert key in s
        assert s["quantized_matmul_layers"] == 2
        assert s["estimated_flops"] > 0

    def test_unknown_backend_raises(self):
        compiled, x = _mlp_graph()
        with pytest.raises(ValueError):
            EdgeRuntime(compiled, EdgeConfig(backend="tpu"))

    def test_custom_backend_registration(self):
        class DoublingBackend(EdgeBackend):
            name = "double"

            def supports(self, node):
                return node.op == "fused"

            def run(self, node, inputs, pool, **kw):
                return node.fn(*inputs) * 2.0

        compiled, x = _mlp_graph()
        xv = np.random.randn(4, 16)
        expected = compiled.forward({"x": xv})
        register_backend("double", lambda **kw: DoublingBackend())
        rt = EdgeRuntime(compiled, EdgeConfig(backend="double"))
        got = rt({"x": xv})
        assert not np.allclose(got, expected)
        assert np.allclose(got / 2.0, expected)

    def test_buffer_pool_disabled(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled, EdgeConfig(buffer_pool=False))
        assert len(rt.pool) == 0
        xv = np.random.randn(4, 16)
        assert rt({"x": xv}).shape == (4, 8)

    def test_pool_reused_across_calls(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled, EdgeConfig(quant_mode="int8"))
        xv = np.random.randn(4, 16)
        for _ in range(3):
            rt({"x": xv})
        assert len(rt.pool) >= 1

    def test_benchmark_bundle(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled, EdgeConfig(quant_mode="int8"))
        xv = np.random.randn(4, 16)
        res = benchmark(rt, {"x": xv}, iterations=5)
        assert "latency_ms" in res
        assert "throughput_ops" in res
        assert "flops" in res
        assert res["latency_ms"] > 0

    def test_flops_grows_with_dimension(self):
        compiled_small, _ = _mlp_graph(16, 16, 8)
        compiled_big, _ = _mlp_graph(16, 64, 8)
        small = EdgeRuntime(compiled_small).flops()
        big = EdgeRuntime(compiled_big).flops()
        assert big > small

    def test_close_releases_pool(self):
        compiled, x = _mlp_graph()
        rt = EdgeRuntime(compiled)
        xv = np.random.randn(4, 16)
        rt({"x": xv})
        assert len(rt.pool) > 0
        rt.close()
        assert len(rt.pool) == 0

    def test_scalar_const_graph(self):
        a = GraphNode.input_("a")
        b = GraphNode.input_("b")
        compiled = GraphCompiler().compile((a + b).sigmoid())
        rt = EdgeRuntime(compiled)
        x = np.random.randn(6)
        y = np.random.randn(6)
        expected = 1.0 / (1.0 + np.exp(-(x + y)))
        assert np.allclose(rt({"a": x, "b": y}), expected)

    def test_int4_mode_not_supported(self):
        compiled, x = _mlp_graph()
        with pytest.raises(ValueError):
            EdgeRuntime(compiled, EdgeConfig(quant_mode="int4"))
