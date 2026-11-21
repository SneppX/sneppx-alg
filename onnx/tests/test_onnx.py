"""Tests for the standalone ONNX toolkit (onnx/).

Run from the repo root::

    python -m pytest onnx/tests/test_onnx.py -q

These tests are numpy-only and safe to run on this machine.
"""

import os
import tempfile

import numpy as np
import pytest

import onnx


def _mlp_model():
    W1 = np.random.randn(4, 8).astype(np.float32)
    b1 = np.random.randn(8).astype(np.float32)
    W2 = np.random.randn(8, 2).astype(np.float32)
    b2 = np.random.randn(2).astype(np.float32)
    graph = onnx.build_graph(
        name="mlp",
        inputs=[onnx.ValueInfo("x", "float32", [1, 4])],
        outputs=[onnx.ValueInfo("y", "float32", [1, 2])],
        initializers={"W1": W1, "b1": b1, "W2": W2, "b2": b2},
        nodes=[
            onnx.Node("MatMul", ["x", "W1"], ["mm1"]),
            onnx.Node("Add", ["mm1", "b1"], ["a1"]),
            onnx.Node("Relu", ["a1"], ["r1"]),
            onnx.Node("MatMul", ["r1", "W2"], ["mm2"]),
            onnx.Node("Add", ["mm2", "b2"], ["y"]),
        ],
    )
    return onnx.Model(graph, producer_name="SNEPPX", model_version=1)


class TestSerializeParse:
    def test_roundtrip(self):
        model = _mlp_model()
        data = onnx.serialize_model(model)
        parsed = onnx.parse_model(data)
        assert parsed.opset_version == 18
        assert len(parsed.graph.nodes) == 5
        assert [n.op_type for n in parsed.graph.nodes] == [
            "MatMul", "Add", "Relu", "MatMul", "Add"
        ]

    def test_save_load_file(self):
        model = _mlp_model()
        path = os.path.join(tempfile.gettempdir(), "t_mlp.onnx")
        onnx.save_model(model, path)
        loaded = onnx.load_model(path)
        assert len(loaded.graph.nodes) == 5
        assert loaded.graph.name == "mlp"
        os.remove(path)

    def test_dynamic_axis(self):
        graph = onnx.build_graph(
            name="dyn",
            inputs=[onnx.ValueInfo("x", "float32", ["batch", 4])],
            outputs=[onnx.ValueInfo("y", "float32", ["batch", 8])],
            initializers={"W": np.ones((4, 8), np.float32)},
            nodes=[onnx.Node("MatMul", ["x", "W"], ["y"])],
        )
        data = onnx.serialize_model(onnx.Model(graph))
        parsed = onnx.parse_model(data)
        assert parsed.graph.inputs[0].shape == ["batch", 4]

    def test_metadata_props(self):
        model = _mlp_model()
        model.metadata_props = [onnx.MetadataProp("author", "sneppx")]
        parsed = onnx.parse_model(onnx.serialize_model(model))
        assert parsed.metadata_props[0].key == "author"

    def test_onnx_prefixed_form(self):
        model = _mlp_model()
        data = onnx.serialize_model(model)
        wrapped = b"ONNX" + len(data).to_bytes(4, "little") + data
        parsed = onnx.parse_model(wrapped)
        assert len(parsed.graph.nodes) == 5


class TestCheckInference:
    def test_check_ok(self):
        ok, errors = onnx.check_model(_mlp_model())
        assert ok, errors

    def test_check_missing_input(self):
        model = _mlp_model()
        model.graph.nodes[1].inputs = ["missing", "b1"]
        ok, errors = onnx.check_model(model)
        assert not ok
        assert any("missing" in e for e in errors)

    def test_infer_shapes(self):
        shapes = onnx.infer_shapes(_mlp_model())
        assert shapes["mm2"][0] == [1, 2]
        assert shapes["mm2"][1] == "float32"
        assert shapes["mm1"][0] == [1, 8]

    def test_infer_batch_symbolic(self):
        graph = onnx.build_graph(
            name="sym",
            inputs=[onnx.ValueInfo("x", "float32", ["N", 4])],
            outputs=[onnx.ValueInfo("y", "float32", ["N", 8])],
            initializers={"W": np.ones((4, 8), np.float32)},
            nodes=[onnx.Node("MatMul", ["x", "W"], ["y"])],
        )
        shapes = onnx.infer_shapes(onnx.Model(graph))
        assert shapes["y"][0] == ["N", 8]

    def test_broadcast_shape(self):
        assert onnx.broadcast_shape([3, 1], [1, 4]) == [3, 4]
        with pytest.raises(onnx.OnnxShapeError):
            onnx.broadcast_shape([3], [4])


class TestOptimizer:
    def test_constant_fold(self):
        C = np.random.randn(8).astype(np.float32)
        graph = onnx.build_graph(
            name="fold",
            inputs=[onnx.ValueInfo("x", "float32", [1, 4])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 8])],
            initializers={"W": np.random.randn(4, 8).astype(np.float32), "C": C},
            nodes=[
                onnx.Node("MatMul", ["x", "W"], ["mm"]),
                onnx.Node("Relu", ["C"], ["c_relu"]),
                onnx.Node("Sigmoid", ["c_relu"], ["c_sig"]),
                onnx.Node("Add", ["mm", "c_sig"], ["y"]),
                onnx.Node("Neg", ["x"], ["dead_out"]),
            ],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        opt = onnx.optimize(model)
        assert len(opt.graph.nodes) == 2  # MatMul + Add (2 folded, 1 DCE)

        ok, errors = onnx.check_model(opt)
        assert ok, errors

        s1, s2 = onnx.Session(model), onnx.Session(opt)
        x = np.random.randn(1, 4).astype(np.float32)
        assert np.allclose(s1.run({"x": x})[0], s2.run({"x": x})[0], atol=1e-5)

    def test_dce(self):
        graph = onnx.build_graph(
            name="dce",
            inputs=[onnx.ValueInfo("x", "float32", [1, 4])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 8])],
            initializers={"W": np.ones((4, 8), np.float32)},
            nodes=[
                onnx.Node("MatMul", ["x", "W"], ["y"]),
                onnx.Node("Neg", ["x"], ["dead"]),
            ],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        opt = onnx.optimize(model, passes=["dead_code_elimination"])
        assert len(opt.graph.nodes) == 1

    def test_unknown_pass_raises(self):
        with pytest.raises(ValueError):
            onnx.optimize(_mlp_model(), passes=["nope"])


class TestQDQ:
    def test_qdq_insertion(self):
        W = np.random.randn(4, 8).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        graph = onnx.build_graph(
            name="q",
            inputs=[onnx.ValueInfo("x", "float32", [1, 4])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 8])],
            initializers={"W": W, "bias": bias},
            nodes=[
                onnx.Node("MatMul", ["x", "W"], ["mm"]),
                onnx.Node("Add", ["mm", "bias"], ["y"]),
            ],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        q = onnx.quantize_model(model, per_channel=True)
        ops = {n.op_type for n in q.graph.nodes}
        assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
        ok, errors = onnx.check_model(q)
        assert ok, errors

    def test_qdq_round_trip(self):
        w = np.random.randn(16, 16).astype(np.float32)
        s = onnx.qdq.symmetric_scale(w, 8)
        rt = onnx.qdq_round_trip(w, s, np.array(0, dtype=np.int8), 8)
        rel = float(np.abs(rt - w).mean() / np.abs(w).mean())
        assert rel < 0.02

    def test_bias_skipped(self):
        W = np.random.randn(4, 8).astype(np.float32)
        graph = onnx.build_graph(
            name="q",
            inputs=[onnx.ValueInfo("x", "float32", [1, 4])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 8])],
            initializers={"W": W, "bias": np.zeros(8, np.float32)},
            nodes=[
                onnx.Node("MatMul", ["x", "W"], ["mm"]),
                onnx.Node("Add", ["mm", "bias"], ["y"]),
            ],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        q = onnx.quantize_model(model)
        bias_init = next(i for i in q.graph.initializers if i.name == "bias")
        assert bias_init.dtype == "float32"


class TestExternalData:
    def test_save_load(self):
        big = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64).astype(np.float32)
        graph = onnx.build_graph(
            name="ext",
            inputs=[onnx.ValueInfo("x", "float32", [1, 64])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 64])],
            initializers={"W": big, "b": b},
            nodes=[
                onnx.Node("MatMul", ["x", "W"], ["mm"]),
                onnx.Node("Add", ["mm", "b"], ["y"]),
            ],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        d = os.path.join(tempfile.gettempdir(), "onnx_ext_test")
        p = os.path.join(tempfile.gettempdir(), "onnx_ext_test.onnx")
        onnx.save_external_data(model, d, size_threshold=64, location="w.bin")
        onnx.save_model(model, p)

        loaded = onnx.external_data.load_external_data(p, base_dir=d)
        w = next(i for i in loaded.graph.initializers if i.name == "W")
        assert np.allclose(w.data, big)
        os.remove(p)


class TestRuntime:
    def test_executor_matches_numpy(self):
        model = _mlp_model()
        sess = onnx.Session(model)
        x = np.random.randn(1, 4).astype(np.float32)
        out = sess.run({"x": x})[0]
        W1 = next(i.data for i in model.graph.initializers if i.name == "W1")
        b1 = next(i.data for i in model.graph.initializers if i.name == "b1")
        W2 = next(i.data for i in model.graph.initializers if i.name == "W2")
        b2 = next(i.data for i in model.graph.initializers if i.name == "b2")
        expected = np.maximum(x @ W1 + b1, 0) @ W2 + b2
        assert np.allclose(out, expected, atol=1e-4)

    def test_conv(self):
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        w = np.random.randn(2, 1, 3, 3).astype(np.float32)
        graph = onnx.build_graph(
            name="conv",
            inputs=[onnx.ValueInfo("x", "float32", [1, 1, 4, 4])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 2, 2, 2])],
            initializers={"W": w},
            nodes=[onnx.Node("Conv", ["x", "W"], ["y"], attributes={"pads": [0, 0, 0, 0]})],
        )
        model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
        sess = onnx.Session(model)
        out = sess.run({"x": x})[0]
        ref = np.zeros((1, 2, 2, 2), np.float32)
        for ci in range(2):
            for i in range(2):
                for j in range(2):
                    ref[0, ci, i, j] = np.sum(x[0, 0, i : i + 3, j : j + 3] * w[ci, 0])
        assert np.allclose(out, ref, atol=1e-4)

    def test_ort_adapter(self):
        model = _mlp_model()
        from onnx.runtime.ort_adapter import has_onnxruntime

        sess = onnx.OnnxRuntimeSession(model)
        x = np.random.randn(1, 4).astype(np.float32)
        out = sess.run({"x": x})[0]
        assert out.shape == (1, 2)
        assert has_onnxruntime() in (True, False)


class TestCLI:
    def test_check_subcommand(self):
        from onnx.cli import main

        path = os.path.join(tempfile.gettempdir(), "t_cli.onnx")
        onnx.save_model(_mlp_model(), path)
        assert main(["check", path]) == 0
        os.remove(path)

    def test_info_subcommand(self):
        from onnx.cli import main

        path = os.path.join(tempfile.gettempdir(), "t_cli2.onnx")
        onnx.save_model(_mlp_model(), path)
        assert main(["info", path]) == 0
        os.remove(path)

    def test_optimize_subcommand(self):
        from onnx.cli import main

        src = os.path.join(tempfile.gettempdir(), "t_cli3.onnx")
        dst = os.path.join(tempfile.gettempdir(), "t_cli3_opt.onnx")
        onnx.save_model(_mlp_model(), src)
        assert main(["optimize", src, "-o", dst]) == 0
        assert os.path.exists(dst)
        os.remove(src)
        os.remove(dst)


class TestExporter:
    def test_build_graph_initializer_dtype(self):
        g = onnx.build_graph(
            name="g",
            inputs=[onnx.ValueInfo("x", "float32", [1, 2])],
            outputs=[onnx.ValueInfo("y", "float32", [1, 2])],
            initializers={"W": np.ones((2, 2), np.float32)},
            nodes=[onnx.Node("MatMul", ["x", "W"], ["y"])],
        )
        assert g.initializers[0].dtype == "float32"
        assert g.initializers[0].data.dtype == np.float32

    def test_to_sneppx_graph(self):
        model = _mlp_model()
        d = onnx.to_sneppx_graph(model)
        assert d["outputs"] == ["y"]
        assert "W1" in d["initializers"]
        assert len(d["nodes"]) == 5
