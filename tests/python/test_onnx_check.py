"""Tests for ONNX shape inference and schema checks (onnx_check)."""

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.onnx_export import (
    OnnxExporter,
    OnnxModel,
    OnnxGraph,
    OnnxNode,
    OnnxTensor,
)
from SneppX_ALG.interface_bindings.onnx_check import (
    infer_shapes,
    onnx_check,
    broadcast_shape,
    OnnxShapeError,
)


def test_broadcast_shape():
    assert broadcast_shape([2, 3, 4], [4]) == [2, 3, 4]
    assert broadcast_shape([2, 1], [1, 4]) == [2, 4]
    assert broadcast_shape([2, 3], [1, 3]) == [2, 3]
    assert broadcast_shape(["batch", 3, 4], [1, 1, 4]) == ["batch", 3, 4]
    with pytest.raises(OnnxShapeError):
        broadcast_shape([2, 3], [4])


def test_infer_conv():
    exporter = OnnxExporter()
    exporter.add_input("x", [1, 3, 224, 224], "float32")
    exporter.add_constant("w", np.random.randn(64, 3, 3, 3).astype(np.float32))
    conv_out = exporter.add_conv("x", "w", strides=(2, 2), pads=(1, 1, 1, 1))
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[conv_out].shape == [1, 64, 112, 112]
    assert inferred[conv_out].dtype == "float32"


def test_infer_conv_symbolic_batch():
    exporter = OnnxExporter()
    exporter.add_input("x", ["batch", 3, 224, 224], "float32")
    exporter.add_constant("w", np.random.randn(64, 3, 3, 3).astype(np.float32))
    conv_out = exporter.add_conv("x", "w", strides=(2, 2), pads=(1, 1, 1, 1))
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[conv_out].shape == ["batch", 64, 112, 112]


def test_infer_gemm():
    exporter = OnnxExporter()
    exporter.add_input("x", [2, 3], "float32")
    exporter.add_constant("w", np.random.randn(2, 3).astype(np.float32))
    gemm_out = exporter.add_gemm("x", "w", trans_b=1)
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[gemm_out].shape == [2, 2]


def test_infer_matmul():
    exporter = OnnxExporter()
    exporter.add_input("a", [2, 3, 4], "float32")
    exporter.add_input("b", [2, 4, 5], "float32")
    mm = exporter.add_matmul("a", "b")
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[mm].shape == [2, 3, 5]


def test_infer_matmul_mismatch():
    exporter = OnnxExporter()
    exporter.add_input("a", [2, 3], "float32")
    exporter.add_input("b", [2, 4], "float32")
    exporter.add_matmul("a", "b")
    model = exporter.build_model()
    with pytest.raises(OnnxShapeError):
        infer_shapes(model)


def test_infer_broadcast_add():
    exporter = OnnxExporter()
    exporter.add_input("a", [2, 3, 4], "float32")
    exporter.add_input("b", [4], "float32")
    add_out = exporter.add_add("a", "b")
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[add_out].shape == [2, 3, 4]


def test_infer_reshape():
    exporter = OnnxExporter()
    exporter.add_input("x", [2, 3, 4], "float32")
    reshape_out = exporter.add_reshape("x", [2, -1, 4])
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[reshape_out].shape == [2, 3, 4]


def test_infer_transpose():
    exporter = OnnxExporter()
    exporter.add_input("x", [1, 3, 224, 224], "float32")
    t = exporter.add_transpose("x", [0, 2, 1, 3])
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[t].shape == [1, 224, 3, 224]


def test_infer_concat():
    exporter = OnnxExporter()
    exporter.add_input("a", [1, 3, 224, 224], "float32")
    exporter.add_input("b", [1, 64, 224, 224], "float32")
    cat = exporter.add_concat(["a", "b"], axis=1)
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[cat].shape == [1, 67, 224, 224]


def test_infer_pool():
    exporter = OnnxExporter()
    exporter.add_input("x", [1, 3, 224, 224], "float32")
    pool = exporter.add_max_pool("x", (3, 3), strides=(2, 2))
    model = exporter.build_model()

    inferred = infer_shapes(model)
    # (224 - 3) // 2 + 1 = 111
    assert inferred[pool].shape == [1, 3, 111, 111]


def test_infer_reduce_mean_keepdims():
    exporter = OnnxExporter()
    exporter.add_input("x", [1, 3, 224, 224], "float32")
    r = exporter.add_reduce_mean("x", [1, 2], keepdims=1)
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[r].shape == [1, 1, 1, 224]


def test_infer_gather():
    exporter = OnnxExporter()
    exporter.add_input("x", [3, 4], "float32")
    exporter.add_constant("idx", np.array([2], dtype=np.int64), "int64")
    g = exporter.add_gather("x", "idx", axis=0)
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[g].shape == [1, 4]


def test_infer_split():
    exporter = OnnxExporter()
    exporter.add_input("x", [6, 4], "float32")
    outs = exporter.add_split("x", [2, 4], axis=0)
    model = exporter.build_model()

    inferred = infer_shapes(model)
    assert inferred[outs[0]].shape == [2, 4]
    assert inferred[outs[1]].shape == [4, 4]


def test_onnx_check_valid():
    exporter = OnnxExporter()
    exporter.add_input("x", ["batch", 3, 224, 224], "float32")
    exporter.add_constant("w", np.random.randn(64, 3, 3, 3).astype(np.float32))
    conv_out = exporter.add_conv("x", "w", strides=(2, 2), pads=(1, 1, 1, 1))
    exporter.add_relu(conv_out)
    exporter.add_output(conv_out, ["batch", 64, 112, 112])
    model = exporter.build_model()

    ok, errors = onnx_check(model)
    assert ok, errors


def test_onnx_check_declared_mismatch():
    exporter = OnnxExporter()
    exporter.add_input("x", [1, 3, 224, 224], "float32")
    exporter.add_constant("w", np.random.randn(64, 3, 3, 3).astype(np.float32))
    conv_out = exporter.add_conv("x", "w", strides=(2, 2), pads=(1, 1, 1, 1))
    exporter.add_output(conv_out, [1, 10, 10, 10])
    model = exporter.build_model()

    ok, errors = onnx_check(model)
    assert not ok
    assert any("declared" in e for e in errors)


def test_onnx_check_arity():
    graph = OnnxGraph("arity")
    graph.add_input(OnnxTensor("x", "float32", [2, 3]))
    graph.add_output(OnnxTensor("y", "float32", [2, 2]))
    graph.add_node(OnnxNode("Gemm", ["x"], ["y"], attributes={"transB": 1}))
    ok, errors = onnx_check(OnnxModel(graph))
    assert not ok
    assert any("inputs" in e for e in errors)


def test_onnx_check_unknown_op():
    graph = OnnxGraph("unknown")
    graph.add_input(OnnxTensor("x", "float32", [2, 3]))
    graph.add_output(OnnxTensor("y", "float32", [2, 2]))
    graph.add_node(OnnxNode("TotallyMadeUpOp", ["x"], ["y"]))
    ok, errors = onnx_check(OnnxModel(graph))
    assert not ok
    assert any("not implemented" in e for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
