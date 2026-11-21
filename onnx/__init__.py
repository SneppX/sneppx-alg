"""SNEPPX-Alg ONNX import/export toolkit (numpy-only, standalone).

A full ONNX module: canonical wire-format parser/serializer, shape inference
and model checker, graph-optimization passes, QDQ weight quantization,
external-data support, a pure-numpy runtime executor, an optional onnxruntime
adapter, and a ``sneppx-onnx`` command-line interface.

Only ``numpy`` is required; the ``onnx``/``onnxruntime`` PyPI packages are
optional and detected at runtime.
"""

__version__ = "1.5.2"

from .model import (
    Attribute,
    AttrType,
    Dimension,
    Graph,
    MetadataProp,
    Model,
    Node,
    OpsetImport,
    Tensor,
    ValueInfo,
    DTYPE_TO_ONNX,
    ONNX_TO_DTYPE,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
)
from .parser import load_model, parse_model, from_bytes, OnnxParseError
from .serializer import save_model, serialize_model, to_bytes
from .inference import (
    infer_shapes,
    check_model,
    broadcast_shape,
    OnnxShapeError,
)
from .optimizer import optimize, Optimizer, constant_fold, eliminate_dead_code
from .qdq import quantize_model, Quantizer, qdq_round_trip
from .external_data import (
    save_external_data,
    extract_external_data,
    load_external_data,
)
from .runtime.numpy_executor import Session, execute, UnsupportedOpError
from .runtime.ort_adapter import OnnxRuntimeSession, has_onnxruntime
from .exporter import build_graph, export, save, to_sneppx_graph

__all__ = [
    # model
    "Attribute", "AttrType", "Dimension", "Graph", "MetadataProp", "Model",
    "Node", "OpsetImport", "Tensor", "ValueInfo",
    "DTYPE_TO_ONNX", "ONNX_TO_DTYPE", "ONNX_IR_VERSION", "ONNX_OPSET_VERSION",
    # parse / serialize
    "load_model", "parse_model", "from_bytes", "save_model",
    "serialize_model", "to_bytes", "OnnxParseError",
    # inference / check
    "infer_shapes", "check_model", "broadcast_shape", "OnnxShapeError",
    # optimize
    "optimize", "Optimizer", "constant_fold", "eliminate_dead_code",
    # qdq
    "quantize_model", "Quantizer", "qdq_round_trip",
    # external data
    "save_external_data", "extract_external_data", "load_external_data",
    # runtime
    "Session", "execute", "UnsupportedOpError",
    "OnnxRuntimeSession", "has_onnxruntime",
    # exporter
    "build_graph", "export", "save", "to_sneppx_graph",
]
