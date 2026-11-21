"""ONNX binary model serializer (numpy-only).

Encodes :mod:`onnx.model` data classes into canonical binary ``.onnx`` (raw
protobuf ModelProto). Field numbers follow the ONNX IR and are interoperable
with the C exporter in ``fs/format/onnx_format.c`` and standard runtimes.

Supports dynamic axes (symbolic dimensions), metadata props, external data
references, and attribute types incl. tensor/graph values.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .model import (
    Attribute,
    AttrType,
    Graph,
    MetadataProp,
    Model,
    Node,
    OpsetImport,
    Tensor,
    ValueInfo,
    DTYPE_TO_ONNX,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
)
from .wire import (
    _pb_bytes_field,
    _pb_float_field,
    _pb_string_field,
    _pb_sub,
    _pb_varint,
    _pb_varint_field,
)

__all__ = ["serialize_model", "save_model", "to_bytes", "serialize_graph"]


def _encode_tensor_data(arr: np.ndarray, dtype_name: str) -> bytes:
    """Encode a numpy array as TensorProto typed fields (dims + raw_data)."""
    dims_packed = b"".join(_pb_varint(int(d)) for d in arr.shape)
    parts = [_pb_bytes_field(1, dims_packed)]
    parts.append(_pb_varint_field(2, DTYPE_TO_ONNX.get(dtype_name, 1)))

    raw = np.ascontiguousarray(arr)
    if raw.dtype.byteorder == ">":
        raw = raw.astype(raw.dtype.newbyteorder("<"))
    raw_bytes = raw.tobytes()

    if dtype_name == "string":
        parts.append(_pb_string_field(8, ""))  # name placeholder set below
        # strings use repeated length-delimited fields under string_data (6)
        payload = b"".join(_pb_varint(len(s)) + s for s in arr.reshape(-1))
        parts.append(_pb_sub(6, payload))
    else:
        parts.append(_pb_sub(9, raw_bytes))
    return b"".join(parts)


def _encode_tensor(tensor: Tensor) -> bytes:
    """Encode a TensorProto: dims, data_type, name, raw_data (or external)."""
    dims_packed = b"".join(_pb_varint(int(d)) for d in tensor.shape)
    parts = [_pb_bytes_field(1, dims_packed)]
    parts.append(_pb_varint_field(2, DTYPE_TO_ONNX.get(tensor.dtype, 1)))

    if tensor.data is not None:
        raw = np.ascontiguousarray(tensor.data)
        if raw.dtype.byteorder == ">":
            raw = raw.astype(raw.dtype.newbyteorder("<"))
        parts.append(_pb_sub(9, raw.tobytes()))
    elif tensor.external_data:
        parts.append(_pb_varint_field(14, 1))  # data_location = EXTERNAL
        for key, value in tensor.external_data.items():
            entry = _pb_string_field(1, key) + _pb_string_field(2, value)
            parts.append(_pb_sub(13, entry))
    if tensor.name:
        parts.append(_pb_string_field(8, tensor.name))
    return b"".join(parts)


def _encode_dim(value: Any) -> bytes:
    """Encode one Dimension (dim_value=1 or dim_param=2)."""
    if isinstance(value, str):
        return _pb_string_field(2, value)
    if value is None:
        return b""
    return _pb_varint_field(1, int(value))


def _encode_value_info(info: ValueInfo) -> bytes:
    dims = [_pb_sub(1, _encode_dim(d)) for d in info.shape]
    tensor = _pb_varint_field(
        1, DTYPE_TO_ONNX.get(info.dtype, 1)
    ) + _pb_sub(2, b"".join(dims))
    type_proto = _pb_sub(1, tensor)
    return _pb_string_field(1, info.name) + _pb_sub(2, type_proto)


def _encode_attr(attr: Attribute) -> bytes:
    name_field = _pb_string_field(1, attr.name)
    atype = attr.attr_type or AttrType.UNDEFINED
    value = attr.value

    if atype == AttrType.FLOAT or (isinstance(value, float) and atype == AttrType.UNDEFINED):
        return (
            name_field
            + _pb_varint_field(20, AttrType.FLOAT)
            + _pb_float_field(2, float(value))
        )
    if atype == AttrType.INT or (isinstance(value, bool) and atype == AttrType.UNDEFINED):
        return (
            name_field
            + _pb_varint_field(20, AttrType.INT)
            + _pb_varint_field(3, 1 if value else 0)
        )
    if isinstance(value, bool):
        return (
            name_field
            + _pb_varint_field(20, AttrType.INT)
            + _pb_varint_field(3, 1 if value else 0)
        )
    if isinstance(value, int):
        return (
            name_field
            + _pb_varint_field(20, AttrType.INT)
            + _pb_varint_field(3, int(value))
        )
    if isinstance(value, float):
        return (
            name_field
            + _pb_varint_field(20, AttrType.FLOAT)
            + _pb_float_field(2, float(value))
        )
    if isinstance(value, bytes):
        return name_field + _pb_varint_field(20, AttrType.STRING) + _pb_bytes_field(4, value)
    if isinstance(value, str):
        return (
            name_field
            + _pb_varint_field(20, AttrType.STRING)
            + _pb_string_field(4, value)
        )
    if isinstance(value, np.ndarray) or isinstance(value, Tensor):
        tensor = value if isinstance(value, Tensor) else Tensor("", "float32", list(value.shape), value)
        return (
            name_field
            + _pb_varint_field(20, AttrType.TENSOR)
            + _pb_sub(5, _encode_tensor(tensor))
        )
    if isinstance(value, Graph):
        return (
            name_field
            + _pb_varint_field(20, AttrType.GRAPH)
            + _pb_sub(6, serialize_graph(value))
        )
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, bool) for v in value):
            ints = b"".join(_pb_varint(1 if v else 0) for v in value)
            return (
                name_field
                + _pb_varint_field(20, AttrType.INTS)
                + _pb_bytes_field(8, ints)
            )
        if value and all(isinstance(v, int) for v in value):
            ints = b"".join(_pb_varint(int(v)) for v in value)
            return (
                name_field
                + _pb_varint_field(20, AttrType.INTS)
                + _pb_bytes_field(8, ints)
            )
        if value and all(isinstance(v, float) for v in value):
            floats = b"".join(np.float32(v).tobytes() for v in value)
            return (
                name_field
                + _pb_varint_field(20, AttrType.FLOATS)
                + _pb_bytes_field(7, floats)
            )
        if value and all(isinstance(v, str) for v in value):
            strings = b"".join(_pb_varint(len(s.encode())) + s.encode() for s in value)
            return (
                name_field
                + _pb_varint_field(20, AttrType.STRINGS)
                + _pb_bytes_field(9, strings)
            )
        if value and all(isinstance(v, bytes) for v in value):
            strings = b"".join(_pb_varint(len(v)) + v for v in value)
            return (
                name_field
                + _pb_varint_field(20, AttrType.STRINGS)
                + _pb_bytes_field(9, strings)
            )
    raise ValueError(f"Unsupported ONNX attribute value for '{attr.name}': {value!r}")


def _encode_node(node: Node) -> bytes:
    parts: List[bytes] = []
    for inp in node.inputs:
        parts.append(_pb_string_field(1, inp))
    for out in node.outputs:
        parts.append(_pb_string_field(2, out))
    if node.name:
        parts.append(_pb_string_field(3, node.name))
    parts.append(_pb_string_field(4, node.op_type))
    for attr_name, attr in node.attributes.items():
        if isinstance(attr, Attribute):
            parts.append(_pb_sub(5, _encode_attr(attr)))
        else:
            parts.append(_pb_sub(5, _encode_attr(Attribute(attr_name, attr))))
    if node.domain:
        parts.append(_pb_string_field(7, node.domain))
    return b"".join(parts)


def serialize_graph(graph: Graph) -> bytes:
    """Encode a GraphProto (node=1, name=2, initializer=5, doc=10, input=11,
    output=12, value_info=13)."""
    parts: List[bytes] = []
    for node in graph.nodes:
        parts.append(_pb_sub(1, _encode_node(node)))
    parts.append(_pb_string_field(2, graph.name))
    for init in graph.initializers:
        parts.append(_pb_sub(5, _encode_tensor(init)))
    if graph.doc_string:
        parts.append(_pb_string_field(10, graph.doc_string))
    for inp in graph.inputs:
        parts.append(_pb_sub(11, _encode_value_info(inp)))
    for out in graph.outputs:
        parts.append(_pb_sub(12, _encode_value_info(out)))
    for info in graph.value_info:
        parts.append(_pb_sub(13, _encode_value_info(info)))
    return b"".join(parts)


def serialize_model(model: Model) -> bytes:
    """Serialize a :class:`Model` to canonical binary ``.onnx`` (raw protobuf)."""
    parts: List[bytes] = [
        _pb_varint_field(1, model.ir_version),
        _pb_string_field(2, model.producer_name),
        _pb_string_field(3, model.producer_version),
        _pb_varint_field(5, model.model_version),
        _pb_sub(7, serialize_graph(model.graph)),
    ]
    if model.domain:
        parts.append(_pb_string_field(4, model.domain))
    if model.doc_string:
        parts.append(_pb_string_field(6, model.doc_string))
    opsets = model.opset_imports or [OpsetImport("", ONNX_OPSET_VERSION)]
    for opset in opsets:
        opset_proto = _pb_string_field(
            1, opset.domain or ""
        ) + _pb_varint_field(2, int(opset.version or ONNX_OPSET_VERSION))
        parts.append(_pb_sub(8, opset_proto))
    for prop in model.metadata_props:
        entry = _pb_string_field(1, prop.key) + _pb_string_field(2, prop.value)
        parts.append(_pb_sub(14, entry))
    return b"".join(parts)


def save_model(model: Model, path: str) -> str:
    """Serialize and write a model to an ``.onnx`` file path."""
    with open(path, "wb") as f:
        f.write(serialize_model(model))
    return path


to_bytes = serialize_model
