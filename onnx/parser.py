"""ONNX binary model parser (numpy-only).

Parses canonical binary ``.onnx`` (raw protobuf ModelProto) into the
:mod:`onnx.model` data classes. Handles all standard AttributeProto value
types, typed TensorProto fields (float_data/int32_data/.../raw_data), external
data references, opset-import blocks, metadata props and symbolic dimensions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    ONNX_TO_DTYPE,
)
from .wire import _iter_fields, _pb_read_varint, _pb_signed

__all__ = [
    "parse_model",
    "load_model",
    "from_bytes",
    "parse_graph",
    "OnnxParseError",
]


class OnnxParseError(ValueError):
    """Raised when binary ONNX parsing fails."""


def _read_attr(data: bytes) -> Tuple[str, Any, int]:
    name = ""
    atype = AttrType.UNDEFINED
    value: Any = None
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 2 and wire == 5:
            value = np.frombuffer(val, dtype="<f4")[0].item()
            atype = AttrType.FLOAT
        elif field == 3 and wire == 0:
            value = _pb_signed(int(val))
            atype = AttrType.INT
        elif field == 4 and wire == 2:
            value = val
            atype = AttrType.STRING
        elif field == 5 and wire == 2:
            value = _parse_tensor_proto(val, load_external=False)
            atype = AttrType.TENSOR
        elif field == 6 and wire == 2:
            value = _parse_graph(val)
            atype = AttrType.GRAPH
        elif field == 7 and wire == 2:
            vals = []
            for i in range(0, len(val) - 3, 4):
                vals.append(np.frombuffer(val[i : i + 4], dtype="<f4")[0].item())
            value = vals
            atype = AttrType.FLOATS
        elif field == 8 and wire == 2:
            pos = 0
            vals = []
            while pos < len(val):
                v, pos = _pb_read_varint(val, pos)
                vals.append(_pb_signed(int(v)))
            value = vals
            atype = AttrType.INTS
        elif field == 9 and wire == 2:
            pos = 0
            vals = []
            while pos < len(val):
                ln, pos = _pb_read_varint(val, pos)
                vals.append(val[pos : pos + ln].decode("utf-8", "replace"))
                pos += ln
            value = vals
            atype = AttrType.STRINGS
        elif field == 10 and wire == 2:
            pos = 0
            vals = []
            while pos < len(val):
                ln, pos = _pb_read_varint(val, pos)
                vals.append(
                    _parse_tensor_proto(val[pos : pos + ln], load_external=False)
                )
                pos += ln
            value = vals
            atype = AttrType.TENSORS
        elif field == 11 and wire == 2:
            pos = 0
            vals = []
            while pos < len(val):
                ln, pos = _pb_read_varint(val, pos)
                vals.append(_parse_graph(val[pos : pos + ln]))
                pos += ln
            value = vals
            atype = AttrType.GRAPHS
        elif field == 20 and wire == 0:
            atype = int(val)
    return name, value, atype


def _parse_value_info(data: bytes) -> ValueInfo:
    name = ""
    dtype = "float32"
    shape: List[Union[int, str, None]] = []
    doc_string = ""
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 2 and wire == 2:
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:  # tensor_type
                    shape = _parse_type_proto_tensor(v2)
                    dtype = _parse_elem_type(v2)
        elif field == 3 and wire == 2:
            doc_string = val.decode("utf-8", "replace")
    return ValueInfo(name, dtype, shape, doc_string)


def _parse_type_proto_tensor(data: bytes) -> List[Union[int, str, None]]:
    shape: List[Union[int, str, None]] = []
    for f2, w2, v2 in _iter_fields(data):
        if f2 == 2 and w2 == 2:  # shape
            for f3, w3, v3 in _iter_fields(v2):
                if f3 == 1 and w3 == 2:  # dim
                    dim_val: Optional[int] = None
                    dim_param: Optional[str] = None
                    for f4, w4, v4 in _iter_fields(v3):
                        if f4 == 1 and w4 == 0:
                            dim_val = int(v4)
                        elif f4 == 2 and w4 == 2:
                            dim_param = v4.decode("utf-8", "replace")
                    if dim_param:
                        shape.append(dim_param)
                    else:
                        shape.append(dim_val)
    return shape


def _parse_elem_type(data: bytes) -> str:
    for f2, w2, v2 in _iter_fields(data):
        if f2 == 1 and w2 == 0:
            return ONNX_TO_DTYPE.get(int(v2), "float32")
    return "float32"


def _onnx_to_np(dtype_name: str) -> Optional[np.dtype]:
    return {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int64": np.int64,
        "uint64": np.uint64,
        "bool": np.bool_,
    }.get(dtype_name)


def _parse_tensor_proto(data: bytes, load_external: bool = True) -> Tensor:
    name = ""
    dtype = "float32"
    dims: List[int] = []
    raw = b""
    data_location = "DEFAULT"
    external_data: Dict[str, str] = {}
    float_data: Optional[bytes] = None
    int32_data: Optional[bytes] = None
    int64_data: Optional[bytes] = None
    double_data: Optional[bytes] = None
    uint64_data: Optional[bytes] = None
    string_data: List[bytes] = []
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:  # dims (packed int64)
            pos = 0
            while pos < len(val):
                v, pos = _pb_read_varint(val, pos)
                dims.append(_pb_signed(int(v)))
        elif field == 2 and wire == 0:
            dtype = ONNX_TO_DTYPE.get(int(val), "float32")
        elif field == 4 and wire == 2:
            float_data = val
        elif field == 5 and wire == 2:
            int32_data = val
        elif field == 6 and wire == 2:
            pos = 0
            while pos < len(val):
                ln, pos = _pb_read_varint(val, pos)
                string_data.append(val[pos : pos + ln])
                pos += ln
        elif field == 7 and wire == 2:
            int64_data = val
        elif field == 8 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 9 and wire == 2:
            raw = val
        elif field == 10 and wire == 2:
            double_data = val
        elif field == 11 and wire == 2:
            uint64_data = val
        elif field == 13 and wire == 2:  # external_data (StringStringEntry)
            key = ""
            value = ""
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:
                    key = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 2:
                    value = v2.decode("utf-8", "replace")
            if key:
                external_data[key] = value
        elif field == 14 and wire == 0:  # data_location
            data_location = {0: "DEFAULT", 1: "EXTERNAL"}.get(int(val), "DEFAULT")

    arr: Optional[np.ndarray] = None
    if external_data and load_external:
        arr = None
    elif raw:
        nptype = _onnx_to_np(dtype)
        if nptype is not None:
            arr = np.frombuffer(raw, dtype=nptype)
            arr = arr.reshape(dims) if dims else arr
    elif float_data is not None:
        arr = np.frombuffer(float_data, dtype="<f4").astype(np.float32)
        arr = arr.reshape(dims) if dims else arr
    elif int32_data is not None:
        arr = np.frombuffer(int32_data, dtype="<i4").astype(np.int32)
        arr = arr.reshape(dims) if dims else arr
    elif int64_data is not None:
        arr = np.frombuffer(int64_data, dtype="<i8").astype(np.int64)
        arr = arr.reshape(dims) if dims else arr
    elif double_data is not None:
        arr = np.frombuffer(double_data, dtype="<f8").astype(np.float64)
        arr = arr.reshape(dims) if dims else arr
    elif uint64_data is not None:
        arr = np.frombuffer(uint64_data, dtype="<u8").astype(np.uint64)
        arr = arr.reshape(dims) if dims else arr
    elif string_data:
        arr = np.asarray(string_data)
        arr = arr.reshape(dims) if dims else arr
    elif dims:
        arr = np.zeros(dims, dtype=_onnx_to_np(dtype) or np.float32)

    return Tensor(
        name=name,
        dtype=dtype,
        shape=list(dims),
        data=arr,
        data_location=data_location,
        external_data=external_data,
    )


def _parse_node(data: bytes) -> Node:
    inputs: List[str] = []
    outputs: List[str] = []
    name = ""
    op_type = ""
    attributes: Dict[str, Attribute] = {}
    domain = ""
    doc_string = ""
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            inputs.append(val.decode("utf-8", "replace"))
        elif field == 2 and wire == 2:
            outputs.append(val.decode("utf-8", "replace"))
        elif field == 3 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 4 and wire == 2:
            op_type = val.decode("utf-8", "replace")
        elif field == 5 and wire == 2:
            attr_name, attr_val, atype = _read_attr(val)
            attributes[attr_name] = Attribute(attr_name, attr_val, atype)
        elif field == 6 and wire == 2:
            doc_string = val.decode("utf-8", "replace")
        elif field == 7 and wire == 2:
            domain = val.decode("utf-8", "replace")
    return Node(op_type, inputs, outputs, name or None, attributes, domain, doc_string)


def _parse_graph(data: bytes) -> Graph:
    name = ""
    nodes: List[Node] = []
    initializers: List[Tensor] = []
    inputs: List[ValueInfo] = []
    outputs: List[ValueInfo] = []
    value_info: List[ValueInfo] = []
    doc_string = ""
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            nodes.append(_parse_node(val))
        elif field == 2 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 5 and wire == 2:
            initializers.append(_parse_tensor_proto(val))
        elif field == 10 and wire == 2:
            doc_string = val.decode("utf-8", "replace")
        elif field == 11 and wire == 2:
            inputs.append(_parse_value_info(val))
        elif field == 12 and wire == 2:
            outputs.append(_parse_value_info(val))
        elif field == 13 and wire == 2:
            value_info.append(_parse_value_info(val))
    return Graph(name, nodes, initializers, inputs, outputs, value_info, doc_string)


def parse_model(data: bytes) -> Model:
    """Parse raw binary ``.onnx`` bytes into a :class:`Model`.

    Accepts both raw protobuf and the 8-byte ``b"ONNX"+len`` SNEPPX-internal
    prefixed form (mirroring the C ``SneppX_onnx_validate``).
    """
    start = 0
    if data[:4] == b"ONNX":
        start = 8
    graph = None
    ir_version = ONNX_IR_VERSION
    producer_name = "SNEPPX"
    producer_version = "1.0"
    domain = ""
    model_version = 0
    doc_string = ""
    opsets: List[OpsetImport] = []
    metadata: List[MetadataProp] = []
    for field, wire, val in _iter_fields(data, start):
        if field == 1 and wire == 0:
            ir_version = int(val)
        elif field == 2 and wire == 2:
            producer_name = val.decode("utf-8", "replace")
        elif field == 3 and wire == 2:
            producer_version = val.decode("utf-8", "replace")
        elif field == 4 and wire == 2:
            domain = val.decode("utf-8", "replace")
        elif field == 5 and wire == 0:
            model_version = int(val)
        elif field == 6 and wire == 2:
            doc_string = val.decode("utf-8", "replace")
        elif field == 7 and wire == 2:
            graph = _parse_graph(val)
        elif field == 8 and wire == 2:
            d = ""
            v = ONNX_OPSET_VERSION
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:
                    d = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 0:
                    v = int(v2)
            opsets.append(OpsetImport(d, v))
        elif field == 14 and wire == 2:
            key = ""
            value = ""
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:
                    key = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 2:
                    value = v2.decode("utf-8", "replace")
            metadata.append(MetadataProp(key, value))
    if graph is None:
        raise OnnxParseError("ModelProto contains no graph")
    return Model(
        graph,
        opset_imports=opsets or None,
        ir_version=ir_version,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        doc_string=doc_string,
        metadata_props=metadata,
    )


def load_model(path: str) -> Model:
    """Load a model from an ``.onnx`` file path."""
    with open(path, "rb") as f:
        return parse_model(f.read())


from_bytes = parse_model
