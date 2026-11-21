"""ONNX export/import for model serialization."""

import os
import json
import struct
import numbers
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
import numpy as np

from .tensor import Tensor

# ONNX constants
ONNX_IR_VERSION = 9
ONNX_OPSET_VERSION = 18

# ONNX data type mappings
DTYPE_TO_ONNX = {
    "float32": 1,  # FLOAT
    "float16": 10,  # FLOAT16
    "int32": 6,  # INT32
    "int64": 7,  # INT64
    "int8": 3,  # INT8
    "uint8": 2,  # UINT8
    "bool": 9,  # BOOL
    "float64": 11,  # DOUBLE
    "uint16": 5,  # UINT16
    "int16": 4,  # INT16
}

ONNX_TO_DTYPE = {v: k for k, v in DTYPE_TO_ONNX.items()}

# ---- protobuf wire-format helpers (canonical ONNX IR field numbers) ----
# Canonical field numbers (onnx/onnx.in.proto), matching the C exporter in
# fs/format/onnx_format.c (SneppX_onnx_save_graph / SneppX_onnx_validate):
#   ModelProto:   ir_version=1, producer_name=2, producer_version=3,
#                 model_version=5, graph=7, opset_import=8
#   GraphProto:   node=1, name=2, initializer=5, input=11, output=12
#   NodeProto:    input=1, output=2, name=3, op_type=4, attribute=5
#   AttrProto:    name=1, f=2, i=3, ints=8, floats=7, type=20 (AttributeType)
#   ValueInfo:    name=1, type=2
#   TypeProto:    tensor_type=1
#   Tensor:       elem_type=1, shape=2
#   ShapeProto:   dim=1
#   Dimension:    dim_value=1, dim_param=2
#   TensorProto:  dims=1 (packed int64), data_type=2, name=8, raw_data=9
#   OpsetId:      domain=1, version=2


def _pb_varint(value: int) -> bytes:
    value &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_tag(field: int, wire: int) -> bytes:
    return _pb_varint((field << 3) | wire)


def _pb_varint_field(field: int, value: int) -> bytes:
    return _pb_tag(field, 0) + _pb_varint(value)


def _pb_bytes_field(field: int, payload: bytes) -> bytes:
    return _pb_tag(field, 2) + _pb_varint(len(payload)) + payload


def _pb_string_field(field: int, value: str) -> bytes:
    return _pb_bytes_field(field, value.encode("utf-8"))


def _pb_float_field(field: int, value: float) -> bytes:
    return _pb_tag(field, 5) + struct.pack("<f", float(value))


def _pb_sub(field: int, payload: bytes) -> bytes:
    return _pb_bytes_field(field, payload)


def _pb_read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _pb_signed(value: int) -> int:
    if value > 0x7FFFFFFFFFFFFFFF:
        value -= 1 << 64
    return value


def _iter_fields(
    data: bytes, start: int = 0, end: Optional[int] = None
) -> Iterator[Tuple[int, int, Any]]:
    """Iterate protobuf fields yielding (field, wire_type, value).

    Wire 0 (varint) and wire 2 (length-delimited) values are decoded; wire 1
    (fixed64) and wire 5 (fixed32) yield the raw bytes.
    """
    pos = start
    limit = len(data) if end is None else end
    while pos < limit:
        key, pos = _pb_read_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _pb_read_varint(data, pos)
            yield field, wire, val
        elif wire == 1:
            yield field, wire, data[pos : pos + 8]
            pos += 8
        elif wire == 2:
            ln, pos = _pb_read_varint(data, pos)
            yield field, wire, data[pos : pos + ln]
            pos += ln
        elif wire == 5:
            yield field, wire, data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")


# AttributeType enum (AttributeProto.type)
_ATTR_FLOAT = 1
_ATTR_INT = 2
_ATTR_STRING = 3
_ATTR_TENSOR = 4
_ATTR_FLOATS = 6
_ATTR_INTS = 7
_ATTR_STRINGS = 8


def _encode_attr(name: str, value: Any) -> bytes:
    """Encode one AttributeProto and wrap it as a NodeProto.attribute field."""
    if isinstance(value, bool):
        body = (
            _pb_string_field(1, name)
            + _pb_varint_field(20, _ATTR_INT)
            + _pb_varint_field(3, 1 if value else 0)
        )
        return _pb_sub(5, body)
    if isinstance(value, numbers.Integral):
        body = (
            _pb_string_field(1, name)
            + _pb_varint_field(20, _ATTR_INT)
            + _pb_varint_field(3, int(value))
        )
        return _pb_sub(5, body)
    if isinstance(value, numbers.Real):
        body = (
            _pb_string_field(1, name)
            + _pb_varint_field(20, _ATTR_FLOAT)
            + _pb_float_field(2, float(value))
        )
        return _pb_sub(5, body)
    if isinstance(value, (list, tuple)):
        items = list(value)
        if items and all(isinstance(v, (bool, numbers.Integral)) for v in items):
            ints = b"".join(_pb_varint(int(v)) for v in items)
            body = (
                _pb_string_field(1, name)
                + _pb_varint_field(20, _ATTR_INTS)
                + _pb_bytes_field(8, ints)
            )
            return _pb_sub(5, body)
        if items and all(isinstance(v, numbers.Real) for v in items):
            floats = b"".join(struct.pack("<f", float(v)) for v in items)
            body = (
                _pb_string_field(1, name)
                + _pb_varint_field(20, _ATTR_FLOATS)
                + _pb_bytes_field(7, floats)
            )
            return _pb_sub(5, body)
    raise ValueError(f"Unsupported ONNX attribute value for '{name}': {value!r}")


class OnnxNode:
    """Represents an ONNX node."""

    def __init__(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        domain: str = "",
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name or f"{op_type}_{id(self)}"
        self.attributes = attributes or {}
        self.domain = domain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "name": self.name,
            "attributes": self.attributes,
            "domain": self.domain,
        }

    def to_proto(self) -> bytes:
        """Encode as a canonical NodeProto (field numbers per ONNX IR)."""
        parts: List[bytes] = []
        for inp in self.inputs:
            parts.append(_pb_string_field(1, inp))
        for out in self.outputs:
            parts.append(_pb_string_field(2, out))
        if self.name:
            parts.append(_pb_string_field(3, self.name))
        parts.append(_pb_string_field(4, self.op_type))
        for attr_name, attr_value in self.attributes.items():
            parts.append(_encode_attr(attr_name, attr_value))
        return b"".join(parts)


class OnnxTensor:
    """Represents an ONNX tensor/value info."""

    def __init__(self, name: str, dtype: str, shape: List[int], doc_string: str = ""):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.doc_string = doc_string

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": DTYPE_TO_ONNX.get(self.dtype, 1),
            "shape": self.shape,
            "doc_string": self.doc_string,
        }

    def to_proto(self) -> bytes:
        """Encode as a canonical ValueInfoProto (name=1, type=2)."""
        shape_dims: List[bytes] = []
        for d in self.shape:
            if int(d) >= 0:
                shape_dims.append(_pb_sub(1, _pb_varint_field(1, int(d))))
            else:
                shape_dims.append(_pb_sub(1, _pb_string_field(2, "None")))
        tensor = _pb_varint_field(
            1, DTYPE_TO_ONNX.get(self.dtype, 1)
        ) + _pb_sub(2, b"".join(shape_dims))
        type_proto = _pb_sub(1, tensor)
        return _pb_string_field(1, self.name) + _pb_sub(2, type_proto)


class OnnxInitializer:
    """Represents an ONNX initializer (constant tensor)."""

    def __init__(self, name: str, data: np.ndarray, dtype: str):
        self.name = name
        self.data = data
        self.dtype = dtype

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": DTYPE_TO_ONNX.get(self.dtype, 1),
            "dims": list(self.data.shape),
            "data": (
                self.data.tobytes().hex()
                if self.data.dtype in [np.float32, np.float64, np.int32, np.int64]
                else self.data.tobytes()
            ),
        }

    def to_proto(self) -> bytes:
        """Encode as a canonical TensorProto (dims=1, data_type=2, name=8, raw_data=9)."""
        dims_packed = b"".join(_pb_varint(int(d)) for d in self.data.shape)
        raw = np.ascontiguousarray(self.data)
        if raw.dtype.byteorder == ">":
            raw = raw.astype(raw.dtype.newbyteorder("<"))
        raw_bytes = raw.tobytes()
        return (
            _pb_bytes_field(1, dims_packed)
            + _pb_varint_field(2, DTYPE_TO_ONNX.get(self.dtype, 1))
            + _pb_string_field(8, self.name)
            + _pb_bytes_field(9, raw_bytes)
        )


class OnnxGraph:
    """ONNX graph container."""

    def __init__(
        self,
        name: str = "model",
        inputs: Optional[List[OnnxTensor]] = None,
        outputs: Optional[List[OnnxTensor]] = None,
        value_info: Optional[List[OnnxTensor]] = None,
        nodes: Optional[List[OnnxNode]] = None,
        initializers: Optional[List[OnnxInitializer]] = None,
        doc_string: str = "",
        version: int = 0,
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.value_info = value_info or []
        self.nodes = nodes or []
        self.initializers = initializers or []
        self.doc_string = doc_string
        self.version = version
        self.metadata_props = metadata_props or {}

    def add_node(self, node: OnnxNode):
        self.nodes.append(node)

    def add_input(self, tensor: OnnxTensor):
        self.inputs.append(tensor)

    def add_output(self, tensor: OnnxTensor):
        self.outputs.append(tensor)

    def add_value_info(self, tensor: OnnxTensor):
        self.value_info.append(tensor)

    def add_initializer(self, initializer: OnnxInitializer):
        self.initializers.append(initializer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "value_info": [t.to_dict() for t in self.value_info],
            "nodes": [n.to_dict() for n in self.nodes],
            "initializers": [i.to_dict() for i in self.initializers],
            "doc_string": self.doc_string,
            "version": self.version,
            "metadata_props": self.metadata_props,
        }

    def to_proto(self) -> bytes:
        """Encode as a canonical GraphProto (node=1, name=2, initializer=5,
        input=11, output=12, value_info=13)."""
        parts: List[bytes] = []
        for node in self.nodes:
            parts.append(_pb_sub(1, node.to_proto()))
        parts.append(_pb_string_field(2, self.name))
        for init in self.initializers:
            parts.append(_pb_sub(5, init.to_proto()))
        for inp in self.inputs:
            parts.append(_pb_sub(11, inp.to_proto()))
        for out in self.outputs:
            parts.append(_pb_sub(12, out.to_proto()))
        for info in self.value_info:
            parts.append(_pb_sub(13, info.to_proto()))
        return b"".join(parts)


class OnnxModel:
    """ONNX model container."""

    def __init__(
        self,
        graph: OnnxGraph,
        opset_imports: Optional[List[Dict[str, Any]]] = None,
        ir_version: int = ONNX_IR_VERSION,
        producer_name: str = "SNEPPX",
        producer_version: str = "1.0",
        domain: str = "",
        model_version: int = 0,
        doc_string: str = "",
        metadata_props: Optional[Dict[str, str]] = None,
    ):
        self.graph = graph
        self.opset_imports = opset_imports or [
            {"version": ONNX_OPSET_VERSION, "domain": ""}
        ]
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.producer_version = producer_version
        self.domain = domain
        self.model_version = model_version
        self.doc_string = doc_string
        self.metadata_props = metadata_props or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "domain": self.domain,
            "model_version": self.model_version,
            "doc_string": self.graph.doc_string,
            "graph": self.graph.to_dict(),
            "opset_import": self.opset_imports,
            "metadata_props": [
                {"key": k, "value": v} for k, v in self.metadata_props.items()
            ],
        }

    def to_bytes(self) -> bytes:
        """Serialize to canonical binary .onnx (raw protobuf ModelProto).

        Field numbers follow the ONNX IR and are loadable by standard
        runtimes; the bytes can be cross-validated with the C
        SneppX_onnx_validate / parsed back via protobuf_to_onnx.
        """
        parts: List[bytes] = [
            _pb_varint_field(1, self.ir_version),
            _pb_string_field(2, self.producer_name),
            _pb_string_field(3, self.producer_version),
            _pb_varint_field(5, self.model_version),
            _pb_sub(7, self.graph.to_proto()),
        ]
        opsets = self.opset_imports or [{"version": ONNX_OPSET_VERSION, "domain": ""}]
        for opset in opsets:
            opset_proto = _pb_string_field(
                1, opset.get("domain", "")
            ) + _pb_varint_field(2, int(opset.get("version", ONNX_OPSET_VERSION)))
            parts.append(_pb_sub(8, opset_proto))
        return b"".join(parts)


class OnnxExporter:
    """Export SNEPPX models to ONNX format."""

    def __init__(self):
        self.graph = OnnxGraph()
        self._tensor_counter = 0
        self._initializer_map: Dict[str, str] = {}
        self._node_counter = 0

    def _new_tensor_name(self, prefix: str = "tensor") -> str:
        self._tensor_counter += 1
        return f"{prefix}_{self._tensor_counter}"

    def _new_node_name(self, op_type: str) -> str:
        self._node_counter += 1
        return f"{op_type}_{self._node_counter}"

    def add_input(self, name: str, shape: List[int], dtype: str = "float32") -> str:
        tensor = OnnxTensor(name, dtype, shape)
        self.graph.add_input(tensor)
        return name

    def add_output(self, name: str, shape: List[int], dtype: str = "float32") -> str:
        tensor = OnnxTensor(name, dtype, shape)
        self.graph.add_output(tensor)
        return name

    def add_constant(self, name: str, data: np.ndarray, dtype: str = "float32") -> str:
        initializer = OnnxInitializer(name, data, dtype)
        self.graph.add_initializer(initializer)
        self._initializer_map[name] = name
        return name

    def add_value_info(
        self, name: str, shape: List[int], dtype: str = "float32"
    ) -> str:
        tensor = OnnxTensor(name, dtype, shape)
        self.graph.add_value_info(tensor)
        return name

    def add_node(
        self,
        op_type: str,
        inputs: List[str],
        outputs: Optional[List[str]] = None,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if outputs is None:
            spec = ONNX_OP_REGISTRY.get(op_type, {})
            n_out = spec.get("outputs", 1)
            if not isinstance(n_out, int) or n_out <= 0:
                n_out = 1
            outputs = [self._new_tensor_name("out") for _ in range(n_out)]

        node = OnnxNode(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            name=name or self._new_node_name(op_type),
            attributes=attributes,
        )
        self.graph.add_node(node)

        # Add value info for outputs
        for out in outputs:
            self.add_value_info(out, [], "float32")

        return outputs

    # Helper methods for common operators
    def add_conv(
        self,
        x: str,
        weight: str,
        bias: Optional[str] = None,
        strides: Tuple[int, int] = (1, 1),
        pads: Tuple[int, int, int, int] = (0, 0, 0, 0),
        dilations: Tuple[int, int] = (1, 1),
        groups: int = 1,
        name: Optional[str] = None,
    ) -> str:
        inputs = [x, weight]
        if bias:
            inputs.append(bias)
        attrs = {
            "strides": list(strides),
            "pads": list(pads),
            "dilations": list(dilations),
            "group": groups,
        }
        return self.add_node("Conv", inputs, attributes=attrs, name=name)[0]

    def add_matmul(self, a: str, b: str, name: Optional[str] = None) -> str:
        return self.add_node("MatMul", [a, b], name=name)[0]

    def add_add(self, a: str, b: str, name: Optional[str] = None) -> str:
        return self.add_node("Add", [a, b], name=name)[0]

    def add_mul(self, a: str, b: str, name: Optional[str] = None) -> str:
        return self.add_node("Mul", [a, b], name=name)[0]

    def add_relu(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Relu", [x], name=name)[0]

    def add_gelu(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Gelu", [x], name=name)[0]

    def add_sigmoid(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Sigmoid", [x], name=name)[0]

    def add_softmax(self, x: str, axis: int = -1, name: Optional[str] = None) -> str:
        return self.add_node("Softmax", [x], attributes={"axis": axis}, name=name)[0]

    def add_batch_norm(
        self,
        x: str,
        scale: str,
        bias: str,
        mean: str,
        var: str,
        epsilon: float = 1e-5,
        momentum: float = 0.9,
        name: Optional[str] = None,
    ) -> str:
        attrs = {"epsilon": epsilon, "momentum": momentum}
        return self.add_node(
            "BatchNormalization",
            [x, scale, bias, mean, var],
            attributes=attrs,
            name=name,
        )[0]

    def add_layer_norm(
        self,
        x: str,
        weight: str,
        bias: str,
        epsilon: float = 1e-5,
        name: Optional[str] = None,
    ) -> str:
        attrs = {"epsilon": epsilon}
        return self.add_node(
            "LayerNormalization", [x, weight, bias], attributes=attrs, name=name
        )[0]

    def add_dropout(
        self, x: str, ratio: float = 0.5, name: Optional[str] = None
    ) -> str:
        attrs = {"ratio": ratio}
        return self.add_node("Dropout", [x], attributes=attrs, name=name)[0]

    def add_max_pool(
        self,
        x: str,
        kernel_shape: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        pads: Tuple[int, int, int, int] = (0, 0, 0, 0),
        name: Optional[str] = None,
    ) -> str:
        attrs = {"kernel_shape": list(kernel_shape)}
        if strides:
            attrs["strides"] = list(strides)
        if pads != (0, 0, 0, 0):
            attrs["pads"] = list(pads)
        return self.add_node("MaxPool", [x], attributes=attrs, name=name)[0]

    def add_avg_pool(
        self,
        x: str,
        kernel_shape: Tuple[int, int],
        strides: Optional[Tuple[int, int]] = None,
        pads: Tuple[int, int, int, int] = (0, 0, 0, 0),
        name: Optional[str] = None,
    ) -> str:
        attrs = {"kernel_shape": list(kernel_shape)}
        if strides:
            attrs["strides"] = list(strides)
        if pads != (0, 0, 0, 0):
            attrs["pads"] = list(pads)
        return self.add_node("AveragePool", [x], attributes=attrs, name=name)[0]

    def add_gemm(
        self,
        a: str,
        b: str,
        c: Optional[str] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        trans_a: int = 0,
        trans_b: int = 0,
        name: Optional[str] = None,
    ) -> str:
        inputs = [a, b]
        if c:
            inputs.append(c)
        attrs = {"alpha": alpha, "beta": beta, "transA": trans_a, "transB": trans_b}
        return self.add_node("Gemm", inputs, attributes=attrs, name=name)[0]

    def add_concat(
        self, inputs: List[str], axis: int = 1, name: Optional[str] = None
    ) -> str:
        return self.add_node("Concat", inputs, attributes={"axis": axis}, name=name)[0]

    def add_reshape(self, x: str, shape: List[int], name: Optional[str] = None) -> str:
        # Add shape as constant
        shape_name = self.add_constant(
            f"{x}_shape", np.array(shape, dtype=np.int64), "int64"
        )
        return self.add_node("Reshape", [x, shape_name], name=name)[0]

    def add_transpose(self, x: str, perm: List[int], name: Optional[str] = None) -> str:
        return self.add_node("Transpose", [x], attributes={"perm": perm}, name=name)[0]

    def add_squeeze(
        self, x: str, axes: Optional[List[int]] = None, name: Optional[str] = None
    ) -> str:
        attrs = {}
        if axes:
            attrs["axes"] = axes
        return self.add_node("Squeeze", [x], attributes=attrs, name=name)[0]

    def add_unsqueeze(self, x: str, axes: List[int], name: Optional[str] = None) -> str:
        return self.add_node("Unsqueeze", [x], attributes={"axes": axes}, name=name)[0]

    def add_gather(
        self, x: str, indices: str, axis: int = 0, name: Optional[str] = None
    ) -> str:
        return self.add_node(
            "Gather", [x, indices], attributes={"axis": axis}, name=name
        )[0]

    def add_reduce_sum(
        self, x: str, axes: List[int], keepdims: int = 0, name: Optional[str] = None
    ) -> str:
        return self.add_node(
            "ReduceSum", [x], attributes={"axes": axes, "keepdims": keepdims}, name=name
        )[0]

    def add_reduce_mean(
        self, x: str, axes: List[int], keepdims: int = 0, name: Optional[str] = None
    ) -> str:
        return self.add_node(
            "ReduceMean",
            [x],
            attributes={"axes": axes, "keepdims": keepdims},
            name=name,
        )[0]

    def add_split(
        self, x: str, split: List[int], axis: int = 0, name: Optional[str] = None
    ) -> List[str]:
        num_outputs = len(split)
        outputs = [self._new_tensor_name(f"split_{i}") for i in range(num_outputs)]
        attrs = {"split": split, "axis": axis}
        self.add_node("Split", [x], outputs, attributes=attrs, name=name)
        return outputs

    def add_softmax(self, x: str, axis: int = -1, name: Optional[str] = None) -> str:
        return self.add_node("Softmax", [x], attributes={"axis": axis}, name=name)[0]

    def add_sigmoid(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Sigmoid", [x], name=name)[0]

    def add_tanh(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Tanh", [x], name=name)[0]

    def add_hard_swish(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("HardSwish", [x], name=name)[0]

    def add_leaky_relu(
        self, x: str, alpha: float = 0.01, name: Optional[str] = None
    ) -> str:
        return self.add_node("LeakyRelu", [x], attributes={"alpha": alpha}, name=name)[
            0
        ]

    def add_elu(self, x: str, alpha: float = 1.0, name: Optional[str] = None) -> str:
        return self.add_node("Elu", [x], attributes={"alpha": alpha}, name=name)[0]

    def add_selu(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Selu", [x], name=name)[0]

    def add_softplus(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Softplus", [x], name=name)[0]

    def add_abs(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Abs", [x], name=name)[0]

    def add_neg(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Neg", [x], name=name)[0]

    def add_sqrt(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Sqrt", [x], name=name)[0]

    def add_exp(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Exp", [x], name=name)[0]

    def add_log(self, x: str, name: Optional[str] = None) -> str:
        return self.add_node("Log", [x], name=name)[0]

    def add_pow(self, x: str, y: str, name: Optional[str] = None) -> str:
        return self.add_node("Pow", [x, y], name=name)[0]

    def build_model(
        self,
        name: str = "model",
        producer_name: str = "SNEPPX",
        producer_version: str = "1.0",
    ) -> OnnxModel:
        model = OnnxModel(
            graph=self.graph,
            producer_name="SNEPPX",
            producer_version="1.0",
            model_version=1,
        )
        return model

    def export_binary(self, path: str) -> str:
        """Export the model as canonical binary .onnx (raw protobuf ModelProto)."""
        model = self.build_model()
        with open(path, "wb") as f:
            f.write(model.to_bytes())
        return path

    def export(self, path: str) -> str:
        model = self.build_model()
        if path.endswith(".onnx"):
            with open(path, "wb") as f:
                f.write(model.to_bytes())
            return path
        with open(path, "w") as f:
            json.dump(model.to_dict(), f, indent=2)
        return path


class OnnxImporter:
    """Import ONNX models to SNEPPX format."""

    def __init__(self):
        self.graph: Optional[OnnxGraph] = None
        self.initializers: Dict[str, np.ndarray] = {}
        self.value_info: Dict[str, Dict[str, Any]] = {}

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(0)
            if head.startswith(b"{") or head.startswith(b"["):
                data = json.load(f)
            else:
                model = protobuf_to_onnx(f.read())
                data = model.to_dict()

        # Parse graph
        graph_data = data.get("graph", {})

        # Parse initializers
        for init in graph_data.get("initializers", []):
            name = init["name"]
            dtype = ONNX_TO_DTYPE.get(init["dtype"], "float32")
            shape = init.get("dims", [])
            if "data" in init:
                data_hex = init["data"]
                if isinstance(data_hex, str):
                    data = bytes.fromhex(data_hex)
                else:
                    data = data_hex
                arr = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
            else:
                arr = np.zeros(shape, dtype=np.dtype(dtype))
            self.initializers[name] = arr

        # Parse value info
        for info in graph_data.get("value_info", []):
            self.value_info[info["name"]] = {
                "dtype": ONNX_TO_DTYPE.get(info["dtype"], "float32"),
                "shape": info.get("shape", []),
            }

        # Parse inputs/outputs
        for inp in graph_data.get("inputs", []):
            self.value_info[inp["name"]] = {
                "dtype": ONNX_TO_DTYPE.get(inp["dtype"], "float32"),
                "shape": inp.get("shape", []),
            }

        for out in graph_data.get("outputs", []):
            self.value_info[out["name"]] = {
                "dtype": ONNX_TO_DTYPE.get(out["dtype"], "float32"),
                "shape": out.get("shape", []),
            }

        # Parse nodes
        nodes = []
        for node in graph_data.get("nodes", []):
            nodes.append(
                {
                    "op_type": node["op_type"],
                    "inputs": node["inputs"],
                    "outputs": node["outputs"],
                    "attributes": node.get("attributes", {}),
                    "name": node.get("name", ""),
                }
            )

        return {
            "initializers": self.initializers,
            "value_info": self.value_info,
            "nodes": nodes,
            "inputs": [i["name"] for i in graph_data.get("inputs", [])],
            "outputs": [o["name"] for o in graph_data.get("outputs", [])],
        }

    def to_sneppx(self, onnx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed ONNX data to SNEPPX model format."""
        # This would be a full conversion pipeline
        # For now, return the parsed data
        return onnx_data


def export_onnx(
    model: Any,
    path: str,
    input_names: List[str],
    output_names: List[str],
    input_shapes: List[List[int]],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
):
    """High-level export function (placeholder for full model export)."""
    exporter = OnnxExporter()

    # This is a simplified example - real implementation would traverse the model
    # and add nodes for each layer
    for i, (name, shape) in enumerate(zip(input_names, input_shapes)):
        exporter.add_input(name, shape)

    # Add dummy output
    exporter.add_output(output_names[0], [1, 1000])

    exporter.export(path)
    return path


def import_onnx(path: str) -> Dict[str, Any]:
    """Import ONNX model."""
    importer = OnnxImporter()
    return importer.load(path)


# ONNX operator registry for custom operators
ONNX_OP_REGISTRY = {
    "Conv": {
        "inputs": 2,
        "outputs": 1,
        "attrs": ["strides", "pads", "dilations", "group"],
    },
    "MatMul": {"inputs": 2, "outputs": 1, "attrs": []},
    "Add": {"inputs": 2, "outputs": 1, "attrs": []},
    "Mul": {"inputs": 2, "outputs": 1, "attrs": []},
    "Relu": {"inputs": 1, "outputs": 1, "attrs": []},
    "Gelu": {"inputs": 1, "outputs": 1, "attrs": []},
    "Sigmoid": {"inputs": 1, "outputs": 1, "attrs": []},
    "Softmax": {"inputs": 1, "outputs": 1, "attrs": ["axis"]},
    "BatchNormalization": {"inputs": 5, "outputs": 1, "attrs": ["epsilon", "momentum"]},
    "LayerNormalization": {"inputs": 3, "outputs": 1, "attrs": ["epsilon"]},
    "Dropout": {"inputs": 1, "outputs": 2, "attrs": ["ratio"]},
    "MaxPool": {
        "inputs": 1,
        "outputs": 1,
        "attrs": ["kernel_shape", "strides", "pads"],
    },
    "AveragePool": {
        "inputs": 1,
        "outputs": 1,
        "attrs": ["kernel_shape", "strides", "pads"],
    },
    "Gemm": {"inputs": 2, "outputs": 1, "attrs": ["alpha", "beta", "transA", "transB"]},
    "Conv": {
        "inputs": 2,
        "outputs": 1,
        "attrs": ["strides", "pads", "dilations", "group"],
    },
    "MatMul": {"inputs": 2, "outputs": 1, "attrs": []},
    "Add": {"inputs": 2, "outputs": 1, "attrs": []},
    "Mul": {"inputs": 2, "outputs": 1, "attrs": []},
    "Sub": {"inputs": 2, "outputs": 1, "attrs": []},
    "Div": {"inputs": 2, "outputs": 1, "attrs": []},
    "Concat": {"inputs": -1, "outputs": 1, "attrs": ["axis"]},
    "Reshape": {"inputs": 2, "outputs": 1, "attrs": []},
    "Transpose": {"inputs": 1, "outputs": 1, "attrs": ["perm"]},
    "Squeeze": {"inputs": 1, "outputs": 1, "attrs": ["axes"]},
    "Unsqueeze": {"inputs": 1, "outputs": 1, "attrs": ["axes"]},
    "Gather": {"inputs": 2, "outputs": 1, "attrs": ["axis"]},
    "ReduceSum": {"inputs": 1, "outputs": 1, "attrs": ["axes", "keepdims"]},
    "ReduceMean": {"inputs": 1, "outputs": 1, "attrs": ["axes", "keepdims"]},
    "Split": {"inputs": 1, "outputs": -1, "attrs": ["split", "axis"]},
    "Softmax": {"inputs": 1, "outputs": 1, "attrs": ["axis"]},
    "Sigmoid": {"inputs": 1, "outputs": 1, "attrs": []},
    "Tanh": {"inputs": 1, "outputs": 1, "attrs": []},
    "HardSwish": {"inputs": 1, "outputs": 1, "attrs": []},
    "LeakyRelu": {"inputs": 1, "outputs": 1, "attrs": ["alpha"]},
    "Elu": {"inputs": 1, "outputs": 1, "attrs": ["alpha"]},
    "Selu": {"inputs": 1, "outputs": 1, "attrs": []},
    "Softplus": {"inputs": 1, "outputs": 1, "attrs": []},
    "Abs": {"inputs": 1, "outputs": 1, "attrs": []},
    "Neg": {"inputs": 1, "outputs": 1, "attrs": []},
    "Sqrt": {"inputs": 1, "outputs": 1, "attrs": []},
    "Exp": {"inputs": 1, "outputs": 1, "attrs": []},
    "Log": {"inputs": 1, "outputs": 1, "attrs": []},
    "Pow": {"inputs": 2, "outputs": 1, "attrs": []},
}

# Mapping from SNEPPX ops to ONNX ops
SNEPPX_TO_ONNX_OP = {
    "conv2d": "Conv",
    "conv1d": "Conv",
    "matmul": "MatMul",
    "add": "Add",
    "mul": "Mul",
    "relu": "Relu",
    "gelu": "Gelu",
    "sigmoid": "Sigmoid",
    "softmax": "Softmax",
    "batch_norm": "BatchNormalization",
    "layer_norm": "LayerNormalization",
    "dropout": "Dropout",
    "max_pool2d": "MaxPool",
    "avg_pool2d": "AveragePool",
    "linear": "Gemm",
    "cat": "Concat",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "squeeze": "Squeeze",
    "unsqueeze": "Unsqueeze",
    "gather": "Gather",
    "reduce_sum": "ReduceSum",
    "reduce_mean": "ReduceMean",
    "split": "Split",
    "softmax": "Softmax",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "abs": "Abs",
    "neg": "Neg",
    "sqrt": "Sqrt",
    "exp": "Exp",
    "log": "Log",
    "pow": "Pow",
}


def convert_sneppx_to_onnx_graph(model_data: Dict[str, Any]) -> OnnxGraph:
    """Convert SNEPPX model graph to ONNX graph."""
    graph = OnnxGraph(name="converted_model")
    # This is a simplified conversion - full implementation would traverse the model
    return graph


# TensorRT integration (placeholder)
class TensorRTExporter:
    """Export to TensorRT engine."""

    def __init__(self):
        self.engine = None

    def export(self, onnx_path: str, engine_path: str, max_batch_size: int = 32):
        """Export ONNX to TensorRT engine (placeholder)."""
        # Real implementation would use tensorrt package
        pass

    def build_engine(self, onnx_model: bytes, max_workspace_size: int = 1 << 30):
        """Build TensorRT engine from ONNX."""
        pass


# PyTorch interop (placeholder)
def export_torch(model: Any, path: str, input_example: Any) -> str:
    """Export PyTorch model to ONNX then to SNEPPX."""
    # Would use torch.onnx.export
    pass


def import_torch(path: str) -> Any:
    """Import PyTorch model from SNEPPX/ONNX."""
    pass


# Format conversion utilities
def onnx_to_protobuf(onnx_model: OnnxModel) -> bytes:
    """Serialize an OnnxModel to canonical binary .onnx (raw protobuf)."""
    return onnx_model.to_bytes()


def _parse_attr(data: bytes) -> Tuple[str, Any]:
    name = ""
    atype = 0
    ival = 0
    fval = 0.0
    ints: List[int] = []
    floats: List[float] = []
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 20 and wire == 0:
            atype = int(val)
        elif field == 2 and wire == 5:
            fval = struct.unpack("<f", val)[0]
        elif field == 3 and wire == 0:
            ival = _pb_signed(int(val))
        elif field == 7 and wire == 2:
            for i in range(0, len(val) - 3, 4):
                floats.append(struct.unpack("<f", val[i : i + 4])[0])
        elif field == 8 and wire == 2:
            pos = 0
            while pos < len(val):
                v, pos = _pb_read_varint(val, pos)
                ints.append(_pb_signed(int(v)))
    if atype == _ATTR_INT:
        return name, ival
    if atype == _ATTR_FLOAT:
        return name, fval
    if atype == _ATTR_INTS:
        return name, ints
    if atype == _ATTR_FLOATS:
        return name, floats
    return name, None


def _parse_node(data: bytes) -> OnnxNode:
    op_type = ""
    name = ""
    ins: List[str] = []
    outs: List[str] = []
    attrs: Dict[str, Any] = {}
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            ins.append(val.decode("utf-8", "replace"))
        elif field == 2 and wire == 2:
            outs.append(val.decode("utf-8", "replace"))
        elif field == 3 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 4 and wire == 2:
            op_type = val.decode("utf-8", "replace")
        elif field == 5 and wire == 2:
            attr_name, attr_val = _parse_attr(val)
            if attr_val is not None:
                attrs[attr_name] = attr_val
    return OnnxNode(op_type, ins, outs, name=name or None, attributes=attrs)


def _parse_value_info(data: bytes) -> OnnxTensor:
    name = ""
    dtype = "float32"
    shape: List[int] = []
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 2 and wire == 2:
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:  # tensor_type
                    for f3, w3, v3 in _iter_fields(v2):
                        if f3 == 1 and w3 == 0:  # elem_type
                            dtype = ONNX_TO_DTYPE.get(int(v3), "float32")
                        elif f3 == 2 and w3 == 2:  # shape
                            for f4, w4, v4 in _iter_fields(v3):
                                if f4 == 1 and w4 == 2:  # dim
                                    for f5, w5, v5 in _iter_fields(v4):
                                        if f5 == 1 and w5 == 0:  # dim_value
                                            shape.append(int(v5))
                                        elif f5 == 2 and w5 == 2:  # dim_param
                                            shape.append(-1)
    return OnnxTensor(name, dtype, shape)


def _parse_initializer(data: bytes) -> OnnxInitializer:
    name = ""
    dtype = "float32"
    dims: List[int] = []
    raw = b""
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            pos = 0
            while pos < len(val):
                v, pos = _pb_read_varint(val, pos)
                dims.append(_pb_signed(int(v)))
        elif field == 2 and wire == 0:
            dtype = ONNX_TO_DTYPE.get(int(val), "float32")
        elif field == 8 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 9 and wire == 2:
            raw = val
    if dims and raw:
        arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(dims)
    elif dims:
        arr = np.zeros(dims, dtype=np.dtype(dtype))
    else:
        arr = np.array([], dtype=np.dtype(dtype))
    return OnnxInitializer(name, arr, dtype)


def _parse_graph(data: bytes) -> OnnxGraph:
    name = ""
    nodes: List[OnnxNode] = []
    inputs: List[OnnxTensor] = []
    outputs: List[OnnxTensor] = []
    initializers: List[OnnxInitializer] = []
    for field, wire, val in _iter_fields(data):
        if field == 1 and wire == 2:
            nodes.append(_parse_node(val))
        elif field == 2 and wire == 2:
            name = val.decode("utf-8", "replace")
        elif field == 5 and wire == 2:
            initializers.append(_parse_initializer(val))
        elif field == 11 and wire == 2:
            inputs.append(_parse_value_info(val))
        elif field == 12 and wire == 2:
            outputs.append(_parse_value_info(val))
    return OnnxGraph(
        name=name,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers,
    )


def protobuf_to_onnx(proto_bytes: bytes) -> OnnxModel:
    """Parse canonical binary .onnx (raw protobuf ModelProto) back into an
    OnnxModel. Accepts both raw protobuf and the 8-byte "ONNX"+len-prefixed
    SNEPPX-internal form (mirroring the C SneppX_onnx_validate)."""
    start = 0
    if proto_bytes[:4] == b"ONNX":
        start = 8
    graph = None
    ir_version = ONNX_IR_VERSION
    producer_name = "SNEPPX"
    producer_version = "1.0"
    model_version = 0
    opsets: List[Dict[str, Any]] = []
    for field, wire, val in _iter_fields(proto_bytes, start):
        if field == 1 and wire == 0:
            ir_version = int(val)
        elif field == 2 and wire == 2:
            producer_name = val.decode("utf-8", "replace")
        elif field == 3 and wire == 2:
            producer_version = val.decode("utf-8", "replace")
        elif field == 5 and wire == 0:
            model_version = int(val)
        elif field == 7 and wire == 2:
            graph = _parse_graph(val)
        elif field == 8 and wire == 2:
            domain = ""
            version = ONNX_OPSET_VERSION
            for f2, w2, v2 in _iter_fields(val):
                if f2 == 1 and w2 == 2:
                    domain = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 0:
                    version = int(v2)
            opsets.append({"version": version, "domain": domain})
    if graph is None:
        raise ValueError("ModelProto contains no graph")
    return OnnxModel(
        graph,
        opset_imports=opsets or None,
        ir_version=ir_version,
        producer_name=producer_name,
        producer_version=producer_version,
        model_version=model_version,
    )


def onnx_validate(onnx_model: OnnxModel) -> Tuple[bool, List[str]]:
    """Validate an OnnxModel, mirroring the C SneppX_onnx_validate checks.

    Structural + connectivity checks: ir_version/opset present, graph name,
    node op_types non-empty, node inputs declared (graph input, initializer,
    or an earlier node output), graph outputs produced by a node, and unique
    node output names.
    """
    errors = []

    # Model-level structure
    if onnx_model.ir_version <= 0:
        errors.append("Model has no valid ir_version")
    if not onnx_model.opset_imports:
        errors.append("Model has no opset import")

    graph = onnx_model.graph
    if not graph.name:
        errors.append("Graph has no name")
    if not graph.inputs:
        errors.append("Model has no inputs")
    if not graph.outputs:
        errors.append("Model has no outputs")
    if not graph.nodes:
        errors.append("Model has no nodes")

    # Declared names: graph inputs + initializers
    declared = {t.name for t in graph.inputs}
    declared |= {i.name for i in graph.initializers}
    produced = set()
    op_types = set()
    all_outputs = set()

    for node in graph.nodes:
        if not node.op_type:
            errors.append("Node has empty op_type")
        else:
            op_types.add(node.op_type)

        for out in node.outputs:
            if out in all_outputs:
                errors.append(f"Duplicate output name: {out}")
            all_outputs.add(out)
            produced.add(out)

        for inp in node.inputs:
            if inp not in declared and inp not in produced:
                errors.append(f"Node input '{inp}' is not declared or produced")

    # Every graph output must be produced by some node
    for out in graph.outputs:
        if out.name not in produced:
            errors.append(f"Graph output '{out.name}' is not produced by any node")

    return len(errors) == 0, errors


# Version compatibility
def onnx_version_compatible(
    onnx_model: OnnxModel, target_opset: int = ONNX_OPSET_VERSION
) -> bool:
    """Check if ONNX model is compatible with target opset."""
    for opset in onnx_model.opset_imports:
        if opset.get("version", 0) > target_opset:
            return False
    return True


def upgrade_opset(onnx_model: OnnxModel, target_opset: int) -> OnnxModel:
    """Upgrade ONNX model to target opset."""
    # Would implement opset version conversion
    onnx_model.opset_imports = [{"version": target_opset, "domain": ""}]
    return onnx_model
