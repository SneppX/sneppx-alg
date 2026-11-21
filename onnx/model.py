"""ONNX IR data model (numpy-only, standalone).

Defines the container classes used across the ``onnx`` package: values, tensors,
nodes, graphs and models. The wire-encoding lives in :mod:`onnx.wire`, parsing in
:mod:`onnx.parser`, serialization in :mod:`onnx.serializer`.

All classes are plain Python objects with ``to_dict``/``from_dict`` helpers so
models can round-trip through JSON as well as the canonical binary format.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .wire import (
    _pb_bytes_field,
    _pb_float_field,
    _pb_string_field,
    _pb_varint_field,
    _pb_sub,
)

__all__ = [
    "AttrType",
    "Attribute",
    "Tensor",
    "ValueInfo",
    "Dimension",
    "Node",
    "Graph",
    "OpsetImport",
    "MetadataProp",
    "Model",
    "DTYPE_TO_ONNX",
    "ONNX_TO_DTYPE",
    "ONNX_IR_VERSION",
    "ONNX_OPSET_VERSION",
]

ONNX_IR_VERSION = 9
ONNX_OPSET_VERSION = 18

# ONNX data type mappings (TensorProto.DataType)
DTYPE_TO_ONNX = {
    "float32": 1,  # FLOAT
    "uint8": 2,  # UINT8
    "int8": 3,  # INT8
    "uint16": 5,  # UINT16
    "int16": 4,  # INT16
    "int32": 6,  # INT32
    "int64": 7,  # INT64
    "bool": 9,  # BOOL
    "float16": 10,  # FLOAT16
    "float64": 11,  # DOUBLE
    "string": 8,  # STRING
    "uint32": 12,  # UINT32
    "uint64": 13,  # UINT64
    "complex64": 14,  # COMPLEX64
    "complex128": 15,  # COMPLEX128
    "bfloat16": 16,  # BFLOAT16
}

ONNX_TO_DTYPE = {v: k for k, v in DTYPE_TO_ONNX.items()}

# numpy dtype -> ONNX canonical names used for external-data round trips
NP_DTYPE_TO_ONNX = {
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("float16"): "float16",
    np.dtype("int8"): "int8",
    np.dtype("uint8"): "uint8",
    np.dtype("int16"): "int16",
    np.dtype("uint16"): "uint16",
    np.dtype("int32"): "int32",
    np.dtype("uint32"): "uint32",
    np.dtype("int64"): "int64",
    np.dtype("uint64"): "uint64",
    np.dtype("bool"): "bool",
}

# AttributeProto.AttributeType enum
class AttrType:
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    TYPE_PROTO = 13
    TYPE_PROTOS = 14


class Attribute:
    """An ONNX node attribute (typed value)."""

    __slots__ = ("name", "value", "attr_type", "doc_string")

    def __init__(
        self,
        name: str,
        value: Any,
        attr_type: Optional[int] = None,
        doc_string: str = "",
    ):
        self.name = name
        self.value = value
        self.attr_type = attr_type
        self.doc_string = doc_string
        if attr_type is None:
            self.attr_type = self._infer_type(value)

    @staticmethod
    def _infer_type(value: Any) -> int:
        import numbers

        if isinstance(value, bool):
            return AttrType.INT
        if isinstance(value, numbers.Integral):
            return AttrType.INT
        if isinstance(value, numbers.Real):
            return AttrType.FLOAT
        if isinstance(value, bytes):
            return AttrType.STRING
        if isinstance(value, str):
            return AttrType.STRING
        if isinstance(value, np.ndarray):
            return AttrType.TENSOR
        if isinstance(value, Tensor):
            return AttrType.TENSOR
        if isinstance(value, Graph):
            return AttrType.GRAPH
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(v, (bool, int)) for v in value):
                return AttrType.INTS
            if value and all(isinstance(v, float) for v in value):
                return AttrType.FLOATS
            if value and all(isinstance(v, str) for v in value):
                return AttrType.STRINGS
            if value and all(isinstance(v, np.ndarray) or isinstance(v, Tensor) for v in value):
                return AttrType.TENSORS
            if value and all(isinstance(v, Graph) for v in value):
                return AttrType.GRAPHS
            if value and all(isinstance(v, bytes) for v in value):
                return AttrType.STRINGS
        return AttrType.UNDEFINED

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "type": self.attr_type}


class Dimension:
    """A single shape dimension (concrete value, symbolic name, or dynamic)."""

    __slots__ = ("value", "param", "denotation")

    def __init__(self, value: Optional[int] = None, param: Optional[str] = None,
                 denotation: str = ""):
        self.value = value
        self.param = param
        self.denotation = denotation

    @property
    def is_dynamic(self) -> bool:
        return self.value is None and not self.param

    def __repr__(self) -> str:
        if self.param:
            return self.param
        if self.value is None:
            return "?"
        return str(self.value)

    def __int__(self) -> int:
        return int(self.value) if self.value is not None else -1


class Tensor:
    """An ONNX tensor: initializer data or value-info type descriptor."""

    __slots__ = (
        "name",
        "dtype",
        "shape",
        "data",
        "dim_params",
        "doc_string",
        "data_location",
        "external_data",
    )

    def __init__(
        self,
        name: str,
        dtype: str = "float32",
        shape: Optional[List[Union[int, str, None]]] = None,
        data: Optional[np.ndarray] = None,
        doc_string: str = "",
        data_location: str = "DEFAULT",
        external_data: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape) if shape is not None else []
        self.data = data
        self.doc_string = doc_string
        self.data_location = data_location
        self.external_data = dict(external_data) if external_data else {}

    @property
    def is_initializer(self) -> bool:
        return self.data is not None or bool(self.external_data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": DTYPE_TO_ONNX.get(self.dtype, 1),
            "shape": [repr(d) if isinstance(d, str) else d for d in self.shape],
            "data": (
                self.data.tobytes().hex() if self.data is not None else None
            ),
            "doc_string": self.doc_string,
        }


class ValueInfo:
    """Graph input/output or intermediate value-info (name + type/shape)."""

    __slots__ = ("name", "dtype", "shape", "doc_string")

    def __init__(self, name: str, dtype: str = "float32",
                 shape: Optional[List[Union[int, str, None]]] = None,
                 doc_string: str = ""):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape) if shape is not None else []
        self.doc_string = doc_string

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "dtype": DTYPE_TO_ONNX.get(self.dtype, 1),
                "shape": self.shape}


class Node:
    """An ONNX graph node (operator instantiation)."""

    __slots__ = ("op_type", "inputs", "outputs", "name", "attributes",
                 "domain", "doc_string")

    def __init__(
        self,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        domain: str = "",
        doc_string: str = "",
    ):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.name = name if name else ""
        self.attributes = attributes if attributes is not None else {}
        self.domain = domain
        self.doc_string = doc_string

    def get_attr(self, name: str, default: Any = None) -> Any:
        if name in self.attributes:
            return self.attributes[name].value if isinstance(
                self.attributes[name], Attribute
            ) else self.attributes[name]
        return default

    def set_attr(self, name: str, value: Any) -> None:
        if isinstance(value, Attribute):
            self.attributes[name] = value
        else:
            self.attributes[name] = Attribute(name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "name": self.name,
            "attributes": {
                k: (v.to_dict() if isinstance(v, Attribute) else v)
                for k, v in self.attributes.items()
            },
            "domain": self.domain,
        }


class Graph:
    """ONNX computation graph."""

    __slots__ = ("name", "nodes", "initializers", "inputs", "outputs",
                 "value_info", "doc_string")

    def __init__(
        self,
        name: str = "model",
        nodes: Optional[List[Node]] = None,
        initializers: Optional[List[Tensor]] = None,
        inputs: Optional[List[ValueInfo]] = None,
        outputs: Optional[List[ValueInfo]] = None,
        value_info: Optional[List[ValueInfo]] = None,
        doc_string: str = "",
    ):
        self.name = name
        self.nodes = list(nodes) if nodes else []
        self.initializers = list(initializers) if initializers else []
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.value_info = list(value_info) if value_info else []
        self.doc_string = doc_string

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_initializer(self, tensor: Tensor) -> None:
        self.initializers.append(tensor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "initializers": [t.to_dict() for t in self.initializers],
            "inputs": [v.to_dict() for v in self.inputs],
            "outputs": [v.to_dict() for v in self.outputs],
            "value_info": [v.to_dict() for v in self.value_info],
        }


class OpsetImport:
    __slots__ = ("domain", "version")

    def __init__(self, domain: str = "", version: int = ONNX_OPSET_VERSION):
        self.domain = domain
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "version": self.version}


class MetadataProp:
    __slots__ = ("key", "value")

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "value": self.value}


class Model:
    """Top-level ONNX model container."""

    __slots__ = ("graph", "opset_imports", "ir_version", "producer_name",
                 "producer_version", "domain", "model_version", "doc_string",
                 "metadata_props", "functions")

    def __init__(
        self,
        graph: Graph,
        opset_imports: Optional[List[OpsetImport]] = None,
        ir_version: int = ONNX_IR_VERSION,
        producer_name: str = "SNEPPX",
        producer_version: str = "1.0",
        domain: str = "",
        model_version: int = 0,
        doc_string: str = "",
        metadata_props: Optional[List[MetadataProp]] = None,
        functions: Optional[List[Any]] = None,
    ):
        self.graph = graph
        self.opset_imports = list(opset_imports) if opset_imports else [
            OpsetImport("", ONNX_OPSET_VERSION)
        ]
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.producer_version = producer_version
        self.domain = domain
        self.model_version = model_version
        self.doc_string = doc_string
        self.metadata_props = list(metadata_props) if metadata_props else []
        self.functions = list(functions) if functions else []

    @property
    def opset_version(self) -> int:
        for opset in self.opset_imports:
            if opset.domain == "":
                return opset.version
        return self.opset_imports[0].version if self.opset_imports else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ir_version": self.ir_version,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "domain": self.domain,
            "model_version": self.model_version,
            "doc_string": self.doc_string,
            "graph": self.graph.to_dict(),
            "opset_import": [o.to_dict() for o in self.opset_imports],
            "metadata_props": [m.to_dict() for m in self.metadata_props],
        }
