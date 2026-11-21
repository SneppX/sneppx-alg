"""High-level SNEPPX <-> ONNX export/import helpers (numpy-only).

The standalone ``onnx`` package builds :class:`onnx.model.Model` objects from
numpy-only descriptions. For importing models produced by the C/SneppX engine
the canonical wire format is identical, so :func:`from_sneppx_graph` and
:func:`to_sneppx_graph` bridge to the ``interface_bindings`` classes when the
full package is importable (otherwise raise a clear ImportError).
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .model import Graph, Model, Node, Tensor, ValueInfo
from .parser import load_model, parse_model
from .serializer import save_model

__all__ = [
    "export",
    "save",
    "build_graph",
    "from_sneppx_graph",
    "to_sneppx_graph",
]


def build_graph(
    name: str = "model",
    nodes: Optional[List[Node]] = None,
    initializers: Optional[Dict[str, np.ndarray]] = None,
    inputs: Optional[List[ValueInfo]] = None,
    outputs: Optional[List[ValueInfo]] = None,
    value_info: Optional[List[ValueInfo]] = None,
) -> Graph:
    """Build a :class:`Graph` from plain dicts/arrays.

    ``initializers`` maps names to numpy arrays (converted to float32
    :class:`Tensor` objects automatically).
    """
    inits: List[Tensor] = []
    if initializers:
        for k, v in initializers.items():
            arr = np.asarray(v)
            if arr.dtype not in (np.float32, np.float64, np.int32, np.int64,
                                 np.int8, np.uint8, np.int16, np.float16):
                arr = arr.astype(np.float32)
            inits.append(
                Tensor(
                    k,
                    _np_to_onnx_dtype(arr.dtype),
                    list(arr.shape),
                    np.ascontiguousarray(arr),
                )
            )
    return Graph(
        name=name,
        nodes=list(nodes) if nodes else [],
        initializers=inits,
        inputs=list(inputs) if inputs else [],
        outputs=list(outputs) if outputs else [],
        value_info=list(value_info) if value_info else [],
    )


def _np_to_onnx_dtype(dtype: np.dtype) -> str:
    return {
        np.dtype("float32"): "float32", np.dtype("float64"): "float64",
        np.dtype("float16"): "float16", np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8", np.dtype("int16"): "int16",
        np.dtype("uint16"): "uint16", np.dtype("int32"): "int32",
        np.dtype("uint32"): "uint32", np.dtype("int64"): "int64",
        np.dtype("uint64"): "uint64", np.dtype("bool"): "bool",
    }.get(np.dtype(dtype), "float32")


def export(
    path: str,
    graph: Graph,
    opset: int = 18,
    producer_name: str = "SNEPPX",
    producer_version: str = "1.0",
) -> str:
    """Serialize a :class:`Graph` to an ``.onnx`` file."""
    model = Model(
        graph,
        opset_imports=None,  # serializer defaults to `opset`
        producer_name=producer_name,
        producer_version=producer_version,
        model_version=1,
    )
    model.opset_imports = []  # force default construction in save path
    return save_model(model, path)


def save(path: str, graph: Graph) -> str:
    """Alias of :func:`export`."""
    return export(path, graph)


def from_sneppx_graph(
    graph: Any, producer_name: str = "SNEPPX", producer_version: str = "1.0"
) -> Model:
    """Convert an ``interface_bindings.OnnxGraph`` into an ``onnx`` Model."""
    try:
        from SneppX_ALG.interface_bindings.onnx_export import OnnxGraph as LegacyGraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_sneppx_graph requires the full SneppX_ALG package"
        ) from exc

    nodes = [
        Node(
            n.op_type,
            list(n.inputs),
            list(n.outputs),
            n.name,
            dict(n.attributes),
            getattr(n, "domain", ""),
        )
        for n in graph.nodes
    ]
    inits = [
        Tensor(i.name, i.dtype, list(i.data.shape), i.data)
        for i in graph.initializers
        if i.data is not None
    ]
    inputs = [ValueInfo(i.name, i.dtype, list(i.shape)) for i in graph.inputs]
    outputs = [ValueInfo(o.name, o.dtype, list(o.shape)) for o in graph.outputs]

    new_graph = Graph(
        name=graph.name,
        nodes=nodes,
        initializers=inits,
        inputs=inputs,
        outputs=outputs,
    )
    return Model(
        new_graph,
        producer_name=producer_name,
        producer_version=producer_version,
        model_version=1,
    )


def to_sneppx_graph(model: Model) -> Dict[str, Any]:
    """Flatten a parsed :class:`Model` into a dict compatible with the legacy
    ``interface_bindings.OnnxImporter.load`` result schema."""
    graph = model.graph
    return {
        "initializers": {i.name: i.data for i in graph.initializers if i.data is not None},
        "value_info": {
            v.name: {"dtype": v.dtype, "shape": v.shape} for v in graph.value_info
        },
        "nodes": [
            {
                "op_type": n.op_type,
                "inputs": list(n.inputs),
                "outputs": list(n.outputs),
                "attributes": {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in n.attributes.items()
                },
                "name": n.name,
            }
            for n in graph.nodes
        ],
        "inputs": [i.name for i in graph.inputs],
        "outputs": [o.name for o in graph.outputs],
    }
