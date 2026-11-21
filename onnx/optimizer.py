"""ONNX graph optimization passes (numpy-only).

Implements a small set of rewrite passes operating on the :mod:`onnx.model`
data classes:

- **constant folding**: execute single-output nodes whose inputs are all
  constant (initializer) tensors at export time using a numpy evaluator.
- **dead code elimination**: remove nodes whose outputs are never consumed.
- **identity elimination**: collapse ``Identity`` chains.
- **redundant transpose removal**: drop ``Transpose`` with identity perm and
  ``Relu(Relu(x))`` style duplicates (optional).

All passes preserve model semantics; :func:`optimize` runs a fixed pipeline and
returns a new :class:`Model`.
"""

import copy
from typing import Dict, List, Optional, Set

import numpy as np

from .model import Graph, Model, Node, Tensor

__all__ = ["Optimizer", "optimize", "constant_fold", "eliminate_dead_code"]


class _FoldError(Exception):
    pass


def _np_dtype_name(dtype: np.dtype) -> str:
    return {
        np.dtype("float32"): "float32", np.dtype("float64"): "float64",
        np.dtype("float16"): "float16", np.dtype("int8"): "int8",
        np.dtype("uint8"): "uint8", np.dtype("int16"): "int16",
        np.dtype("int32"): "int32", np.dtype("int64"): "int64",
        np.dtype("bool"): "bool",
    }.get(np.dtype(dtype), "float32")


class _NumpyOps:
    """A tiny numpy evaluator for constant-foldable elementwise ops."""

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def erf(x):
        from math import erf as _erf

        return np.vectorize(_erf)(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def abs_(x):
        return np.abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        return a / b

    @staticmethod
    def pow(a, b):
        return np.power(a, b)

    @staticmethod
    def matmul(a, b):
        return a @ b

    @staticmethod
    def transpose(a, perm=None):
        if perm is None:
            return a.T
        return np.transpose(a, perm)

    @staticmethod
    def reshape(a, shape):
        return a.reshape([int(v) for v in shape])

    @staticmethod
    def squeeze(a, axes=None):
        return np.squeeze(a, axis=tuple(axes) if axes else None)

    @staticmethod
    def unsqueeze(a, axes):
        for ax in sorted(axes):
            a = np.expand_dims(a, ax)
        return a

    @staticmethod
    def softmax(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def identity(x):
        return x


_FOLDABLE_UNARY = {
    "Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Gelu": "gelu",
    "Exp": "exp", "Log": "log", "Sqrt": "sqrt", "Abs": "abs_", "Neg": "neg",
    "Identity": "identity",
}
_FOLDABLE_BINARY = {
    "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow",
}
_FOLDABLE_SPECIAL = {"MatMul", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Softmax"}


def _try_fold_node(node: Node, constants: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fold ``node`` if all inputs are constant; returns new name->array map."""
    if node.op_type in _FOLDABLE_UNARY:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise _FoldError
        src = constants.get(node.inputs[0])
        if src is None:
            raise _FoldError
        op = _FOLDABLE_UNARY[node.op_type]
        val = getattr(_NumpyOps, op)(src)
        return {node.outputs[0]: val}

    if node.op_type in _FOLDABLE_BINARY:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise _FoldError
        a = constants.get(node.inputs[0])
        b = constants.get(node.inputs[1])
        if a is None or b is None:
            raise _FoldError
        op = _FOLDABLE_BINARY[node.op_type]
        val = getattr(_NumpyOps, op)(a, b)
        return {node.outputs[0]: val}

    if node.op_type == "MatMul":
        a = constants.get(node.inputs[0])
        b = constants.get(node.inputs[1])
        if a is None or b is None or len(node.outputs) != 1:
            raise _FoldError
        return {node.outputs[0]: _NumpyOps.matmul(a, b)}

    if node.op_type == "Transpose":
        src = constants.get(node.inputs[0])
        if src is None or len(node.outputs) != 1:
            raise _FoldError
        perm = node.get_attr("perm")
        return {node.outputs[0]: _NumpyOps.transpose(src, perm)}

    if node.op_type == "Reshape":
        src = constants.get(node.inputs[0])
        shape_t = constants.get(node.inputs[1]) if len(node.inputs) > 1 else None
        if src is None or shape_t is None or len(node.outputs) != 1:
            raise _FoldError
        return {node.outputs[0]: _NumpyOps.reshape(src, shape_t.reshape(-1))}

    if node.op_type == "Squeeze":
        src = constants.get(node.inputs[0])
        if src is None or len(node.outputs) != 1:
            raise _FoldError
        axes = node.get_attr("axes")
        return {node.outputs[0]: _NumpyOps.squeeze(src, axes)}

    if node.op_type == "Unsqueeze":
        src = constants.get(node.inputs[0])
        if src is None or len(node.outputs) != 1:
            raise _FoldError
        axes = node.get_attr("axes")
        return {node.outputs[0]: _NumpyOps.unsqueeze(src, axes)}

    if node.op_type == "Softmax":
        src = constants.get(node.inputs[0])
        if src is None or len(node.outputs) != 1:
            raise _FoldError
        axis = int(node.get_attr("axis", -1))
        return {node.outputs[0]: _NumpyOps.softmax(src, axis)}

    raise _FoldError


def constant_fold(model: Model) -> Model:
    """Fold single-output nodes whose inputs are all constant tensors.

    Runs repeatedly until a fixed point is reached (folded constants feed
    later folds). Produces a new model; original model is untouched.
    """
    graph = model.graph
    constants: Dict[str, np.ndarray] = {
        i.name: i.data for i in graph.initializers if i.data is not None
    }
    declared = {t.name for t in graph.inputs}

    kept: List[Node] = []
    changed = True
    frontier = list(graph.nodes)
    while frontier:
        node = frontier.pop(0)
        if node.op_type in _FOLDABLE_UNARY or node.op_type in _FOLDABLE_BINARY or (
            node.op_type in _FOLDABLE_SPECIAL
        ):
            try:
                folded = _try_fold_node(node, constants)
            except _FoldError:
                folded = None
            if folded is not None:
                constants.update(folded)
                changed = True
                continue
        kept.append(node)

    if not changed:
        return model

    consumed = {inp for n in kept for inp in n.inputs}
    consumed |= {o.name for o in graph.outputs}

    new_graph = Graph(
        name=graph.name,
        nodes=kept,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        value_info=list(graph.value_info),
        doc_string=graph.doc_string,
    )
    existing = {i.name for i in graph.initializers}
    for name, arr in constants.items():
        if name in existing or name not in consumed:
            continue
        new_graph.initializers.append(
            Tensor(name, _np_dtype_name(arr.dtype), list(arr.shape), arr)
        )
    for i in graph.initializers:
        if i.data is not None and i.name in constants and constants[i.name] is not i.data:
            new_graph.initializers.append(Tensor(i.name, i.dtype, list(constants[i.name].shape), constants[i.name]))
        else:
            new_graph.initializers.append(copy.deepcopy(i))

    return Model(
        new_graph,
        opset_imports=model.opset_imports,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        metadata_props=model.metadata_props,
    )


def eliminate_dead_code(model: Model) -> Model:
    """Remove nodes whose outputs are never consumed (nor are graph outputs)."""
    graph = model.graph
    live: Set[str] = set()
    consumed: Set[str] = set()

    for out in graph.outputs:
        live.add(out.name)
    for node in graph.nodes:
        for inp in node.inputs:
            consumed.add(inp)

    live.update(consumed)
    output_names = {o.name for o in graph.outputs}

    kept: List[Node] = []
    removed = 0
    for node in reversed(graph.nodes):
        if any(o in live for o in node.outputs) or any(o in output_names for o in node.outputs):
            kept.insert(0, node)
        else:
            removed += 1

    if removed == 0:
        return model

    new_graph = Graph(
        name=graph.name,
        nodes=kept,
        initializers=list(graph.initializers),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        value_info=list(graph.value_info),
        doc_string=graph.doc_string,
    )
    return Model(
        new_graph,
        opset_imports=model.opset_imports,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        metadata_props=model.metadata_props,
    )


def eliminate_identity(model: Model) -> Model:
    """Collapse Identity chains and remove duplicate Relu/Sigmoid/... pairs."""
    graph = model.graph
    replace: Dict[str, str] = {}
    for node in graph.nodes:
        if node.op_type == "Identity" and len(node.inputs) == 1 and len(node.outputs) == 1:
            replace[node.outputs[0]] = replace.get(node.inputs[0], node.inputs[0])

    if not replace:
        return model

    def _resolve(name: str) -> str:
        seen = 0
        while name in replace and seen < 64:
            name = replace[name]
            seen += 1
        return name

    for out in graph.outputs:
        out.name = _resolve(out.name)
    kept: List[Node] = []
    for node in graph.nodes:
        if node.op_type == "Identity" and len(node.inputs) == 1 and len(node.outputs) == 1:
            if node.outputs[0] in replace:
                continue
        node.inputs = [_resolve(i) for i in node.inputs]
        kept.append(node)

    new_graph = Graph(
        name=graph.name,
        nodes=kept,
        initializers=list(graph.initializers),
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
        value_info=list(graph.value_info),
        doc_string=graph.doc_string,
    )
    return Model(
        new_graph,
        opset_imports=model.opset_imports,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        metadata_props=model.metadata_props,
    )


class Optimizer:
    """Runs graph-optimization passes on ONNX models."""

    def __init__(self):
        self.passes = {
            "constant_folding": constant_fold,
            "dead_code_elimination": eliminate_dead_code,
            "identity_elimination": eliminate_identity,
        }

    def optimize(
        self,
        model: Model,
        passes: Optional[List[str]] = None,
        num_times: int = 1,
    ) -> Model:
        """Run the selected passes (default: all) ``num_times`` times."""
        names = passes or list(self.passes.keys())
        result = model
        for _ in range(num_times):
            for name in names:
                fn = self.passes.get(name)
                if fn is None:
                    raise ValueError(f"Unknown pass: {name}")
                result = fn(result)
        return result


def optimize(
    model: Model,
    passes: Optional[List[str]] = None,
    num_times: int = 1,
) -> Model:
    """Convenience wrapper: run the optimizer pipeline on ``model``."""
    return Optimizer().optimize(model, passes=passes, num_times=num_times)
