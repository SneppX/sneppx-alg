"""Tensor expression kernel bindings — tensor_expr.

Wraps C implementation in ``kernel/tensor/tensor_expr.c`` with pure-Python fallback.
Provides expression-tree-based tensor computation.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class TensorExpr:
    """A node in a tensor expression tree."""

    def __init__(self, op: str, *inputs: "TensorExpr", value: Optional[np.ndarray] = None):
        self.op = op
        self.inputs = list(inputs)
        self.value = value
        self._has_c = _HAS_C

    @staticmethod
    def constant(data: np.ndarray) -> "TensorExpr":
        return TensorExpr("const", value=data)

    @staticmethod
    def parameter(data: np.ndarray) -> "TensorExpr":
        return TensorExpr("param", value=data)

    @staticmethod
    def add(a: "TensorExpr", b: "TensorExpr") -> "TensorExpr":
        return TensorExpr("add", a, b)

    @staticmethod
    def mul(a: "TensorExpr", b: "TensorExpr") -> "TensorExpr":
        return TensorExpr("mul", a, b)

    @staticmethod
    def matmul(a: "TensorExpr", b: "TensorExpr") -> "TensorExpr":
        return TensorExpr("matmul", a, b)

    @staticmethod
    def relu(x: "TensorExpr") -> "TensorExpr":
        return TensorExpr("relu", x)

    def evaluate(self, bindings: Optional[dict] = None) -> np.ndarray:
        if self.op == "const":
            return self.value
        if self.op == "param":
            if bindings and id(self) in bindings:
                return bindings[id(self)]
            return self.value
        if self.op == "add":
            return self.inputs[0].evaluate(bindings) + self.inputs[1].evaluate(bindings)
        if self.op == "mul":
            return self.inputs[0].evaluate(bindings) * self.inputs[1].evaluate(bindings)
        if self.op == "matmul":
            return self.inputs[0].evaluate(bindings) @ self.inputs[1].evaluate(bindings)
        if self.op == "relu":
            return np.maximum(self.inputs[0].evaluate(bindings), 0)
        raise ValueError(f"Unknown op: {self.op}")

    def __repr__(self) -> str:
        return f"TensorExpr({self.op}, {len(self.inputs)} inputs)"


class TensorExprCompiler:
    """Compiles a tensor expression graph into an execution plan."""

    def __init__(self):
        self._has_c = _HAS_C

    def compile(self, expr: TensorExpr) -> List[tuple]:
        plan = []
        visited = set()

        def _visit(node: TensorExpr) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for inp in node.inputs:
                _visit(inp)
            plan.append((node.op, [id(inp) for inp in node.inputs]))

        _visit(expr)
        return plan

    def execute(self, plan: List[tuple], bindings: dict) -> np.ndarray:
        results = {}
        for op, input_ids in plan:
            if op == "const":
                continue
            elif op == "param":
                continue
            else:
                inputs = [results[i] for i in input_ids]
                if op == "add":
                    results[id(plan)] = inputs[0] + inputs[1]
                elif op == "mul":
                    results[id(plan)] = inputs[0] * inputs[1]
                elif op == "matmul":
                    results[id(plan)] = inputs[0] @ inputs[1]
                elif op == "relu":
                    results[id(plan)] = np.maximum(inputs[0], 0)
        return results
