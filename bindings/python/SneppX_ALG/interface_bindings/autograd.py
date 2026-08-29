"""Autograd Engine — tape-based automatic differentiation.

Provides the `Function` base class for defining differentiable operations
and the graph traversal logic for `backward()`.

Typical usage (via the op subclasses in autograd_ops.py):
    out = Add.apply(a, b)     # tracked if a or b requires_grad
    out.backward()            # populates a.grad, b.grad
"""

import inspect
import numpy as np

from .tensor import Tensor

# ---------------------------------------------------------------------------
# Grad-enabled flag — controls whether operations build a backward graph.
# ---------------------------------------------------------------------------

_GRAD_ENABLED = [True]
_CREATE_GRAPH = [False]


def _invoke_backward(op_cls, ctx, grad, create_graph):
    """Call an op's backward, tolerant of either signature.

    Ops that support higher-order AD accept ``(ctx, grad_output, create_graph)``;
    legacy ops accept only ``(ctx, grad_output)`` and ignore the flag (their
    gradients are returned as plain arrays, breaking the graph past that node).
    """
    params = inspect.signature(op_cls.backward).parameters
    if len(params) >= 3:
        return op_cls.backward(ctx, grad, create_graph)
    return op_cls.backward(ctx, grad)


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED[0]


def set_grad_enabled(mode: bool):
    """Enable or disable gradient tracking. Returns a context manager that
    restores the previous state on exit."""
    _GRAD_ENABLED[0] = bool(mode)
    return enable_grad() if mode else no_grad()


class enable_grad:
    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = True
        return self

    def __exit__(self, *exc):
        _GRAD_ENABLED[0] = self._prev


class no_grad:
    def __enter__(self):
        self._prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = False
        return self

    def __exit__(self, *exc):
        _GRAD_ENABLED[0] = self._prev

# ---------------------------------------------------------------------------
# Context — saved state between forward and backward
# ---------------------------------------------------------------------------


class Context:
    """Stores tensors and arbitrary attributes from forward() for
    use in backward()."""

    def __init__(self):
        self._saved_tensors = {}
        self._attrs = {}

    def save_for_backward(self, **tensors):
        """save_for_backward(x=..., y=...) — keyword form so we can
        retrieve by name in backward()."""
        self._saved_tensors.update(tensors)

    def save_attr(self, **attrs):
        self._attrs.update(attrs)

    def get_saved_tensor(self, name):
        return self._saved_tensors.get(name)

    def get_attr(self, name, default=None):
        return self._attrs.get(name, default)


# ---------------------------------------------------------------------------
# Function — base class for all differentiable operations
# ---------------------------------------------------------------------------

_GradFn = None  # forward declaration, set below


class Function:
    """Subclass and define `forward(ctx, *args)` → Tensor and
    `backward(ctx, grad_output)` → list of gradients.

    Call via `MyOp.apply(...)`.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Subclasses must implement backward")

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()

        # Determine whether any input requires gradient. Under create_graph the
        # gradient tensors themselves carry grad_fn, so treat those as needing
        # grad too (this is what wires up higher-order derivatives).
        needs_grad = _any_requires_grad(args) or (
            _CREATE_GRAPH[0] and _GRAD_ENABLED[0]
        )

        output = cls.forward(ctx, *args, **kwargs)

        if needs_grad and _GRAD_ENABLED[0] and isinstance(output, Tensor):
            output.requires_grad = True
            fn = _GradFn(cls, ctx, _tensor_inputs(args))
            output._attach_grad_fn(fn)

        return output


# ---------------------------------------------------------------------------
# GradFn — edge in the graph (operation + inputs)
# ---------------------------------------------------------------------------


class _GradFn:
    """Wraps the operation, context, and input tensors for one edge."""

    __slots__ = ("op_cls", "ctx", "inputs")

    def __init__(self, op_cls, ctx, inputs):
        self.op_cls = op_cls
        self.ctx = ctx
        self.inputs = inputs  # list of Tensors that were inputs to the op


_GradFn_ref = _GradFn  # export for tensor.py


# ---------------------------------------------------------------------------
# Graph traversal — backward()
# ---------------------------------------------------------------------------


def _merge_grads(a, b, create_graph):
    """Merge two gradient contributions for one input."""
    if b is None:
        return a
    if a is None:
        return b
    if create_graph:
        from .autograd_ops import Add

        return Add.apply(a, b)
    return a + b


def _backward(root, grad_output=None, create_graph=False, return_grads=False):
    """Traverse the computation graph from `root` and accumulate
    gradients into leaf tensors.

    If `create_graph` is True, gradient tensors are themselves attached to a
    computation graph (higher-order differentiation). If `return_grads` is True,
    collected leaf gradients are returned in a dict instead of being stored in
    `.grad`.
    """

    if not isinstance(root, Tensor):
        return None
    if root.grad_fn is None:
        return None

    if grad_output is None:
        grad_output = Tensor(np.ones_like(root.data), dtype=root.dtype)

    prev_cg = _CREATE_GRAPH[0]
    _CREATE_GRAPH[0] = create_graph
    try:
        # ---- 1. Topological sort (reverse order: children before parents) ----
        order = []
        visited = set()

        def _topo(t):
            if t is None or id(t) in visited:
                return
            visited.add(id(t))
            if t.grad_fn is not None:
                for inp in t.grad_fn.inputs:
                    if isinstance(inp, Tensor) and inp.grad_fn is not None:
                        _topo(inp)
                order.append(t)

        _topo(root)

        # ---- 2. Gradient accumulation (parent-first = reversed order) ----
        grads = {id(root): grad_output}
        out_grads = {}

        for t in reversed(order):
            if t.grad_fn is None:
                continue
            curr_grad = grads.get(id(t))
            if curr_grad is None:
                continue

            # Call the op's backward (pass create_graph so it can build a graph).
            # Tolerant of ops whose backward does not yet accept create_graph.
            input_grads = _invoke_backward(
                t.grad_fn.op_cls, t.grad_fn.ctx, curr_grad, create_graph
            )

            if input_grads is None:
                continue

            # Distribute gradients to inputs
            for inp, ig in zip(t.grad_fn.inputs, input_grads):
                if not isinstance(inp, Tensor) or ig is None:
                    continue
                if inp.is_leaf:
                    if return_grads:
                        out_grads[id(inp)] = _merge_grads(
                            out_grads.get(id(inp)), ig, create_graph
                        )
                    else:
                        _acc_grad(inp, ig, create_graph)
                else:
                    grads[id(inp)] = _merge_grads(
                        grads.get(id(inp)), ig, create_graph
                    )
        return out_grads if return_grads else None
    finally:
        _CREATE_GRAPH[0] = prev_cg


def _acc_grad(tensor, grad, create_graph=False):
    """Accumulate (add) a gradient into a leaf tensor."""
    if not isinstance(grad, Tensor):
        grad = Tensor(grad)
    if tensor.grad is None:
        tensor.grad = grad
    elif create_graph:
        from .autograd_ops import Add

        tensor.grad = Add.apply(tensor.grad, grad)
    else:
        tensor.grad = tensor.grad + grad


# ---------------------------------------------------------------------------
# Hooks into Tensor — called from tensor.py
# ---------------------------------------------------------------------------


def _wrap_tensor_backward(tensor, grad_output=None, create_graph=False):
    """Replace Tensor.backward() so it clears and traverses."""
    if grad_output is None and tensor.numel == 1:
        grad_output = Tensor(np.ones_like(tensor.data), dtype=tensor.dtype)

    _backward(tensor, grad_output, create_graph=create_graph)


def grad(outputs, inputs, create_graph=False, allow_unused=False):
    """Compute gradients of `outputs` w.r.t. `inputs`.

    Returns a list of gradient Tensors (graph-connected when `create_graph`
    is True). Unlike ``Tensor.backward``, this does not mutate ``.grad``.
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    input_ids = {id(i) for i in inputs}
    result = [None] * len(inputs)

    for out in outputs:
        grads = _backward(out, create_graph=create_graph, return_grads=True)
        if grads is None:
            continue
        for idx, inp in enumerate(inputs):
            if grads.get(id(inp)) is not None:
                result[idx] = grads[id(inp)]

    if not allow_unused:
        for idx, r in enumerate(result):
            if r is None:
                raise RuntimeError(
                    f"grad: input {idx} did not receive a gradient "
                    "(not part of the graph)."
                )
    return result


def _tensor_inputs(args):
    """Extract only the Tensor objects from an argument list."""
    result = []
    for a in args:
        if isinstance(a, Tensor):
            result.append(a)
        elif isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor):
                    result.append(item)
    return result


def _any_requires_grad(args):
    for a in args:
        if isinstance(a, Tensor) and (a.requires_grad or a._grad_fn is not None):
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor) and (
                    item.requires_grad or item._grad_fn is not None
                ):
                    return True
    return False


__all__ = [
    "Function",
    "Context",
    "_GradFn",
    "_backward",
    "_wrap_tensor_backward",
    "grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "no_grad",
    "enable_grad",
]
