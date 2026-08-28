"""Autograd Engine — tape-based automatic differentiation.

Provides the `Function` base class for defining differentiable operations
and the graph traversal logic for `backward()`.

Typical usage (via the op subclasses in autograd_ops.py):
    out = Add.apply(a, b)     # tracked if a or b requires_grad
    out.backward()            # populates a.grad, b.grad
"""

import numpy as np

from .tensor import Tensor

# ---------------------------------------------------------------------------
# Grad-enabled flag — controls whether operations build a backward graph.
# ---------------------------------------------------------------------------

_GRAD_ENABLED = [True]


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

        # Determine whether any input requires gradient
        needs_grad = _any_requires_grad(args)

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


def _backward(root, grad_output=None):
    """Traverse the computation graph from `root` and accumulate
    gradients into leaf tensors."""

    if not isinstance(root, Tensor):
        return
    if root.grad_fn is None:
        return

    if grad_output is None:
        grad_output = Tensor(np.ones_like(root.data), dtype=root.dtype)

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

    for t in reversed(order):
        if t.grad_fn is None:
            continue
        curr_grad = grads.get(id(t))
        if curr_grad is None:
            continue

        # Call the op's backward
        input_grads = t.grad_fn.op_cls.backward(t.grad_fn.ctx, curr_grad)

        if input_grads is None:
            continue

        # Distribute gradients to inputs
        for inp, ig in zip(t.grad_fn.inputs, input_grads):
            if not isinstance(inp, Tensor) or ig is None:
                continue
            if inp.is_leaf:
                _acc_grad(inp, ig)
            else:
                ga = grads.get(id(inp))
                if ga is None:
                    grads[id(inp)] = ig
                else:
                    grads[id(inp)] = ga + ig


def _acc_grad(tensor, grad):
    """Accumulate (add) a gradient into a leaf tensor."""
    if tensor.grad is None:
        tensor.grad = grad
    else:
        tensor.grad = tensor.grad + grad


# ---------------------------------------------------------------------------
# Hooks into Tensor — called from tensor.py
# ---------------------------------------------------------------------------


def _wrap_tensor_backward(tensor, grad_output=None):
    """Replace Tensor.backward() so it clears and traverses."""
    if grad_output is None and tensor.numel == 1:
        grad_output = Tensor(np.ones_like(tensor.data), dtype=tensor.dtype)

    _backward(tensor, grad_output)


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
        if isinstance(a, Tensor) and a.requires_grad:
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor) and item.requires_grad:
                    return True
    return False


__all__ = ["Function", "Context", "_GradFn", "_backward", "_wrap_tensor_backward"]
