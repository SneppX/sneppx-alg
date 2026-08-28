"""Gradient checkpointing (activation recomputation).

Mirrors ``torch.utils.checkpoint``: during the forward pass the wrapped
function is executed under :class:`no_grad` so intermediate activations are
not retained. On the backward pass the function is recomputed under
:class:`enable_grad` and the recomputed graph is differentiated, producing
gradients identical to a non-checkpointed run while trading compute for
memory.
"""

from .autograd import (
    Function,
    _backward,
    no_grad,
    enable_grad,
    _GradFn,
    _tensor_inputs,
    Context,
)
from .tensor import Tensor

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointFunction",
    "CheckpointSegment",
    "GradientCheckpointer",
]


class CheckpointFunction(Function):
    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx._fn
        args = ctx._args
        # Recompute the forward graph with gradients enabled, then
        # differentiate it. Gradients accumulate directly into the original
        # argument tensors (which are the leaves of both graphs).
        with enable_grad():
            output = fn(*args)
        if isinstance(output, Tensor) and output.grad_fn is not None:
            _backward(output, grad_output)
        # The real gradients were written into args' .grad by the recompute;
        # report None so the outer engine does not double-count them.
        return (None,) + tuple(None for _ in ctx._args)


def checkpoint(fn, *args):
    """Run ``fn(*args)`` with gradient checkpointing.

    ``fn`` must return a single Tensor. The returned tensor carries a
    backward graph that, on backward, recomputes ``fn`` and backpropagates
    through it. This trades extra compute for reduced activation memory.
    """
    with no_grad():
        output = fn(*args)

    if isinstance(output, Tensor):
        # The inner forward ran under no_grad (so activations are not stored),
        # but we still need a backward edge that triggers recomputation.
        ctx = Context()
        ctx._fn = fn
        ctx._args = args
        output.requires_grad = True
        output._attach_grad_fn(_GradFn(CheckpointFunction, ctx, _tensor_inputs(args)))
    return output


def checkpoint_sequential(functions, input, segments=1):
    """Checkpoint a sequential list of callables ``functions`` applied to
    ``input``. ``segments`` controls how many checkpointed chunks the sequence
    is split into (1 = checkpoint the whole sequence as one unit)."""
    if segments < 1:
        segments = 1
    chunks = [functions[i::segments] for i in range(segments)] if segments > 1 else [functions]

    def run_chunk(fns, x):
        for f in fns:
            x = f(x)
        return x

    out = input
    for i in range(segments):
        fns = chunks[i] if segments > 1 else functions
        out = checkpoint(lambda xs, fns=fns: run_chunk(fns, xs), out)
    return out


class CheckpointSegment:
    """A reusable checkpointed segment: wraps a list of callables and runs them
    under :func:`checkpoint_sequential`."""

    def __init__(self, functions):
        self.functions = list(functions)

    def __call__(self, x):
        return checkpoint_sequential(self.functions, x, segments=1)


class GradientCheckpointer:
    """Checkpointer that wraps a single callable under gradient checkpointing.

    Example::

        cp = GradientCheckpointer(model)
        out = cp(x)            # equivalent to checkpoint(model, x)
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return checkpoint(self.fn, *args)
