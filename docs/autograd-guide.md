---
description: "Learn reverse-mode automatic differentiation, selected higher-order derivatives, and current operator coverage limits in SneppX-ALG."
---

# Autograd Guide

SneppX-ALG provides reverse-mode automatic differentiation and selected
higher-order differentiation paths. Operator and backend coverage is still
evolving; the presence of an API does not imply parity with PyTorch or a
guarantee that every gradient path is validated.

!!! tip "See it happen"
    New to autograd? Start with the [Autograd visualizer](autograd-visualizer.md) —
    type a scalar expression and watch gradients flow backwards through the graph.

## Basic usage

```python
import numpy as np
from SneppX_ALG import Tensor

x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [4, 6]
```

## No-grad context

```python
from SneppX_ALG.interface_bindings.autograd import no_grad

with no_grad():
    z = x * 2  # no graph built
```

## Higher-order AD (create_graph)

```python
x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
y = (x ** 2).sum()
y.backward(create_graph=True)
# x.grad now has a graph attached
g2 = (x.grad * x.grad).sum()
g2.backward()
```

## Forward-mode (jvp) and reverse-mode (vjp)

```python
from SneppX_ALG.interface_bindings.autograd import jvp, vjp

def f(x):
    return (x * x).sum()

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
tangent = Tensor(np.array([1.0, 1.0, 1.0]))
jvp_result = jvp(f, x, tangent)
```

## Gradient checkpointing

```python
from SneppX_ALG.interface_bindings.grad_checkpoint import checkpoint

def heavy(x):
    return x * x

out = checkpoint(heavy, x)
```

## Higher-order differentiation coverage

Higher-order differentiation is implemented for a growing set of operations.
Coverage differs by operation, dtype, and backend. Consult the current tests
and the [framework parity roadmap](roadmap/tensorflow-pytorch-parity.md)
before relying on a specific higher-order gradient. Do not infer complete
coverage from the existence of a class or function.

The `jvp` API is also under active correctness work; verify its behavior for
your operation and backend before using it in a production numerical pipeline.
