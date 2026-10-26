# Autograd Guide

SNEPPX-Alg provides a full autograd engine with higher-order differentiation.

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

## Supported ops with create_graph

All 40+ ops support `create_graph=True`, including:
Conv1d, Conv2d, Conv3d, Linear, LayerNorm, RMSNorm, BatchNorm, Embedding,
all losses (MSE, CrossEntropy, BCE, etc.), all activations, MatMul, and more.