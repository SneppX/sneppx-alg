# Higher-Order Autodiff

SNEPPX-Alg supports higher-order differentiation via `create_graph=True`.

## Computing Hessians

```python
import numpy as np
from SneppX_ALG import Tensor

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = (x ** 2).sum()
y.backward(create_graph=True)
# x.grad is now a tensor with a graph
hessian = (x.grad * x.grad).sum()
hessian.backward()
```

## Exact Hessian verification

The higher-order AD is verified against exact Hessians for:
- Linear functions: H = 2 A^T A
- Quadratic forms
- Conv1d (kernel and input)
- LayerNorm, RMSNorm
- All loss functions

## jvp (forward-mode)

```python
from SneppX_ALG.interface_bindings.autograd import jvp

def f(x):
    return x * x

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
tangent = Tensor(np.array([1.0, 1.0, 1.0]))
result = jvp(f, x, tangent)  # J @ tangent
```

## vjp (reverse-mode)

```python
from SneppX_ALG.interface_bindings.autograd import vjp

def f(x):
    return x * x

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
cotangent = Tensor(np.array([1.0, 1.0, 1.0]))
result = vjp(f, x, cotangent)  # grad of dot(cotangent, f(x))
```

## Gradient checkpointing

```python
from SneppX_ALG.interface_bindings.grad_checkpoint import checkpoint

def heavy(x):
    return x * x

out = checkpoint(heavy, x)  # saves memory
```

## Supported ops with create_graph

All 40+ ops support `create_graph=True`:
- Arithmetic: Add, Sub, Mul, Div, Pow, MatMul
- Activations: ReLU, Sigmoid, Tanh, GELU, SiLU
- Norms: LayerNorm, RMSNorm, BatchNorm
- Losses: MSE, CrossEntropy, BCE, KLDiv, NLL, SmoothL1, Huber
- Conv: Conv1d, Conv2d, Conv3d
- Pooling: MaxPool, AvgPool, AdaptiveAvgPool, AdaptiveMaxPool
- And more...