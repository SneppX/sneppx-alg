# Cookbook — Tensors

## 1. Create a tensor from a list, numpy array, or scalar

**Intent:** Bootstrap input data.

```python
import numpy as np
from SneppX_ALG import Tensor

t1 = Tensor([1.0, 2.0, 3.0])            # from list
t2 = Tensor(np.random.randn(4, 8))      # from numpy
t3 = Tensor(7.0, shape=(2, 2))          # scalar broadcast to a shape
t4 = Tensor.zeros((4, 8))               # zeros
t5 = Tensor.ones((4, 8))                # ones
t6 = Tensor.randn((4, 8))               # N(0,1)
t7 = Tensor.eye(4)                      # identity
```

**Notes:** `Tensor` auto-detects the C backend; the NumPy fallback is always
active. Pass `dtype="float32"` (default) or `"int64"`.

## 2. Convert to / from numpy

**Intent:** Interop with the rest of the Python ecosystem.

```python
from SneppX_ALG import Tensor
import numpy as np

arr = np.random.randn(3, 3).astype(np.float32)
t = Tensor.from_numpy(arr)
back = t.numpy()        # np.ndarray
assert np.allclose(back, arr)
```

**Notes:** `.numpy()` returns a **copy**; mutating it does not mutate the
`Tensor`. CPU-safe.

## 3. Reshape / transpose / view

**Intent:** Change tensor layout without copying (where possible).

```python
from SneppX_ALG import Tensor
t = Tensor.randn((2, 3, 4))

t.reshape((6, 4))       # alias of view
t.view(-1, 4)           # last-dim preserved
t.permute(1, 0, 2)      # -> (3, 2, 4)
t.transpose(0, 1)       # swap axes 0 and 1
t.T                     # 2-D shortcut for transpose(0,1)
t.squeeze(0)            # drop size-1 dim
t.unsqueeze(0)          # insert dim at front
```

**Notes:** Requires C backend for autograd-tracking `view`; pure-NumPy path
works for the math. CPU-safe for inference.

## 4. Element-wise math and activations

**Intent:** Apply pointwise ops.

```python
from SneppX_ALG import Tensor

t = Tensor.randn((4, 4))
_ = t + t                  # __add__  (also -, *, /, __matmul__ for @)
_ = t.relu()
_ = t.gelu()
_ = t.silu()
_ = t.sigmoid()
_ = t.tanh()
_ = t.exp()
_ = t.log()
_ = t.sqrt()
_ = -t                     # __neg__
_ = t ** 2                 # __pow__
```

**Notes:** All activations route through `autograd_ops` (CPU NumPy fallback
when no C backend). GPU: uses CUDA kernels if `_HAS_CUDA` (needs build with
`SNEPPX_BUILD_CUDA=ON`).

## 5. Reduction ops

**Intent:** Collapse dimensions.

```python
from SneppX_ALG import Tensor

t = Tensor.randn((4, 8))
m  = t.mean()              # scalar Tensor
s  = t.sum(dim=1)          # -> (4,)
v  = t.var(dim=1)          # variance
st = t.std(dim=1)          # std
mn = t.min()               # python float
mx = t.max()               # python float
```

**Notes:** `mean`/`sum` are differentiable; `min`/`max` (scalar) are not.

## 6. Matrix ops and concatenation

**Intent:** Combine and project tensors.

```python
from SneppX_ALG import Tensor

a = Tensor.randn((3, 4))
b = Tensor.randn((4, 5))
c = a @ b                 # matmul -> (3, 5)

xs = [Tensor.randn((2, 4)) for _ in range(3)]
cat = Tensor.cat(xs, dim=0)   # -> (6, 4)
stk = Tensor.stack(xs, dim=0) # -> (3, 2, 4)
```

**Notes:** `Tensor.cat`/`stack` are static; also available as module-level
`cat`/`stack`.

## 7. Convolution and pooling

**Intent:** CNN-style ops on CPU.

```python
from SneppX_ALG import Tensor

x = Tensor.randn((1, 3, 32, 32))     # NCHW
k = Tensor.randn((16, 3, 3, 3))      # out_ch, in_ch, kh, kw
y = x.conv2d(k, stride_h=1, stride_w=1, pad_h=1, pad_w=1)  # -> (1,16,32,32)

p = x.pool2d(kernel_h=2, kernel_w=2)            # 2x2 avg pool
```

**Notes:** `conv2d` uses `scipy.signal` on the NumPy fallback; CUDA path uses
`conv2d_kernel` when available. CPU-safe.

## 8. Normalization primitives

**Intent:** Layer / group / batch normalization as tensor methods.

```python
from SneppX_ALG import Tensor

x = Tensor.randn((2, 4, 8))
gamma = Tensor.ones((8,))
beta  = Tensor.zeros((8,))
ln  = x.layer_norm(gamma, beta, eps=1e-5)

gn  = x.group_norm(gamma, beta, num_groups=2, eps=1e-5)
bn  = x.batch_norm(gamma, beta, rm, rv, eps=1e-5)
```

**Notes:** Pass `Tensor` or raw `np.ndarray` for statistics.

## 9. Save / load a tensor

**Intent:** Persist a single tensor to disk.

```python
from SneppX_ALG import Tensor

t = Tensor.randn((4, 8))
t.save("/tmp/t.npy")            # .npy
loaded = Tensor.load("/tmp/t.npy")
```

**Notes:** Uses `np.save`/`np.load`. For full model state use
`model.save_checkpoint` (see [Checkpointing](checkpointing.md)).

## 10. Detach and requires_grad

**Intent:** Break the graph for inference; mark params.

```python
from SneppX_ALG import Tensor

x = Tensor.randn((4, 4), requires_grad=True)
y = x @ x                        # y.grad_fn set
leaf = y.detach()                # no grad tracking
x.requires_grad_(True)
```

**Notes:** The autograd tape records operations automatically. `detach()` produces a leaf tensor sharing memory but disconnected from gradient history.

## 11. Higher-Order Automatic Differentiation

**Intent:** Compute gradients of gradients (e.g., Hessian-vector products).

```python
from SneppX_ALG import Tensor

x = Tensor([3.0], requires_grad=True)
# First derivative y' = 3 * x^2
y = x ** 3
# With create_graph=True, backward graph itself is differentiable
y.backward(create_graph=True)
print("first grad:", x.grad.data)  # [27.0]

# Take gradient of gradient
first_grad = x.grad
first_grad.backward()
print("second grad:", x.grad.data) # [27.0 + 18.0]
```

