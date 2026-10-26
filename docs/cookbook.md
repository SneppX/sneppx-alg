# Cookbook — common recipes

Small, copy-pasteable recipes. Every Python block has a **▶ Run** button that
executes it in your browser (Pyodide + SneppX-Alg), so you can experiment
without installing anything.

## Freeze a layer (stop gradients)

```python
import numpy as np
from SneppX_ALG.interface_bindings import nn

m = nn.Linear(4, 4)
for p in m.parameters():
    p.requires_grad_(False)
trainable = sum(p.numel for p in m.parameters() if p.requires_grad)
print("trainable elements:", trainable)
```

## Gradient clipping (global norm)

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim
from SneppX_ALG.interface_bindings.trainer_v2 import clip_grad_norm_

m = nn.Linear(4, 4)
opt = optim.Adam(m.parameters(), lr=1e-2)
x = Tensor(np.random.randn(8, 4).astype(np.float32))
y = Tensor(np.random.randn(8, 4).astype(np.float32))
loss = nn.functional.mse_loss(m(x), y)
loss.backward()
before = clip_grad_norm_(list(m.parameters()), max_norm=1.0)
print("total grad norm before clipping:", round(float(before), 4))
```

## Custom autograd Function

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import autograd

class SquareFn(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x=x)
        return Tensor(x.data ** 2, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = ctx.get_saved_tensor("x")
        return [Tensor(2 * x.data * grad_output.data, dtype=grad_output.dtype)]

x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
y = SquareFn.apply(x)
y.sum().backward()
print("forward:", y.data)
print("gradient:", np.asarray(x.grad.data))
```

## Matrix multiply with a linear layer

```python
import numpy as np
from SneppX_ALG import Tensor

w = Tensor(np.random.randn(4, 4).astype(np.float32))
x = Tensor(np.random.randn(2, 4).astype(np.float32))
out = x @ w
print("dtype / shape:", out.data.dtype, out.shape)
```

## Sparse embedding lookup

```python
import numpy as np
from SneppX_ALG.interface_bindings import sparse

indices = np.array([[0, 1, 2], [0, 1, 2]])
values = np.array([1.0, 2.0, 3.0])
sp = sparse.sparse_coo_tensor(indices, values, (3, 3))
print("dense:")
print(sp.to_dense().data)
```

## Save / load a state dict

```python
from SneppX_ALG.interface_bindings import nn

m = nn.Linear(4, 4)
sd = m.state_dict()

m2 = nn.Linear(4, 4)
m2.load_state_dict(sd)
print("keys:", list(sd.keys()))
```

## Learning-rate warmup + cosine decay

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim, schedulers

m = nn.Linear(4, 2)
opt = optim.Adam(m.parameters(), lr=0.01)
sched = schedulers.LinearWarmupCosineDecay(opt, total_steps=20, warmup_steps=5)

lrs = []
for step in range(8):
    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    loss = m(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    sched.step()
    lrs.append(round(float(opt.lr), 6))
print("lr schedule:", lrs)
```

## Reproducibility seed

```python
import numpy as np
from SneppX_ALG import Tensor

np.random.seed(42)
a = Tensor(np.random.randn(3).astype(np.float32))

np.random.seed(42)
b = Tensor(np.random.randn(3).astype(np.float32))

print("reproducible:", np.allclose(a.data, b.data))
```