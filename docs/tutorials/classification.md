# Tutorial — Image Classification with SNEPPX

**Notebook:** [`classification.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/classification.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/classification.ipynb))

## What you'll build

A small MLP classifier trained on synthetic 8×8 image data, using the
SNEPPX `nn` module and `AdamW` optimizer. You'll see how `Tensor` operator
overloads, `Linear`, `LayerNorm`, and `CrossEntropyLoss` compose, and how to
log loss with the built-in `Profiler`.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np
from SneppX_ALG import (
    Tensor, Module, Sequential, Linear, LayerNorm, GELU, Dropout,
    AdamW, CrossEntropyLoss, CosineAnnealingLR, Profiler, Timer,
)
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Synthetic data

We use 8×8 "images" with 10 classes:

```python
def make_data(n=640):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 64)).astype(np.float32)
    y = rng.integers(0, 10, size=(n,)).astype(np.int64)
    return Tensor.from_numpy(X), Tensor.from_numpy(y)
```

## 2. The model

```python
class Classifier(Module):
    def __init__(self, in_dim=64, hidden=128, classes=10):
        super().__init__()
        self.net = Sequential(
            Linear(in_dim, hidden), GELU(),
            LayerNorm(hidden),
            Linear(hidden, hidden), GELU(),
            Dropout(0.1),
            Linear(hidden, classes),
        )
    def forward(self, x):
        return self.net(x)

model = Classifier()
```

## 3. Train (needs C backend for backward)

```python
X, y = make_data()
opt  = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = CosineAnnealingLR(opt, min_lr=1e-5, max_lr=2e-3, total_steps=200)
prof = Profiler(enabled=True)

for step in range(200):
    idx = np.random.randint(0, X.shape[0], 64)
    xb, yb = X[idx], y[idx]
    with Timer(prof, "forward"):
        logits = model(xb)
        loss = CrossEntropyLoss()(logits, yb)
    if not HAS_C:
        print("C backend not available — skipping backward/step")
        break
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 25 == 0:
        print(step, loss.item())
prof.print_summary()
```

## 4. Evaluate

```python
def accuracy(model, X, y, batch=128):
    if not HAS_C:
        return 0.0
    correct = 0
    for i in range(0, len(X), batch):
        preds = model(X[i:i+batch]).data.argmax(-1)
        correct += (preds == y[i:i+batch].data).sum()
    return correct / len(X)

print("accuracy:", accuracy(model, X, y))
```

## Key takeaways

- `Tensor` overloads `+`, `@`, `*` — write math, not `matmul()` calls.
- `Linear` uses PyTorch layout `(out, in)` weight, i.e. `x @ W.T + b`.
- `CrossEntropyLoss` expects logits of shape `(N, C)` and targets `(N,)`.
- Without the C backend, `backward()`/`step()` are unavailable — build the
  extension first (`cmake --build build --config Release`).

## Next steps

- Swap `Classifier` for `Transformer` (see [Generation](generation.md)).
- Add `DistributedWrapper` for multi-GPU (see
  [Distributed Training](distributed_training.md)).
