---
description: "Interactive matrix multiplication visualizer: hover any output cell to see the row, column and dot product that produced it."
---

# Matmul visualizer

Matrix multiplication is a sum of row · column dot products. Hover any cell
of **C** to see exactly which row of **A** and which column of **B** produced
it — and the arithmetic term by term.

<div id="sx-mm" class="sx-mm">
  <div class="sx-mm-row">
    <div class="sx-mm-panel"><div class="sx-mm-title">A <span>3×4</span></div><div class="sx-mm-grid sx-mm-a"></div></div>
    <div class="sx-mm-op">×</div>
    <div class="sx-mm-panel"><div class="sx-mm-title">B <span>4×3</span></div><div class="sx-mm-grid sx-mm-b"></div></div>
    <div class="sx-mm-op">=</div>
    <div class="sx-mm-panel"><div class="sx-mm-title">C <span>3×3</span></div><div class="sx-mm-grid sx-mm-c"></div></div>
  </div>
  <div class="sx-mm-controls">
    <button type="button" class="sx-mm-rand">🎲 Randomize</button>
    <button type="button" class="sx-mm-sweep">▶ Sweep all cells</button>
  </div>
  <div class="sx-mm-explain"></div>
</div>

## Why it matters for training

The same shape rule governs `linear` layers: an input `(N, K)` times a weight
`(K, N_out)` produces `(N, N_out)`. A shape mismatch like `(N, C, L) @ (C,)`
fails instead of broadcasting — reshape the bias to `(1, C, 1)` first, as
SNEPPX-Alg's `Conv` layers do.

```python
import numpy as np
from SneppX_ALG import Tensor

x = Tensor(np.random.randn(2, 3).astype(np.float32))
w = Tensor(np.random.randn(3, 4).astype(np.float32))
print((x @ w).shape)          # (2, 4) — batch dimension preserved
```

## How matrix multiplication works

For `A` of shape `(m, k)` and `B` of shape `(k, n)`, each output entry is a dot
product:

```text
C[i][j] = A[i][0]·B[0][j] + A[i][1]·B[1][j] + ... + A[i][k-1]·B[k-1][j]
```

Three rules follow from that formula:

- **Shape rule.** The inner dimensions must match (`k` and `k`), and the result
  takes the outer dimensions: `(m, k) @ (k, n) → (m, n)`.
- **Cost.** Every one of the `m·n` outputs needs `k` multiply-adds, so a matmul
  costs `m·n·k` operations. This is why matmuls dominate the run time of
  Transformers and other dense networks.
- **Not commutative.** `A @ B` and `B @ A` are generally different, and often
  only one of them has valid shapes.

In backpropagation, the gradients of `C = A @ B` are themselves matmuls:
`∂L/∂A = ∂L/∂C @ Bᵀ` and `∂L/∂B = Aᵀ @ ∂L/∂C`.

See also: [Autograd visualizer](autograd-visualizer.md) ·
[Conv2d visualizer](conv-visualizer.md) ·
[Higher-order AD](higher-order-ad.md).
