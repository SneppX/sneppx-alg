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

See also: [Autograd visualizer](autograd-visualizer.md) ·
[Conv2d visualizer](conv-visualizer.md) ·
[Higher-order AD](higher-order-ad.md).