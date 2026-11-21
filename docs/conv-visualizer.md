# Conv2d visualizer

A convolution slides a small **kernel** over the input. Hover any output cell
to see the exact window of input it read — and how `stride` and `padding`
change the output size.

<div id="sx-cv" class="sx-cv">
  <div class="sx-cv-row">
    <div class="sx-cv-panel">
      <div class="sx-cv-title">padded input</div>
      <div class="sx-cv-grid sx-cv-in"></div>
    </div>
    <div class="sx-cv-op">∗</div>
    <div class="sx-cv-panel">
      <div class="sx-cv-title">kernel</div>
      <div class="sx-cv-grid sx-cv-k"></div>
    </div>
    <div class="sx-cv-op">=</div>
    <div class="sx-cv-panel">
      <div class="sx-cv-title">output</div>
      <div class="sx-cv-grid sx-cv-out"></div>
    </div>
  </div>
  <div class="sx-cv-controls">
    <label>stride <input class="sx-cv-stride" type="range" min="1" max="2" step="1" value="1" /> <span class="sx-cv-stride-val">1</span></label>
    <label>padding <input class="sx-cv-pad" type="range" min="0" max="1" step="1" value="0" /> <span class="sx-cv-pad-val">0</span></label>
    <button type="button" class="sx-cv-rand">🎲 Randomize</button>
  </div>
  <div class="sx-cv-shape"></div>
  <div class="sx-cv-explain"></div>
</div>

## The output-size formula

```
out = floor((H + 2·pad − kernel) / stride) + 1
```

With `H = 7`, `kernel = 3`:

| stride | pad | output |
|--------|-----|--------|
| 1 | 0 | 5×5 |
| 1 | 1 | 7×7 |
| 2 | 0 | 3×3 |
| 2 | 1 | 4×4 |

SNEPPX-Alg's `nn.Conv2d` supports `stride`, `padding`, `dilation`, and
`groups`, and its backward pass is exact (verified against finite differences
in `tests/python/test_conv2d_autograd.py`).

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn

conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
x = Tensor(np.random.randn(1, 1, 7, 7).astype(np.float32))
print(conv(x).shape)   # (1, 4, 7, 7)
```

See also: [Matmul visualizer](matmul-visualizer.md) ·
[Autograd visualizer](autograd-visualizer.md).