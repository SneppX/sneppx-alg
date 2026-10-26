# Optimizer race

Four optimizers, same starting point, same loss surface. Press **Start** and
watch `SGD` (momentum), `Adam`, `AdamW`, and `NAdam` race to the minimum.

<div id="sx-race" class="sx-race">
  <div class="sx-race-controls">
    <label>surface
      <select class="sx-race-surface">
        <option value="bowl">elliptic bowl</option>
        <option value="rosenbrock">Rosenbrock</option>
        <option value="valley">curved valley</option>
      </select>
    </label>
    <button type="button" class="sx-race-start">▶ Start</button>
    <button type="button" class="sx-race-reset">Reset</button>
  </div>
  <canvas class="sx-race-canvas" aria-label="Optimizer trajectories"></canvas>
  <div class="sx-race-legend"></div>
  <div class="sx-race-readout"></div>
</div>

## What to look for

- **Elliptic bowl** — high curvature in one direction. SGD with momentum
  zig-zags; Adam-family methods rescale per-coordinate and cut straight across.
- **Rosenbrock** — the classic narrow curved valley. Momentum helps, but plain
  SGD needs far more steps; adaptive methods follow the valley quickly.
- **AdamW vs Adam** — AdamW decouples weight decay, so it drifts slightly less.

These are *illustrative* dynamics (a plain JS implementation of the update
rules), not a benchmark of the C engine. For the real implementations see
[`optim.py`](https://github.com/SneppX/sneppx-alg/blob/main/bindings/python/SneppX_ALG/interface_bindings/optim.py)
and [Cookbook → learning-rate warmup](cookbook.md).

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim

for name, cls in [("SGD", optim.SGD), ("Adam", optim.Adam),
                  ("AdamW", optim.AdamW), ("NAdam", optim.NAdam)]:
    m = nn.Linear(4, 2)
    opt = cls(m.parameters(), lr=1e-2)
    x = Tensor(np.random.randn(8, 4).astype(np.float32))
    m(x).sum().backward()
    opt.step()
    opt.zero_grad()
    print(name, "ok")
```

See also: [Autograd visualizer](autograd-visualizer.md) ·
[Loss gallery](loss-gallery.md).