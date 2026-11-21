# Loss gallery

Pick a loss function and see both its **value** and its **gradient**. The
gradient is what the optimiser actually follows, so two losses that look
similar can behave very differently in training.

<div id="sx-loss" class="sx-loss">
  <div class="sx-loss-controls">
    <label>loss
      <select class="sx-loss-select"></select>
    </label>
  </div>
  <div class="sx-loss-params"></div>
  <canvas class="sx-loss-canvas" aria-label="Loss and gradient curves"></canvas>
  <div class="sx-loss-legend"></div>
  <div class="sx-loss-readout"></div>
</div>

## Why the gradient matters more than the value

| Loss | Gradient for large error | Effect |
|------|--------------------------|--------|
| `MSELoss` | grows linearly | outliers dominate the update |
| `L1Loss` | constant | robust to outliers, but noisy near zero |
| `HuberLoss` | clipped at δ | quadratic near zero, linear in the tails |
| `SmoothL1Loss` | → sign(z) | same idea, C¹-continuous at ±β |
| `log-cosh` | → ±1, smoothly | robust *and* differentiable everywhere |
| `Geman-McClure` | → 0 | bounded influence: very strong outlier rejection |
| `HingeLoss` | 0 once satisfied | no gradient for already-correct margins |
| `BCEWithLogits` | → −1 | never stops pushing on badly wrong predictions |
| `focal loss` | → 0 for easy examples | down-weights well-classified samples |
| `quantile loss` | asymmetric | pinball loss for a τ-quantile forecast |

## Robust vs sensitive, side by side

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn

pred = Tensor(np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32), requires_grad=True)
tgt = Tensor(np.zeros(4, dtype=np.float32))

mse = nn.MSELoss()(pred, tgt)
mse.backward()
print("MSE grad:", np.asarray(pred.grad.data))     # outlier at 10 dominates
```

Add an outlier to the target and `MSELoss` moves the whole batch; `L1Loss`
or `HuberLoss` barely notice. That is the entire reason those losses exist.

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn

pred = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
tgt = Tensor(np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32))

for name, loss_fn in [("L1", nn.L1Loss()), ("Huber", nn.HuberLoss())]:
    loss = loss_fn(pred, tgt)
    loss.backward()
    print(name, "grad:", np.asarray(pred.grad.data))
    pred.zero_grad_()
```

## Classification losses

For a binary problem the logit `z` is mapped through a sigmoid. Plotting the
loss against `z` shows why `BCEWithLogitsLoss` is numerically preferred over
`sigmoid` followed by `BCELoss`: the fused form avoids `log(0)` when a
prediction saturates.

```python
import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn

logits = Tensor(np.array([-20.0, 0.0, 20.0], dtype=np.float32), requires_grad=True)
target = Tensor(np.array([1.0, 1.0, 1.0], dtype=np.float32))

loss = nn.BCEWithLogitsLoss()(logits, target)
loss.backward()
print("loss:", float(np.asarray(loss.data).reshape(-1)[0]))
print("grad:", np.asarray(logits.grad.data))   # saturates to 0 and -1, never NaN
```

See also: [Optimizer race](optimizer-race.md) ·
[Autograd visualizer](autograd-visualizer.md) ·
[Cookbook](cookbook.md).
