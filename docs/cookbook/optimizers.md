# Cookbook — Optimizers

## 1. AdamW (the default)

**Intent:** Decoupled-weight-decay Adam — the workhorse of LLM training.

```python
from SneppX_ALG import AdamW

opt = AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
opt.zero_grad(); loss.backward(); opt.step()
```

**Notes:** Both a **pure-Python** (`optim.py`) and a **fused CUDA**
(`kernel/cuda/optim_cuda.cu`) implementation exist. The CUDA kernel is used
automatically when `_HAS_CUDA` is set. CPU-safe.

## 2. SGD with momentum

**Intent:** Simple, reproducible baseline.

```python
from SneppX_ALG import SGD

opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

**Notes:** Momentum buffer is stored per-parameter in `opt.state`.

## 3. Lion (EvoLved Sign Direction)

**Intent:** Memory-light optimizer (stores 1 buffer instead of 2).

```python
from SneppX_ALG import Lion

opt = Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)
```

## 4. LAMB / LARS (large-batch / layerwise)

**Intent:** Scale batch size to thousands of devices.

```python
from SneppX_ALG import LAMB, LARS

opt = LAMB(model.parameters(), lr=1e-3, weight_decay=0.01)
# or, for convolutional nets:
opt = LARS(model.parameters(), lr=0.1, momentum=0.9, trust_coefficient=1e-3)
```

**Notes:** `LAMB` trusts layer norms; both trust-region optimizers.

## 5. Swap optimizers mid-run (state-dict portability)

**Intent:** Save AdamW state, resume as SGD for fine-tuning.

```python
from SneppX_ALG import AdamW, SGD

sd = opt.state_dict()                 # portable dict
opt2 = SGD(model.parameters(), lr=1e-2)
opt2.load_state_dict(sd)              # SGD ignores m/v, keeps lr
```

**Notes:** `state_dict`/`load_state_dict` store `lr`, `weight_decay`, and
per-param state. Mismatched optimizer types keep the scalar fields; mismatched
momentum buffers are re-initialized. CPU-safe.

## 6. Schedule-free AdamW

**Intent:** Remove LR scheduling overhead from the hot loop.

```python
from SneppX_ALG import ScheduleFreeAdamW   # (optim_extra)

opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3, weight_decay=0.01, warmup=500)
for step in range(10000):
    loss = model(x)
    loss.backward()
    opt.step()                       # no scheduler.step() needed
    opt.pretrain() / opt.train()     # toggle schedule-free mode
```

## 7. AdaFactor (memory-efficient, no momentum)

**Intent:** Train huge models with <1 extra buffer per param.

```python
from SneppX_ALG import AdaFactor

opt = AdaFactor(model.parameters(), lr=1e-3, scale_parameter=True, relative_step=True)
```

## 8. Sophia / SOAP (second-order-aware)

**Intent:** Faster convergence on convex-ish losses.

```python
from SneppX_ALG import Sophia, SOAP

opt = Sophia(model.parameters(), lr=1e-4)      # Hessian-free diagonal approx
opt = SOAP(model.parameters(), lr=1e-3)        # block-wise second order
```

## 9. Distributed optimizer (ZeRO-1)

**Intent:** Shard optimizer *state* across ranks.

```python
from SneppX_ALG import DistributedAdam       # distributed-aware

opt = DistributedAdam(model.parameters(), lr=1e-3, zero_stage=1)
# gradients are all-reduced before the step; state is per-rank
```

## 10. Pick a scheduler factory

**Intent:** One-liner scheduler selection.

```python
from SneppX_ALG import get_scheduler

sched = get_scheduler("cosine", opt, num_warmup_steps=500, num_training_steps=10_000)
```

**Notes:** Supports `"linear"`, `"cosine"`, `"cosine_with_restarts"`,
`"polynomial"`, `"constant"`, `"constant_with_warmup"`, `"reduce_on_plateau"`.
CPU-safe.

## 12. Optimizer Selection Reference Table

| Optimizer | State Memory Overhead | Primary Use Case | Key Hyperparameters |
|---|---|---|---|
| **AdamW** | $2\times$ parameter count ($m, v$) | General LLM and vision pretraining | `lr=2e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.01` |
| **SGD** | $1\times$ with momentum | Computer vision baselines, fine-tuning | `lr=0.01-0.1`, `momentum=0.9`, `weight_decay=1e-4` |
| **Lion** | $1\times$ parameter count | Memory-constrained large-batch training | `lr=1e-4`, `betas=(0.9, 0.99)`, `weight_decay=0.1` |
| **AdaFactor** | Factored row/col stats ($<1\times$) | Extremely large architectures | `scale_parameter=True`, `relative_step=True` |
| **SparseAdam**| Sparse tracking on indexed slices | Embedding tables, sparse features | `lr=1e-3`, `betas=(0.9, 0.999)` |
| **ScheduleFree**| $2\times$ (internal eval point) | Zero-scheduler training loops | `warmup_steps=500`, `weight_decay=0.01` |

