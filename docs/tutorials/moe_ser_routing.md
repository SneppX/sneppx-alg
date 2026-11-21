# Tutorial — MoE SER Expert Routing

**Notebook:** [`moe_ser_routing.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/moe_ser_routing.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/moe_ser_routing.ipynb))

## What you'll build

A sparse **Mixture-of-Experts** layer using SNEPPX's SER module: route each
token to the top-2 of 8 experts, compute the load-balancing loss, and inspect
the routing distribution to detect expert collapse.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np
from SneppX_ALG import SERModel, SERConfig, Tensor
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Configure an 8-expert, top-2 MoE

```python
cfg = SERConfig()
cfg.num_experts = 8
cfg.num_active  = 2        # top-k
cfg.input_dim   = 64
cfg.expert_dim  = 128
cfg.output_dim  = 64
cfg.top_k_method = 0       # softmax routing
cfg.load_balance_coef = 0.01
cfg.dropout_rate = 0.0

moe = SERModel(cfg, seed=42, num_layers=1)
```

## 2. Forward pass

```python
x = Tensor.randn((4, 16, cfg.input_dim))   # (batch, seq, dim)
out = moe.forward(x)                         # (4, 16, 64)
```

## 3. Inspect routing weights

With the C backend, the router probabilities are exposed via the
`SNEPPXSERRouter`; here we approximate the load-balancing signal with a
pure-NumPy router replica:

```python
if not HAS_C:
    # Pure-NumPy illustration of the top-k routing math
    rng = np.random.default_rng(0)
    router_logits = rng.standard_normal((16, 8))      # (seq, num_experts)
    scores = np.softmax(router_logits, axis=-1)
    top2 = np.argsort(-scores, axis=-1)[:, :2]
    load = np.zeros(8)
    for t in range(16):
        load[top2[t]] += 1
    print("tokens-per-expert:", load.astype(int))
```

## 4. Load-balancing loss

```python
# C backend: moe.load_balance_loss(routing_weights)
# Pure-Python approximation:
routes = np.zeros(8)
for t in range(16):
    routes[top2[t]] += 1
frac = routes / 16.0
p_avg = scores.mean(axis=0)
loss = float(np.sum(frac * p_avg) * cfg.load_balance_coef)
print("balance loss:", loss)      # lower + uniform is better
```

## 5. Stacked MoE layers

```python
class MoETransformer(Module):
    def __init__(self):
        super().__init__()
        self.inp = Linear(64, 64)
        self.moe = SERModel(SERConfig(), seed=1, num_layers=4)   # 4 stacked MoE layers
        self.out = Linear(64, 64)
    def forward(self, x):
        return self.out(self.moe.forward(self.inp(x)))

net = MoETransformer()
```

## Key takeaways

- `num_active` (top-k) must be ≤ `num_experts`; increasing k raises FLOPs
  linearly but improves quality.
- `load_balance_coef` trades quality for uniform expert usage; 0.01 is the
  default in the C config.
- Expert parallelism (EP) shards experts across GPUs — set
  `ep_size = num_experts / tp_size` in `DistributedConfig`.
- Without the C backend (`_HAS_C_BACKEND is False`), `SERModel.forward` raises
  `RuntimeError: C backend not available` — the illustration above shows the
  equivalent NumPy math.

## Next steps

- Combine with [Distributed Training](distributed_training.md) for EP across
  nodes (see `algorithms/ser/core/gater.c` all-to-all dispatch).
- Add a learned MLP gater (`use_mlp_gater=1` in `SNEPPXSERConfig`).
