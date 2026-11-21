---
description: "Overview of the SneppX-ALG five-stage algorithm pipeline, HSS, SER, ARC, NPE, and FM, with implementation context."
---

# The 5-Pipeline Architecture

SNEPPX-Algo is a **composable 5-component AI algorithm pipeline** wrapped in
**10 security layers (S0–S9)**. Data flows through five sequential algorithm
components, each of which can also operate as a standalone module.

```
                    ┌──────────────────────────────────────────┐
                    │           Security Ring (S0–S9)           │
                    │  PQ Crypto · Secure Mem · Obfuscation      │
                    │  Monitor · Network · AI Sanitizer ·       │
                    │  Key Vault · Updates · Formal · Pentest   │
                    └──────────────────────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │          Algorithm Pipeline          │
                    │  HSS → SER → ARC → NPE → FM         │
                    │ (SSM)  (MoE)  (Guard) (VM) (FedMem) │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │         Integrity Layer             │
                    │  ZK Proofs (opt-in) · Formal Safety │
                    │  On-Device Attestation              │
                    └─────────────────────────────────────┘
```

## Pipeline at a glance

| Stage | Acronym | Type | Role | C backend | Python entry |
|-------|---------|------|------|-----------|--------------|
| 1 | **HSS** | SSM | Sequence encoding via hierarchical state spaces | `algorithms/hss/core/` | `HSSModel`, `HSSConfig` |
| 2 | **SER** | MoE | Sparse expert routing with top-k load balancing | `algorithms/ser/core/` | `SERModel`, `SERConfig` |
| 3 | **ARC** | Guard | Adversarial robustness: input guard, gradient obfuscation, output verifier | `algorithms/arc/core/` | `ARCModel`, `ARCConfig` |
| 4 | **NPE** | VM | Neural *program* execution on a 16-register, 32-opcode VM | `algorithms/npe/core/` | `NPEModel`, `NPEConfig`, `NPEProgram` |
| 5 | **FM** | FedMem | Federated memory bank with trust-weighted all-reduce | `algorithms/fm/core/` | `FMModel`, `FMConfig` |

Each stage can be **stitched** into larger graphs (attention, MLP, residual
streams) through the `nn.Module` API in `SneppX_ALG`. The stages communicate
only through `Tensor` objects, so any stage can be swapped for a CUDA
accelerated variant (`algorithms/<stage>/cuda/`) when a GPU is present.

## 1. HSS — Hierarchical State Space (SSM)

HSS encodes a sequence by maintaining a per-layer hidden state that is updated
at each timestep with zero-order-hold (ZOH) discretization of a continuous-time
state-space model.

**Math.** For layer `l`, timestep `k`:

```
A̅ = exp(D·A)                         (discretized state transition)
B̅ = (D·A)⁻¹ · (exp(D·A) − I) · D·B    (discretized input projection)

h_k  = A̅·h_{k-1} + B̅·x_k              (state update)
y_k  = C·h_k                           (readout)
y    = Σ_l  W^l·y^l_k + x_k            (residual merge)
```

A **Blelloch parallel scan** over the state dimension runs by default on CPU
(`use_parallel_scan=1`); set it to `0` for the sequential path. CUDA
kernels live in `algorithms/hss/cuda/hss_cuda.cu`.

```python
from SneppX_ALG import HSSModel, HSSConfig, Tensor

cfg = HSSConfig()
cfg.state_dim   = 16
cfg.input_dim   = 64
cfg.output_dim  = 64
cfg.num_layers  = 4
cfg.seq_len     = 128

model = HSSModel(cfg, seed=42)
x = Tensor.randn((4, cfg.seq_len, cfg.input_dim))   # [batch, seq, dim]
y = model.forward(x)
```

> **Need the C backend?** `HSSModel.forward` delegates to
> `_neural_engine_bridge._HSSModel`. Without it (`_HAS_C_BACKEND is False`),
> `forward` raises `RuntimeError`. Build with `cmake --preset release` and
> reinstall the bindings to enable it.

## 2. SER — Sparse Expert Routing (Mixture of Experts)

SER routes each token to `top_k` of `num_experts` parallel MLP experts, then
combines outputs with normalized routing weights.

```
Router:   r(x) = Softmax(W_r·x + b_r)        W_r ∈ ℝ^{num_experts × d_model}
Top-k:    indices, weights = topk(r(x), k)
Output:   y = Σ_{i=1}^{k} weights_i · E_{indices_i}(x)

Load balancing loss:   L = α · n · Σ_i  f_i · P_i
  f_i = token fraction to expert i ;  P_avg prob of expert i
```

```python
from SneppX_ALG import SERModel, SERConfig, Tensor

cfg = SERConfig()
cfg.num_experts = 8
cfg.num_active  = 2       # top-k
cfg.input_dim   = 64
cfg.expert_dim  = 128
cfg.output_dim  = 64
cfg.load_balance_coef = 0.01

moe = SERModel(cfg, seed=42, num_layers=1)
x = Tensor.randn((2, 16, cfg.input_dim))
out = moe.forward(x)
```

## 3. ARC — Adversarial Robustness Core (Guard)

ARC wraps a stage in **three** defenses and can simulate FGSM/PGD/CW attacks
for adversarial training:

1. **Input guard** — z-score anomaly detection; anomalous samples are projected
   onto a learned subspace before processing.
2. **Gradient obfuscator** — adds magnitude-proportional Gaussian noise and/or
   clamps gradients (`NONE | NOISE | CLAMP | MIXED`).
3. **Output verifier** — cosine-similarity + temporal smoothing against a
   reference output; rejects or sanitizes out-of-distribution outputs.

> See `docs/API.md` and `algorithms/arc/core/layer.c` for the full threat
> model and the `SNEPPXARCConfig` schema.

## 4. NPE — Neural Program Executor (VM)

NPE executes *programs* on a virtual machine: **16 registers**, **32 opcodes**
(MATMUL, ATTENTION, SOFTMAX, LAYERNORM, BRANCH, HALT, …), a program counter,
and a flat tensor memory. Neural subroutines are compiled into bytecode and
run through a JIT pipeline:

```
parse → DCE → constant folding → matmul+relu fusion → register allocation → emit
```

Auto-JIT mode triggers when a hot instruction threshold is crossed; the
verifier checks programs for type/shape safety before execution.

```python
from SneppX_ALG import NPEModel, NPEConfig, Tensor

cfg = NPEConfig()
cfg.max_program_length = 1024
cfg.register_count     = 16
cfg.step_limit         = 1000
cfg.verification_mode  = True
cfg.trace_execution    = True

vm = NPEModel(cfg)
# In practice you build a program via the compiler; here is the shape contract:
vm.load_program([])                 # load bytecode (list of SNEPPXNPEInstruction)
out = vm.run(Tensor.randn((1, 4, 4)))
```

> The public instruction set is documented in `include/neural_core/architecture/neural_programming_engine.h`;
> see `docs/walkthrough-npe.md` for a bytecode walkthrough.

## 5. FM — Federated Memory

FM keeps a per-node memory bank with **Euclidean similarity search** and **LRU
eviction**, then synchronizes banks with trust-weighted all-reduce. A Laplace
differential-privacy noise term is added to the sync scalar, and an NCCL
callback bridge handles GPU all-reduce.

```
sync_all_reduce():  m_i ← (1-α)·m_i + α·Σ_j  w_{ij}·(m_j + noise)
```

```python
from SneppX_ALG import FMModel, FMConfig, Tensor

cfg = FMConfig()
cfg.num_nodes          = 4
cfg.memory_dim         = 64
cfg.memory_capacity    = 1000
cfg.sync_interval      = 10
cfg.privacy_epsilon    = 1.0

fm = FMModel(cfg)
x = Tensor.randn((1, cfg.memory_dim))
y = fm.forward(node_id=0, input=x)
fm.sync_all_reduce()        # commit + sync across the 4-node ring
```

## Data flow & stitching

The five stages are designed to compose: the output `Tensor` of one stage feeds
the next. In the C core this is explicit (`SNEPPX_*_forward` chains in
`algorithms/*/core/`); from Python you compose them with `nn.Sequential`,
`Tensor` operators, and the `nn.TransformerBlock` to build full transformer
backbones that route attention through one or more pipeline stages.

## Where to go next

- **Mathematical detail** per stage → `docs/ARCHITECTURE.md`
- **Python quickstart** → `docs/guide/quickstart.md`
- **C/C++ reference** → `docs/api/c.md` and the generated Doxygen site
  (`doxygen Doxyfile`, output in `docs/api/doxygen/html/`)
