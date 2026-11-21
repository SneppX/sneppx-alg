# Algorithms — Neural Architecture Extensions

The `algorithms/` directory contains five advanced neural architecture algorithms that extend
the core SNEPPX tensor engine with state-space models, mixture-of-experts, adversarial robustness,
neural program execution, and fractal memory orchestration.

## HSS — Hierarchical State Space

Based on structured state-space sequence models (S4 / Mamba). Implements multi-resolution
temporal processing with learnable timescale discretization.

```
HSSConfig {
    state_dim, input_dim, output_dim,
    num_layers, seq_len,
    dt_min, dt_max,
    use_hierarchical       # Enable hierarchical (multi-resolution) mode
}
```

**Key files:**
- `hss.h / hss.c` — Core HSS model: selective scan, S4 forward, HiPPO matrix init, SSM convolution
- `hss_kernels.c` — Hierarchical softmax, sparse-dense matmul, top-k, Blelloch parallel prefix scan
- `cuda/hss_cuda_kernels.cu` — CUDA selective scan, SSM convolution

**CUDA kernels:** Selective scan (Mamba/S6), S4 forward, HiPPO matrix init, SSM convolution,
hierarchical softmax, sparse-dense matmul, top-k, parallel prefix scan (Blelloch).

---

## SER — Sparse Expert Routing

Mixture-of-Experts (MoE) implementation with top-k gating, load balancing, and expert parallelism.

```
SERConfig {
    num_experts, num_active,
    input_dim, expert_dim, output_dim,
    top_k_method,             # GREEDY or NOISY
    load_balance_coef,
    dropout_rate
}
```

**Key files:**
- `ser.h / ser.c` — SER model: top-k gating (softmax + selection), MoE dispatch/combine
- `ser_kernels.c` — Load balancing loss, auxiliary loss computation
- `cuda/ser_cuda_kernels.cu` — Fused MoE forward, expert all-to-all

**CUDA kernels:** Top-k gating, MoE dispatch/combine, load balancing loss, expert all-to-all, fused MoE forward.

---

## ARC — Adversarial Robustness Certification

Adversarial attack simulation and defense mechanisms for gradient obfuscation and input validation.

```
ARCConfig {
    input_guard_strength,
    gradient_obfuscation_method,   # NONE, NOISE, CLAMP, or MIXED
    gradient_noise_scale,
    gradient_clip_max,
    output_verify_layers,
    output_verify_threshold,
    adversarial_training,
    attack_simulation_types        # FGSM, PGD, or CW
}
```

**Key files:**
- `arc.h / arc.c` — ARC layer: PGD/FGSM adversarial attacks, gradient obfuscation, randomized smoothing
- `cuda/arc_cuda_kernels.cu` — PGD/FGSM CUDA attack kernels

**CUDA kernels:** PGD/FGSM adversarial attacks, gradient obfuscation, randomized smoothing.

---

## NPE — Neural Programming Engine

A differentiable neural computer that executes learned programs as sequences of tensor operations.

```
NPEConfig {
    max_program_length, register_count,
    step_limit, verification_mode,
    trace_execution
}
```

**Supported opcodes:** NOP, LOAD, STORE, ADD, SUB, MUL, DIV, MATMUL, RELU, SOFTMAX,
LAYERNORM, ATTENTION, BRANCH, HALT, EXP, LOG, SQRT, POW, SIN, COS, TANH, SIGMOID,
GELU, SILU, DROPOUT, CONV2D, POOL2D, BATCHNORM, EMBEDDING, CROSSENTROPY, MSE, CONCAT, SPLIT.

**Key files:**
- `npe.h / npe.c` — NPE VM: instruction dispatch, program execution, program compiler
- `cuda/npe_cuda_kernels.cu` — Instruction dispatch kernel

**CUDA kernels:** Neural VM instruction dispatch, differentiable program execution.

---

## FM — Fractal Memory Orchestrator

Distributed memory system for federated learning and multi-node synchronization
with privacy-preserving gradient compression.

```
FMConfig {
    num_nodes, memory_dim, memory_capacity,
    sync_interval, sync_method,       # ALL_REDUCE, GOSSIP, or TOPOLOGY
    compression_ratio, privacy_epsilon,
    catastrophic_forgetting_protection,
    ewm_alpha
}
```

**Key files:**
- `fm.h / fm.c` — FM controller: forward pass, sync methods (all-reduce, gossip, topology-based)
- `cuda/fm_cuda_kernels.cu` — Ring/butterfly all-reduce, gradient quantization

**CUDA kernels:** Ring/butterfly all-reduce, gradient quantization (FP16→INT8), Top-K sparsification,
federated averaging, memory bank sync.

---

## Build Target

| Target | Libraries |
|--------|-----------|
| `neural_architecture_layer` | All five algorithm libraries linked together |

All algorithms expose C API via `SNEPPX_`-prefixed functions and have Python bindings
in `bindings/python/SneppX_ALG/interface_bindings/`.
