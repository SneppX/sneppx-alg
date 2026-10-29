<div class="sx-hero" markdown>

# The Universal Open-Source AI Algorithm

Secure, composable, production-grade AI in C/C++ & CUDA — tensor engine,
autograd, MoE/SSM transformers, quantization, distributed training, and 10
security layers (S0–S9). MIT licensed.

<div class="sx-hero-art" role="img" aria-label="SNEPPX pipeline: HSS to SER to ARC to NPE to FM, observed by security layers S0-S9">
<svg viewBox="0 0 640 110" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
<defs>
<marker id="sxarrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" class="sx-art-arrow"/></marker>
</defs>
<g class="sx-art-node">
<rect x="8" y="34" width="104" height="42" rx="10"/>
<rect x="136" y="34" width="104" height="42" rx="10"/>
<rect x="264" y="34" width="104" height="42" rx="10"/>
<rect x="392" y="34" width="104" height="42" rx="10"/>
<rect x="520" y="34" width="104" height="42" rx="10"/>
</g>
<g class="sx-art-label">
<text x="60" y="60" text-anchor="middle">HSS</text>
<text x="188" y="60" text-anchor="middle">SER</text>
<text x="316" y="60" text-anchor="middle">ARC</text>
<text x="444" y="60" text-anchor="middle">NPE</text>
<text x="572" y="60" text-anchor="middle">FM</text>
</g>
<g class="sx-art-edge">
<line x1="112" y1="55" x2="132" y2="55" marker-end="url(#sxarrow)"/>
<line x1="240" y1="55" x2="260" y2="55" marker-end="url(#sxarrow)"/>
<line x1="368" y1="55" x2="388" y2="55" marker-end="url(#sxarrow)"/>
<line x1="496" y1="55" x2="516" y2="55" marker-end="url(#sxarrow)"/>
</g>
<g class="sx-art-sec">
<path d="M320 4 l14 6 v10 c0 9 -7 15 -14 18 c-7 -3 -14 -9 -14 -18 v-10 z"/>
<text x="320" y="24" text-anchor="middle">S0&#8211;S9</text>
</g>
</svg>
</div>

<div class="sx-stats" markdown>

<span class="sx-stat">
<span class="sx-stat-value" data-count="253" data-suffix="K+">253K+</span>
<span class="sx-stat-label">Lines of code</span>
</span>

<span class="sx-stat">
<span class="sx-stat-value" data-count="300" data-suffix="+">300+</span>
<span class="sx-stat-label">Python tests</span>
</span>

<span class="sx-stat">
<span class="sx-stat-value" data-count="40" data-suffix="+">40+</span>
<span class="sx-stat-label">Autograd ops</span>
</span>

<span class="sx-stat">
<span class="sx-stat-value" data-count="10">10</span>
<span class="sx-stat-label">Security layers</span>
</span>

</div>

<div class="sx-cards" markdown>

<div class="sx-card" markdown>
### 5-Stage Pipeline
HSS → SER → ARC → NPE → FM. Each stage independently testable.
[Pipeline overview →](architecture/5-pipeline-architecture.md)
</div>

<div class="sx-card" markdown>
### Security S0–S9
Post-quantum crypto (Kyber/Dilithium/SPHINCS+), memory hardening, signed updates.
[Security layers →](security_layers.md)
</div>

<div class="sx-card" markdown>
### Python API
Fully differentiable `nn`, higher-order autograd, functional API, 14 extra packages.
[Python API →](api/python.md)
</div>

<div class="sx-card" markdown>
### v1.5.0
Every `nn` module now trains. Higher-order autograd. New ecosystem packages.
[Release notes →](release_notes/v1.5.0.md)
</div>

</div>

<p align="center">
<a href="https://pypi.org/project/sneppx-alg/"><img src="https://img.shields.io/pypi/v/sneppx-alg?color=blue" alt="PyPI"></a>
<a href="https://github.com/SneppX/sneppx-alg/stargazers"><img src="https://img.shields.io/github/stars/SneppX/sneppx-alg?style=social" alt="Stars"></a>
<a href="https://github.com/SneppX/sneppx-alg/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT"></a>
<a href="https://github.com/SneppX/sneppx-alg/commits/main"><img src="https://img.shields.io/github/last-commit/SneppX/sneppx-alg" alt="Last commit"></a>
</p>

</div>

## Overview

SneppX-ALG is a composable 5-component AI algorithm pipeline wrapped in 10 security layers. v1.5.0 delivers a production-stable pipeline with a fully differentiable `nn` stack (every layer trains), higher-order autograd, a PyTorch-style functional API, sparse tensors, and fourteen extended Python packages — alongside the Model Zoo, Model Hub (`sneppx-hub`), C HTTP serving control plane, distributed training (ZeRO-1/2/3, pipeline/tensor/expert parallelism), advanced architectures (Differential Attention, Mamba-2, FlexAttention, MoD), quantization (INT8/INT4/FP8/AWQ/GPTQ), async checkpointing, profiling & debugging tools, and weight converters for LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2.

## Quickstart

### Build from source

```powershell
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### Run a demo

```powershell
build\tests\Release\test_hss.exe
build\tests\Release\test_ser.exe
build\tests\Release\test_arc.exe
build\tests\Release\test_npe.exe
build\tests\Release\test_fm.exe
```

### Python quickstart

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "from SneppX_ALG import *; print('ok')"
```

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │         Security Layer (S0-S9)        │
                     │  Crypto · Secure Mem · Obfuscation    │
                     │  Monitor · Network · AI Sanitizer     │
                     │  Key Vault · Updates · Formal · Pentest│
                     └──────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │         Algorithm Pipeline            │
                     │  HSS → SER → ARC → NPE → FM          │
                     │  (SSM) (MoE) (Guard) (VM)  (Fed Mem) │
                     └──────────────────────────────────────┘
                                     │
                      ┌───────────────▼──────────────────────┐
                      │         Integrity Layer               │
                      │  ZK Proofs · Formal Safety ·          │
                      │  On-Device Attestation                │
                      └──────────────────────────────────────┘
```

## Component Descriptions

### HSS — Hierarchical State Space

Multi-layer state space model with zero-order hold discretization. Blelloch parallel scan over state dimension enabled by default on CPU. Supports hierarchical decomposition for long-range dependencies. Training through the autodiff tape is supported (see `docs/hss_training.md`).

- `docs/ARCHITECTURE.md` for mathematical details
- `algorithms/hss/core/` for implementation
- `tests/unit/hss/` for tests

### SER — Sparse Expert Routing

Softmax-based routing with top-k selection. Each input token is routed to k experts, and the outputs are combined via weighted sum. Includes learned MLP gating (2-layer MLP per expert) with autodiff subgraph, plus load-balancing loss to prevent expert collapse.

- `docs/ARCHITECTURE.md` for routing details
- `algorithms/ser/core/` for implementation
- `tests/unit/ser/` for tests

### ARC — Adversarial Robustness Core

Three-layer defense: input guard (z-score anomaly detection), gradient obfuscation (noise + clamping), output verifier (cosine similarity + temporal smoothing). Includes attack simulation (FGSM, PGD, CW) and adversarial training graph builder that constructs shared-weight clean+adversarial forward graphs.

- `docs/ARCHITECTURE.md` for threat model
- `algorithms/arc/core/` for implementation
- `tests/unit/arc/` for tests

### NPE — Neural Program Executor

16-register virtual machine with 15+ opcodes (MATMUL, ATTENTION, SOFTMAX, LAYERNORM, fused MATMUL+ADD+RELU, etc.). Includes JIT pipeline (DCE → constant folding → fusion → compilation), auto-JIT mode in VM (hot threshold triggers optimization), and a static verifier for program correctness.

- `docs/ARCHITECTURE.md` for instruction set and JIT passes
- `algorithms/npe/core/` for implementation
- `tests/unit/npe/` for tests

### FM — Federated Memory

Per-node memory banks with euclidean similarity search and LRU eviction. Supports trust-weighted all-reduce synchronization with Laplace differential privacy noise. Includes NCCL sync bridge with callback pattern.

- `docs/ARCHITECTURE.md` for sync protocols
- `algorithms/fm/core/` for implementation
- `tests/unit/fm/` for tests

### Trainer — Training Loop

CPU training loop with MSE loss, tape-based autodiff backward, and optimizer step. Optional CUDA-accelerated optimizer: SGD via pure cuBLAS on GPU, AdamW via cuBLAS + host fallback. Configurable via `use_cuda_optimizer` flag.

- `kernel/train/trainer.c` for CPU loop
- `kernel/train/trainer_cuda.c` for CUDA bridge
- `tests/unit/train/` for tests

## Status Table

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| Tensor Core | ~2,500 | 57+27 edge | ✅ Real |
| Memory | ~800 | 13 | ✅ Real |
| Thread Pool | ~300 | 11 | ✅ Real |
| Autodiff | ~600 | tape backward | ✅ Real (layer-norm fixed) |
| Optimizer (CPU) | ~400 | 1 | ✅ Real (Adam/SGD/RMSprop/Adagrad) |
| Optimizer (CUDA) | ~1,100 | 4 | ✅ Real (cuBLAS SGD, AdamW) |
| HSS | ~700 | 2 + integration | ✅ Real (training, parallel scan) |
| SER | ~800 | 6 | ✅ Real (MLP gater + autodiff) |
| ARC | ~700 | 6 | ✅ Real (FGSM/PGD/CW + adversarial train graph) |
| NPE | ~1,200 | 10 | ✅ Real (JIT pipeline, auto-JIT, fused ops) |
| FM | ~700 | 5 | ✅ Real (NCCL sync bridge) |
| Trainer (CPU) | ~500 | 3 | ✅ Real |
| Trainer (CUDA) | ~400 | 4 | ✅ Real (device detection, async transfers) |
| Python API | ~800 | 11 | ✅ Real (ARC/NPE/FM/Trainer wrappers) |
| Model Zoo | ~1,200 | 49 | ✅ Real (from_pretrained, weight converters) |
| Distributed | ~1,200 | 44 | ✅ Real (ZeRO-1/2/3, TP/PP/EP, elastic) |
| Quantization | ~1,300 | 17 | ✅ Real (INT8/INT4/FP8/AWQ/GPTQ) |
| Advanced Arch | ~500 | 20 | ✅ Real (DifferentialAttn, Mamba-2, FlexAttn, MoD) |
| Async Checkpoint | ~500 | 23 | ✅ Real (fault tolerance, elastic) |
| Profiling | ~300 | 13 | ✅ Real (profiler, logger, sanitizers) |
| CUDA Backend | ~9,600 | 25 | ✅ Real (FlashAttn, fused optim, ZeRO) |
| S0 Crypto | ~2,500 | 10 | ✅ Real |
| S1 Secure Mem | ~800 | 3 | ✅ Real |
| S2 Obfuscation | ~1,500 | 4 | ✅ Complete |
| S3 Monitor | ~100 | 0 | ✅ Complete |
| S4-S9 | ~3,000 | 5 | ✅ Complete |

## Directory Structure

```
SneppX_ALG/
├── CMakeLists.txt
├── CMakePresets.json
├── include/neural_core/        # Public headers
│   ├── kernel/                  # Tensor, memory, autodiff, optimizer, trainer
│   ├── architecture/            # Algorithm configs and API
│   └── security/                # S0-S9 security headers
├── kernel/                      # Core implementations
│   ├── tensor/                  # Tensor ops, IO, format readers
│   ├── autodiff/                # Autodiff tape and gradient ops
│   ├── train/                   # Trainer and CUDA bridge
│   ├── optimizer/               # SGD, Adam/W, RMSprop, Adagrad
│   ├── attention/               # Flash Attention v2/v3, RoPE, KV-cache
│   ├── arch/                    # Advanced architectures
│   ├── distributed/             # DDP, ZeRO, pipeline, tensor/ep parallel
│   ├── quantization/            # INT8/FP8/AWQ/GPTQ quantization
│   └── cuda/                    # CUDA kernels (opt-in)
├── algorithms/                  # Algorithm pipeline implementations
│   ├── hss/core/                # HSS forward, training, CUDA kernels
│   ├── ser/core/                # SER routing, MLP gater, CUDA kernels
│   ├── arc/core/                # ARC defense, attacks, adversarial train graph
│   ├── npe/core/                # NPE VM, compiler, JIT pipeline, CUDA kernels
│   └── fm/core/                 # FM memory bank, NCCL sync, CUDA kernels
├── security/                    # S0-S9 security layer source
│   ├── crypto/                  # S0 — Cryptographic Core
│   ├── memory/                  # S1 — Secure Memory
│   ├── cpp/                     # S2 — Obfuscation Engine (+ S3 monitor)
│   ├── network/                 # S4 — Network Security
│   ├── ai/                      # S5 — AI Sanitizer
│   ├── ui/                      # S6 — Security UI / Key Vault
│   ├── updates/                 # S7 — Secure Updates
│   ├── formal/                  # S8 — Formal Verification
│   ├── pentest/                 # S9 — Penetration Testing
│   ├── fuzzing/                 # Fuzz harnesses and corpora
│   └── asm/                     # x86_64 MASM helpers
├── tests/
│   ├── unit/                    # Per-component unit tests
│   ├── integration/             # Multi-component integration tests
│   ├── benchmark/               # Performance benchmarks
│   ├── security/                # S0-S9 security tests
│   ├── python/                  # Python wrapper tests
│   └── quantization/            # Quantization tests
├── bindings/python/             # Python bindings
│   └── SneppX_ALG/interface_bindings/  # Per-algorithm Python wrappers
├── examples/                    # Demo programs
├── docs/                        # Documentation
├── scripts/                     # Build and release scripts
├── cmake/                       # CMake modules
└── config/                      # Model zoo configs
```

## Documentation Standards

This project uses a **four-layer commenting standard** defined in [COMMENTING.md](COMMENTING.md):

1. **Layer 1** — File header blocks (WHAT/CONCEPT/ROLE/REFERENCES) on every source file
2. **Layer 2** — Concept blocks before non-obvious algorithms
3. **Layer 3** — Inline "why" comments for non-obvious decisions
4. **Layer 4** — Doxygen `@brief/@param/@return` on every public `SNEPPX_*` function

The `sneppx-format` linter enforces Layers 1 and 4 via `--docs` flag. All contributors must follow these conventions when modifying or adding source files. See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code formatting rules and [COMMENTING.md](COMMENTING.md) for the full commenting standard.

- **C/C++ source**: ~25,000 lines across all components
- **Tests**: 100+ registered (all pass except 2 pre-existing S0 edge cases)
- **Build time**: ~30s on modern hardware (Release, 8 cores)
- **Dependencies**: None for C core. Python stdlib-only for Python wrappers

## Model Zoo (v1.0.0)

Load pre-trained models from Hugging Face with one call:

```python
from SneppX_ALG.model_zoo import from_pretrained

model = from_pretrained("meta-llama/Llama-2-7b-hf")
```

Supported architectures: LLaMA 2 (7B/13B/70B), LLaMA 3 (8B/70B), Mistral (7B), Qwen 2 (7B/72B), DeepSeek V2 (Lite/Full). See [`docs/guide/model_zoo.md`](guide/model_zoo.md) for the full guide.

## ONNX import/export (v1.2.0)

Standalone numpy-only ONNX toolkit (`import onnx`, also exposed as `SneppX_ALG.onnx`):

```python
import numpy as np
import onnx

graph = onnx.build_graph(
    name="mlp",
    inputs=[onnx.ValueInfo("x", "float32", ["batch", 4])],
    outputs=[onnx.ValueInfo("y", "float32", ["batch", 2])],
    initializers={"W1": np.random.randn(4, 8).astype(np.float32), ...},
    nodes=[
        onnx.Node("MatMul", ["x", "W1"], ["mm1"]),
        onnx.Node("Add", ["mm1", "b1"], ["a1"]),
        onnx.Node("Relu", ["a1"], ["r1"]),
        onnx.Node("MatMul", ["r1", "W2"], ["mm2"]),
        onnx.Node("Add", ["mm2", "b2"], ["y"]),
    ],
)
onnx.save_model(onnx.Model(graph, producer_name="SNEPPX"), "model.onnx")
m = onnx.load_model("model.onnx")          # parse
ok, errors = onnx.check_model(m)           # validate
shapes = onnx.infer_shapes(m)              # shape inference
m2 = onnx.optimize(m)                      # const-fold + DCE
q = onnx.quantize_model(m)                 # QDQ insert
out = onnx.Session(m).run({"x": x})[0]     # numpy executor
```

- `sneppx-onnx` CLI: `info`, `check`, `shapes`, `optimize`, `convert`, `quantize`, `run`, `save-external`, `load-external`
- `OnnxRuntimeSession` runs through onnxruntime when installed (falls back to the numpy executor)
- `onnx/exporter.py` bridges to the legacy `interface_bindings` exporter; wire format is byte-compatible both ways
- Tests: `onnx/tests/test_onnx.py` (25 tests, numpy-only)

Full guide: [`docs/guide/onnx.md`](guide/onnx.md)

## Context Extension (v1.0.0)

Extend any supported model to 128K context via NTK-aware RoPE scaling:

```python
cfg = LLMConfig.from_name("llama3", "8B")
cfg.extend_context(131072)
```

See [`docs/guide/128k_context_extension.md`](guide/128k_context_extension.md).

## Multi-Head Attention Forward (v1.0.0)

Full MHA forward pass with GQA, Flash Attention, RoPE, and MLA (DeepSeek V2):

```python
output = cfg.forward_mha(hidden_states, attention_mask, position_ids)
```

See [`docs/guide/mha_forward_pass.md`](guide/mha_forward_pass.md).

## Contribution Framework (v1.1.0)

SNEPPX-Algo uses a **five-tier merit system** with a track-based Git Flow branching model.

| Document | Purpose |
|----------|---------|
| [CONTRIBUTOR_TIERS.md](CONTRIBUTOR_TIERS.md) | Tier definitions, gates, badges, promotion/demotion |
| [BRANCHING_STRATEGY.md](BRANCHING_STRATEGY.md) | Branch types, naming conventions, tier→branch mapping |
| [CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md) | Review levels (L1–L3), per-subsystem criteria, review process |
| [LEARNING_PATHS.md](LEARNING_PATHS.md) | Track-specific month-by-month roadmaps |
| [PR_TEMPLATE.md](PR_TEMPLATE.md) | Pull request submission template |
| [CONTRIBUTING.md](https://github.com/SneppX/sneppx-alg/blob/main/../CONTRIBUTING.md) | Contribution workflow and acceptance criteria |
| [GOVERNANCE.md](https://github.com/SneppX/sneppx-alg/blob/main/../GOVERNANCE.md) | BDFL + Maintainer Council governance |
| [MAINTAINERS.md](https://github.com/SneppX/sneppx-alg/blob/main/../MAINTAINERS.md) | Tier tracking and maintainer table |

## Reference Docs

| Document | Description |
|----------|-------------|
| [architecture.md](https://github.com/SneppX/sneppx-alg/blob/main/architecture.md) | System layers, data flow, memory model, distributed architecture |
| [build.md](build.md) | Build prerequisites, options, targets, troubleshooting |
| [api_quickref.md](api_quickref.md) | C and Python API quick reference with enums |
| [API_REFERENCE.md](API_REFERENCE.md) | Full REST API reference for the inference server (21 endpoints) |
| [security_layers.md](security_layers.md) | S0–S9 security deep dive with threat model |

## Next Steps

- Read [docs/guide/index.md](guide/index.md) for feature guides
- Read [docs/ARCHITECTURE.md](ARCHITECTURE.md) for deep technical details
- Read [docs/hss_training.md](hss_training.md) for an end-to-end HSS training walkthrough
- Read [docs/API.md](API.md) for the full C and Python API reference
- Read [docs/API_REFERENCE.md](API_REFERENCE.md) for the complete REST API reference (21 endpoints)
- Read [docs/installation.md](installation.md) for platform-specific build guides
- Read [docs/ROADMAP.md](ROADMAP.md) for the project timeline

























































































































































































  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 



























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































