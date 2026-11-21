# Code Walkthrough — The SNEPPX-Alg Pipeline

This is the entry point for reading the SNEPPX-Alg source. It maps the five
pipeline stages to their directories, header APIs, and per-module walkthroughs.

## The five-stage pipeline

SNEPPX-Alg processes data through a fixed pipeline of five algorithm families.
Each stage is a directory under `algorithms/` and a public header under
`include/neural_core/architecture/`.

| Stage | Name | Header | Directory | Job |
|-------|------|--------|-----------|-----|
| 1 | **HSS** — Hierarchical State Space | `hierarchical_state_space.h` | `algorithms/hss/` | Recurrent sequence modeling with a learnable state |
| 2 | **SER** — Sparse Expert Routing | `sparse_expert_routing.h` | `algorithms/ser/` | Mixture-of-experts gating, routing, and load balancing |
| 3 | **ARC** — Adversarial Robustness Certification | `adversarial_robustness_certification.h` | `algorithms/arc/` | Input guard, gradient obfuscation, output verification |
| 4 | **NPE** — Neural Programming Engine | `neural_programming_engine.h` | `algorithms/npe/` | 16-register neural VM with 32 opcodes and a JIT |
| 5 | **FM** — Fractal Memory Orchestrator | `fractal_memory_orchestrator.h` | `algorithms/fm/` | Distributed memory banks, sync, and federated state |

The pipeline is HSS → SER → ARC → NPE → FM: sequence features are extracted,
sparsely routed through experts, adversarially hardened, executed as a
differentiable program, then written into distributed memory.

## Shared foundations

Every stage is built on three kernel subsystems (under `include/neural_core/`
and `kernel/`):

- `multidimensional_tensor_engine.h` — the `SNEPPXTensor` core used by every
  stage as its data type.
- `automatic_differentiation_framework.h` — the `SNEPPXTape` /
  `SNEPPXVariable` autodiff graph used for training.
- `polymorphic_memory_allocator.h` — the memory allocator used for allocation
  in each module's C sources.

## How each module is organized

Every algorithm directory follows the same layout:

- `algorithms/<stage>/core/*.c` — CPU implementation of the stage.
- `algorithms/<stage>/cuda/*.cu` / `*.cuh` — CUDA kernels (built only when
  `-DSNEPPX_BUILD_CUDA=ON`).
- `include/neural_core/architecture/<stage>.h` — the public API.

## Per-module walkthroughs

| Module | Walkthrough |
|--------|-------------|
| HSS | [walkthrough-hss.md](walkthrough-hss.md) |
| SER | [walkthrough-ser.md](walkthrough-ser.md) |
| ARC | [walkthrough-arc.md](walkthrough-arc.md) |
| NPE | [walkthrough-npe.md](walkthrough-npe.md) |
| FM | [walkthrough-fm.md](walkthrough-fm.md) |

## Recommended reading order

1. Start with [walkthrough-hss.md](walkthrough-hss.md) — it introduces the
   tensor, layer, and train-graph patterns that the other four modules reuse.
2. [walkthrough-ser.md](walkthrough-ser.md) — adds gating and routing on top.
3. [walkthrough-arc.md](walkthrough-arc.md) — the security hardening stage.
4. [walkthrough-npe.md](walkthrough-npe.md) — a different execution model
   (a bytecode VM instead of plain tensor layers).
5. [walkthrough-fm.md](walkthrough-fm.md) — memory and distributed sync.

Each walkthrough ends with a "minimal example" that compiles against the
library, and the full public API of the module.
