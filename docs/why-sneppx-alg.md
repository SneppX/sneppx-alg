---
hide:
  - toc
---

# Why SNEPPX-Alg?

SNEPPX-Alg is a universal, open-source AI algorithm framework written in **C11 / C++20 with CUDA**.
It organizes model construction and inference as a composable, secure **5-stage pipeline**
(HSS → SER → ARC → NPE → FM) and wraps it with ten security layers (S0–S9).

This page is a **honest** comparison, not a marketing pitch. It lists what SNEPPX-Alg does
today, where it is competitive, and where it is still catching up to the incumbents.

## Feature matrix

| Capability | SNEPPX-Alg | PyTorch | TensorFlow | JAX |
|---|---|---|---|---|
| Autograd (forward + reverse) | ✅ 40+ ops, higher-order `create_graph` | ✅ | ✅ | ✅ |
| Autograd forward mode (`jvp`/`vjp`) | ✅ (reverse-of-reverse) | ✅ | ✅ | ✅ |
| Mixture-of-Experts routing (SER) | ✅ first-class | ➖ (via add-ons) | ➖ | ➖ |
| State-Space Models (HSS) | ✅ first-class | ➖ | ➖ | ➖ |
| Adversarial guard (ARC) | ✅ first-class | ➖ | ➖ | ➖ |
| Neural VM (NPE) | ✅ 16-reg / 32-opcode | ➖ | ➖ | ➖ |
| Fractal memory (FM) | ✅ first-class | ➖ | ➖ | ➖ |
| Quantization (INT8/INT4/FP8/AWQ/GPTQ) | ✅ | ✅ | ✅ | ➖ |
| Distributed training (ZeRO/TP/PP/EP) | 🟡 wired, CPU-fallback collectives | ✅ | ✅ | ✅ |
| Post-quantum crypto (Kyber/Dilithium/SPHINCS+) | 🟡 (not FIPS-KAT validated) | ➖ | ➖ | ➖ |
| Signed updates (S7) | 🟡 Ed25519, dev fallback | ➖ | ➖ | ➖ |
| Formal verification (S8) | 🟡 bounded safety LTL | ➖ | ➖ | ➖ |
| Model zoo (LLaMA/Mistral/Qwen/DeepSeek) | 🟡 config + skeleton, weights in progress | ✅ | ✅ | ✅ |
| C/CUDA native core | ✅ C11/C++20 + CUDA | ✅ | ✅ | ➖ |
| Python API | ✅ NumPy-backed | ✅ | ✅ | ✅ |
| Serving / HTTP REST | 🟡 demo scaffold | ✅ | ✅ | ➖ |
| Mobile / edge runtime | 🟡 | ➖ | ✅ | ➖ |

✅ = shipped and usable · 🟡 = shipped but partial/experimental · ➖ = not a core feature

## Where SNEPPX-Alg genuinely differs

1. **Security is a first-class pipeline, not a bolt-on.** Ten layers (S0–S9) — classical + post-quantum
   crypto, memory hardening, obfuscation, monitoring, network, AI sanitizer, key vault, signed updates,
   formal verification, and pentesting — observe every stage of the 5-stage pipeline. Incumbents leave
   this to you.
2. **MoE, SSM, adversarial guard, neural VM, and fractal memory are primitives**, not libraries
   you stitch together. HSS/SER/ARC/NPE/FM ship as composable, testable stages.
3. **The hot path is readable C/C++/CUDA** you can audit and verify — no opaque runtime.
4. **Open and extensible** — opt-in backends (Vulkan/TPU/HTTP/ZK/Metal/oneAPI), a model zoo, and a
   Python API on top of the native core. MIT licensed.

## Where SNEPPX-Alg is still catching up

Be honest with yourself before choosing it:

- **Performance.** The public Python `Tensor`/`nn` API is NumPy-backed (CPU). The real speed lives in
  the C/CUDA core, which is not yet wired into the public Python path end-to-end. For training at scale,
  PyTorch/JAX are ahead today.
- **Ecosystem.** No autograd-on-GPU in Python, no `torchvision`-equivalent, no HuggingFace Hub
  integration for weights (in progress).
- **Maturity.** Distributed collectives run on a CPU fallback path; the post-quantum primitives are
  not yet FIPS-KAT validated; signed updates and formal verification are implemented but bounded.
- **Tooling.** No profiler-grade tooling, no ONNX export parity, no mobile SDK yet.

## When to choose SNEPPX-Alg

- You want an **auditable, composable pipeline** (HSS/SER/ARC/NPE/FM) you can read end-to-end.
- You care about **AI security as a first-class concern** (PQC, signed updates, formal verification).
- You're building **research prototypes** where MoE/SSM/adversarial primitives save you glue code.
- You want a **small, MIT-licensed native core** you can embed or fork.

## When to choose an incumbent

- You need **GPU training today** at scale → PyTorch / JAX.
- You need **production serving, mobile, or edge** → TensorFlow / PyTorch.
- You need **the ecosystem** (Hub, datasets, pretrained weights) → PyTorch / HuggingFace.

## See also

- [Benchmarks](BENCHMARKS.md) — measured numbers (where available).
- [Framework comparison](comparison/ml_framework_feature_gap_analysis.md) — detailed gap analysis.
- [Architecture overview](ARCHITECTURE.md) — the 5-stage pipeline in depth.
- [Security layers](security_layers.md) — S0–S9 mapping.
- [Quick Start](guide/quickstart.md) — get running in five minutes.