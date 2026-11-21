# Core Kernel — Computational Substrate

The `kernel/` directory contains the entire computational engine of SNEPPX-ALG: tensor operations,
automatic differentiation, optimization, training, CUDA kernels, attention mechanisms, and more.

## Directory Layout

### `tensor/` — Tensor Engine
SIMD-optimized tensor library with AVX2/AVX-512 GEMM, element-wise ops, reductions,
broadcasting, convolution, pooling, normalization, and tensor expression JIT compilation.

- `tensor.c` — Core tensor operations (create, reshape, slice, concat, split, tile, gather, scatter)
- `tensor_ops_impl.c` — SIMD-optimized implementations (AVX2/AVX-512)
- `tensor_expr.c` — Tensor expression graph compiler
- `simd_gemm.c` — SIMD GEMM kernels (AVX2 fp32, AVX-512 fp32/bf16)

### `autodiff/` — Automatic Differentiation
Reverse-mode autograd engine recording operations on a tape and computing gradients
via backpropagation. Supports 100+ differentiable operations.

### `optimizer/` — Optimizer Suite
14 optimizers: SGD, Adam(W), Lion, LAMB, LARS, AdaFactor, RAdam, Sophia, Adan,
ScheduleFreeAdamW, SM3, Demeter, CaProp, SOAP, DistributedAdam, OrthoAdam.
Each with fused CUDA kernel variants.

### `train/` — Training Pipeline
Complete training loop: `SNEPPXTrainer` with train_step, evaluate, checkpoint save/load,
gradient accumulation, gradient scaling, gradient clipping.

### `cuda/` — CUDA Kernels
NVIDIA GPU acceleration (~6,900 lines):

| File | Purpose |
|------|---------|
| `common.cuh` | Architecture detection, tile config, FP16/BF16, MMA/WGMMA, cp.async, warp shuffle, Philox RNG, CUBLAS TLS |
| `tensor_cuda.h/.cu` | Tensor-core GEMM (128x128 tiling), element-wise, reduction, layernorm, softmax, CUBLAS fallback |
| `attention_cuda.h/.cu` | Flash Attention v2/v3, GQA, paged attention, KV cache, RoPE |
| `autodiff_cuda.h/.cu` | GEMM backward, activation backward, layernorm backward, convolution backward |
| `optim_cuda.h/.cu` | Fused AdamW, SGD+momentum, Lion, LAMB, LARS, AdaFactor; ZeRO-1; in-kernel LR scheduling |
| `memory_cuda.h/.cu` | Memory pool, stream pool, event pool, pinned/managed memory |
| `rng_cuda.h/.cu` | Philox RNG, Xavier/Kaiming init, fused dropout, batch RNG |

### `attention/` — Attention Mechanisms
- `differential_attention.c` — Differential Attention (λ-scaled QK pairs), Multi-head Latent Attention (DeepSeek MLA)
- `flex_attention.c` — FlexAttention (block-sparse + mask modulation), multi-modal cross-attention, Mixture of Depth
- `mamba2.c` — Mamba-2 selective SSM with HiPPO initialization, 1D convolution, discretized scan

### `arch/` — Advanced Architectures
Unified header for DifferentialAttn, MLA, FlexAttn, Mamba2, MoD, gated activations (SwiGLU/GeGLU/ReGLU),
position encodings (YaRN NTK-RoPE, ALiBi).

### `quantization/` — Model Compression
- `quantize_int8.c` — INT8 symmetric/asymmetric, per-channel; INT4 packed
- `quantize_fp8.c` — FP8 E4M3/E5M2 bit-level encode/decode
- `awq.c` — AWQ optimal scale grid search
- `gptq.c` — GPTQ Hessian-based quantization

### `distributed/` — Distributed Training
- `zero.c` — ZeRO-1/2/3 optimizer state partitioning
- `pipeline.c` — 1F1B schedule with microbatches
- `tensor_parallel.c` — Row/column split linear
- `expert_parallel.c` — Expert-to-rank mapping, all-to-all dispatch
- `ddp.c` — Bucket-based gradient all-reduce
- `gradient_comm.c` — Hierarchical all-reduce, Top-K compression
- `checkpoint.c` — Async distributed checkpoint, fault tolerance
- `heartbeat.c` — UDP heartbeat, SUSPECT→DEAD state machine
- `elastic.c` — Elastic training (join/leave/failure/reconfigure)
- `distributed_sampler.c` — Sharded data loading, Fisher-Yates shuffle

### `data/` — Data Pipeline
Dataset, TensorDataset, DataLoader, DistributedSampler.

### `inference/` — Inference Engine
Optimized inference with KV cache, continuous batching, prefix caching.

### `model_zoo/` — Model Zoo
Config presets and weight name mappings for LLaMA 2/3, Mistral, Qwen2, DeepSeek V2.
JSON serialize/parse, `from_pretrained()` workflow.

### `thread/` — Thread Management
Work-stealing thread pool for parallel task dispatch.

### `memory/` — Memory Management
Polymorphic memory allocator with pool, TLS cache, stats tracking.

### `tokenizer/` — Tokenizer
Embedding lookup, tokenization utilities.

### `activations/` — Activation Functions
GELU, SiLU, ReLU variants, Tanh, Sigmoid, SwiGLU.

### `position/` — Position Encodings
RoPE, YaRN NTK-aware RoPE, ALiBi.

### `profiler.c` — Profiling Infrastructure
Kernel timing, JSON export, NVTX markers, range stack.

### `logger.c` — Structured Logging
6 severity levels, JSON/color stdout, per-rank logging.

## Build Targets

| Target | Description |
|--------|-------------|
| `neural_core_kernel` | Core tensor/memory/trainer library (C) |
| `neural_cuda_kernels` | CUDA kernels (requires `-DSNEPPX_BUILD_CUDA=ON`) |

## Key Types

All public types use the `SNEPPX_` prefix:

- `SNEPPXTensor` — Multi-dimensional array
- `SNEPPXVariable` — Autograd variable wrapping a tensor
- `SNEPPXTape` — Gradient tape for reverse-mode AD
- `SNEPPXOptimizer` — Optimizer state
- `SNEPPXModel` — Neural network model
- `SNEPPXTrainer` — Training pipeline
