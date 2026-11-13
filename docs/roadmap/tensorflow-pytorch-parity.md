# TensorFlow and PyTorch Feature Parity Roadmap

## Goal and compatibility policy

The long-term goal is a coherent, documented tensor and machine-learning framework
covering the major capabilities exposed by TensorFlow/Keras and PyTorch. This is
a compatibility and capability target, not a claim that SNEPPX already implements
either framework in full.

The project will prioritize a stable native tensor/autograd core, then build
higher-level APIs on top of it. API names alone do not count as compatibility:
semantics, gradients, shape rules, dtype behavior, device behavior, serialization,
and error handling must be tested.

## Current baseline

The repository already contains a substantial but uneven foundation:

- A C tensor engine with creation, shape, arithmetic, reduction, linear algebra,
  neural-network operations, and binary I/O APIs.
- A Python Tensor facade with NumPy-backed fallback storage and optional native
  bindings.
- Reverse-mode autograd, a growing collection of differentiable operations,
  neural-network modules, and a functional API with many PyTorch-like names.
- Optimizers, learning-rate schedulers, data loaders/datasets, sparse tensor
  support, CUDA-related code, distributed-training components, ONNX and
  SafeTensors-related utilities.
- A Keras-style compatibility layer and a separate PyTorch-style API.

These surfaces are not yet equivalent to TensorFlow or PyTorch. Some backends
are experimental, some operations have partial gradient coverage, and several
accelerator/distributed paths are optional or incomplete.

## Recent correctness work

The tensor foundation is being hardened before broader API expansion:

- Python fallback dtype, device and dense-layout enum values now follow the
  native C ABI.
- The Python tensor supports NumPy's array protocol with explicit copy behavior,
  complex scalar construction and scalar-only `item()` semantics.
- Native tensor creation checks invalid dtypes and allocation-size overflow.
- Typed element access validates null pointers and indices; strided views validate
  storage bounds, and narrow views no longer have a fixed 16-dimension stack limit.
- Fill operations traverse logical indices using view strides rather than assuming
  contiguous storage.
- Added differentiable `flip`, `roll`, `tril`, `triu`, `index_select`, `gather`
  and `scatter_add` tensor operations with forward and reverse-mode gradient tests.
  Duplicate index selections and scatter destinations accumulate gradients;
  invalid index dtype, bounds, dtype mismatches and non-gather dimensions are rejected.
- Added PyTorch-style TensorDataset, Subset and ConcatDataset composition helpers.
- Expanded Python DataLoader behavior with seeded shuffling, drop-last batches,
  custom collation and ordered thread-pool fetching. Multiprocessing workers,
  prefetch queues, pinned memory and distributed sharding remain planned.
- Added validation for ArrayDataLoader batch dimensions and batch-index inputs.
- FLOAT8 remains intentionally limited: the repository does not yet select one
  encoding (E4M3 or E5M2), so operations that need a numeric one-value encoding
  fail closed instead of guessing.

## Capability matrix

| Area | Existing foundation | Required work before parity |
|---|---|---|
| Tensor creation and indexing | C and Python tensor APIs | Complete dtype/device semantics, advanced indexing, views, aliasing rules, factories and edge cases |
| Broadcasting and arithmetic | Basic arithmetic and broadcast support | Full promotion rules, complex semantics, integer overflow policy, named dimensions and exhaustive broadcasting tests |
| Autograd | Reverse-mode graph and backward operations | Higher-order gradients, forward-mode AD, vmap, functional transforms, checkpointing, anomaly detection and gradient correctness matrix |
| Neural-network modules | Linear, normalization, attention, convolution and transformer components | Full module lifecycle, parameter/buffer semantics, initialization, hooks, lazy modules and parity-tested layer coverage |
| Functional operators | Large PyTorch-like functional namespace | Verify every operator's shape, padding, dtype, numerical and backward semantics; remove placeholders |
| Losses and metrics | Common classification/regression losses | Complete reduction modes, weighting, ignore-index, label smoothing, numerical stability and metric state |
| Optimizers | SGD/Adam/AdamW and additional optimizer implementations | Complete optimizer state, parameter groups, sparse gradients, closures, fused variants and checkpoint parity |
| Learning-rate schedules | Several scheduler implementations | Full scheduler families, warmup/composition, per-step semantics and serialization |
| Data pipeline | Array datasets and minibatch loaders | Composable transforms, streaming/iterable datasets, workers, prefetch, pinned memory, caching, sharding and deterministic resume |
| Sparse and ragged tensors | COO/CSR/CSC/BSR-related support | Sparse autograd, sparse-dense kernels, coalescing, block layouts, ragged/nested tensors and operator coverage |
| Convolution and signal processing | 1D/2D operations and signal utilities | 3D/transposed/depthwise/grouped/dilated coverage, gradients, FFT kernels and backend parity |
| Distributed training | Ring all-reduce and parameter synchronization foundations | Collective API, process groups, DDP-equivalent synchronization, sharding, elastic recovery and multi-node integration tests |
| Accelerators | CUDA code and optional Metal/Vulkan/TPU/NPU/AMD/Intel build options | Real device allocation, streams/events, kernel coverage, synchronization, memory accounting and hardware CI |
| Compilation and graph execution | JIT/graph optimization components | Stable graph capture, tracing, shape guards, operator lowering, fusion, recompilation policy and compiler diagnostics |
| Serialization | Tensor binary I/O, checkpoint and SafeTensors utilities | Versioned model/state formats, optimizer/RNG state, atomic writes, schema migration and cross-version tests |
| Model ecosystem | ONNX and Hugging Face integrations | Import/export operator coverage, tokenizer/model compatibility, quantized models and reproducible loading |
| Quantization | Research/low-level primitives | Eager and graph quantization, calibration, PTQ/QAT, observers, per-channel schemes and hardware validation |
| Mixed precision | FLOAT16/BFLOAT16/FLOAT8 dtype identifiers | Real storage/arithmetic support, autocast, gradient scaling and overflow tests |
| Profiling and debugging | Benchmarking and observability components | Operator profiler, memory profiler, trace export, tensor anomaly reports and reproducible performance baselines |
| Deployment | HTTP inference and packaging components | Stable serving API, batching, streaming, model versioning, runtime isolation and production health checks |
| Interoperability | NumPy, ONNX, Hugging Face and Keras-like surfaces | DLPack, array protocols, Python buffer protocol, TensorFlow/PyTorch import/export and documented compatibility boundaries |

## Implementation phases

### P0 — Correctness and contracts

- Align Python fallback enums with the native C ABI.
- Define dtype promotion, device, layout, scalar, view, copy and aliasing contracts.
- Audit tensor allocation, bounds, overflow, strides and storage lifetime.
- Add property-based tests for shapes, broadcasting, reductions and indexing.
- Ensure tests remain active in Release builds and fail on numerical mismatches.

### P1 — Tensor API completeness

- Finish creation, indexing, views, in-place operations and reductions.
- Add complete dtype conversions and explicit unsupported-dtype errors.
- Implement gather/scatter, advanced indexing and sparse layout semantics.
- Provide consistent NumPy and DLPack interoperability.

### P2 — Autograd parity

- Establish finite-difference gradient tests for every differentiable operator.
- Complete higher-order reverse-mode gradients and forward-mode AD.
- Add vectorized transforms, checkpointing, hooks and anomaly detection.
- Document non-differentiable operations and gradient behavior.

### P3 — Neural-network and optimization stack

- Complete module, parameter, buffer, initialization and state-dict semantics.
- Complete layers, losses, optimizer state and scheduler families.
- Add training utilities: gradient clipping, accumulation, AMP, EMA and callbacks.
- Validate end-to-end training on small deterministic reference models.

### P4 — Data and model ecosystem

- Build composable map/iterable datasets, transforms and worker/prefetch pipelines.
- Complete checkpointing, SafeTensors and ONNX interoperability.
- Add tokenizer, model-loading and reproducible dataset utilities.

### P5 — Performance and accelerators

- Establish CPU reference kernels and numerical tolerances.
- Expand SIMD and CUDA kernel coverage; implement device memory and stream semantics.
- Add real backend conformance tests before advertising a backend as supported.
- Add graph compilation, fusion, quantization and mixed-precision execution.

### P6 — Distributed and production runtime

- Complete collective communication and distributed data parallel training.
- Add sharded training, checkpoint recovery and elastic membership.
- Complete serving, batching, observability and deployment workflows.

## Definition of done

Every feature must have:

1. A documented API and explicit supported dtype/device/layout matrix.
2. A correct reference implementation and validation of edge cases.
3. Forward tests and gradient tests where differentiable.
4. Native/Python parity tests where both interfaces exist.
5. Error-path tests for invalid shapes, unsupported devices and allocation failures.
6. CI coverage on supported platforms; hardware-specific claims require matching hardware CI.
7. Documentation that distinguishes implemented, experimental, unsupported and planned behavior.

## Compatibility labels

- **Supported:** semantics and tests are stable on the documented backend.
- **Experimental:** usable implementation exists, but API or backend behavior may change.
- **Partial:** only a subset of expected semantics is implemented.
- **Planned:** no complete implementation is available yet.
- **Unsupported:** deliberately unavailable; never silently emulate a capability with incorrect semantics.

The project must not advertise full TensorFlow or PyTorch compatibility until the
relevant conformance suites demonstrate it. The roadmap is intentionally broad;
work should proceed in small, reviewable, test-backed changes.
