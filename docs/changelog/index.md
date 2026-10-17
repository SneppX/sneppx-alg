# Changelog

Auto-generated from git tags + commit history. Regenerate with `python docs/changelog/generate.py`. No CI writes this file.

<!-- prettier-ignore -->

[TOC]

## Unreleased
_unreleased_

### Features

- feat(onnx): add standalone numpy-only ONNX import/export toolkit
- feat(integrations): add bi-directional HuggingFace integration module
- feat(edge): add lightweight C edge runtime skeleton, build system, and model converter
- feat(compiler): add static graph optimizer with constant folding and IR serialization
- feat(api): add PyTorch-like high-level Python API layer and 10 examples
- feat(hub): add model hub CLI, server, registry, storage, and C client

### Bug Fixes

- fix(security): Dilithium S0 sign rejection from NTT domain + zeta typos
- fix(arc,security): argon2 timing-safe password clobber + ARC input_guard output_dim projection
- fix(security): S7 version enforcement + S8 symex paths + S9 redteam return
- fix(security): VM dispatch scrambling + HALT register wipe
- fix(autodiff): no_grad scope zeroes requires_grad on outputs
- fix(tensor): matmul of two 1-D vectors returns NULL
- fix(attention): correct in-place RoPE contamination + NULL sin bail
- fix(kernel): SNEPPX_arch_config_default input/output_dim 16 -> 512
- fix(compress): add SNEPPX_COMPRESS_NONE pass-through codec
- fix(ser): correct malloc/_aligned_free mismatch in route+forward
- fix(hub): correct download card path, total_size, upload overrides, and CLI errors
- fix(vizmon): fix access violation, DLL export macros, and JSON format in histogram snapshot
- fix(crypto): correct Dilithium NTT twiddle table and Kyber/DRBG crypto primitives
- fix: tensor.c header insertion broke BINARY_OP_F32 macro
- fix: argon2.c header insertion broke macro definition
- fix: replace missing tensor.h include and wrong API names in attention_module.c

### Build / Tooling

- build(ninja): auto-resolve x64 MASM to ml64 so 'cmake -B build -G Ninja' works
- build: add Ninja generator to profiles README
- build: set Ninja generator in devcontainer config
- build: add Ninja generator to remaining cmake .. commands
- build: switch docker-compose to Ninja generator
- build: switch install.sh and Dockerfile to Ninja generator
- build: switch install.sh to Ninja generator
- build: add Ninja generator to shell scripts
- build: add Ninja generator to Makefile
- build: add Ninja generator to run_sanitizers.ps1
- build: add Ninja generator to install.ps1
- build: switch build.ps1 to Ninja generator
- build: wire sneppx-format --docs into pre-commit and dev-tools.ps1

### Other

- test: fix RoPE self-attention + causal masking failures
- test: use INT32 index tensor in embedding grad check
- test: migrate C/C++ test suite to GoogleTest (135 .c + 6 .cpp)
- docs(onnx): document ONNX import/export toolkit
- docs: update READMEs with edge runtime information
- docs(readme): update README with Model Hub, serving guides, and v1.1.1 highlights
- docs: update main documentation index for v1.1.1
- docs: update roadmap with v1.1 completion and model hub status
- docs(guide): add model hub and serving system guides
- docs(guide): add model hub and serving system guides
- Serving system: C control plane + Python engine/client + FastAPI endpoints + example config
- Fix save/load_hf_model for modules without named_parameters
- Add paged attention, MX formats, JIT tracing, MLflow-like observability, hw backends
- Fix stale summary-matrix rows for layer API and experiment tracking
- Document hyperparameter-search closure and complete gap analysis
- Add hyperparameter-search orchestrator with random/grid/halving/bayesian samplers
- Document edge runtime closure in gap analysis
- Add mobile/edge runtime with device detection and INT8 inference
- Document graph compiler closure in gap analysis
- Add graph compiler with element-wise fusion, tiling and C codegen
- Document Keras-style layer API and autograd fix in gap analysis
- Add Keras-style layer API with compile/fit/evaluate/predict/summary
- Fix broadcast-aware backward for Add/Sub/Mul/Div autograd ops
- Document experiment/run tracking closure in gap analysis
- Add experiment/run tracking with metadata and metrics persistence
- Document ONNX shape inference and QAT closure in gap analysis
- Add ONNX shape inference and op-schema checks (onnx_check)
- Document Python canonical binary ONNX bridge in gap analysis
- Add canonical binary ONNX export/import to Python bindings
- Mark QAT fake-quant as implemented in gap analysis
- Fix allocator mismatch in reduce_grad_to_shape error path
- Add QAT fake_quant op with straight-through estimator backward
- Fix double-free in autodiff tape backward tests
- Add ONNX structural validator (SneppX_onnx_validate) with rejection tests
- Reflect general-graph ONNX export implementation in gap analysis
- Add general-graph ONNX exporter with multi-node graph test
- Add ML framework feature gap analysis and roadmap
- Add real binary ONNX exporter for linear models with round-trip test
- docs: fix stale Ninja build output paths (build_test/Release subdirs)
- docs(changelog): record Ninja build switch + Dilithium/Kyber/DRBG fixes
- docs(quickstart): use release preset + correct ninja build output paths
- docs(testing): align test dirs/commands to Ninja preset layout
- docs(agents): note stable Ninja + auto-ml64 for Windows builds
- docs(build): align build/test instructions to Ninja presets + VS+MASM notes
- docs: add Ninja generator to remaining README build commands
- docs: add Ninja generator to remaining doc build commands
- docs: add Ninja generator to Docs.md build commands
- docs: add Ninja generator to README build commands
- docs: add Ninja generator to build.md
- docs: add Ninja generator to build commands
- docs: index the five pipeline code walkthroughs
- docs: add FM code walkthrough
- docs: add NPE code walkthrough
- docs: add ARC code walkthrough
- docs: add SER code walkthrough
- docs: add HSS code walkthrough
- docs: add pipeline code-walkthrough overview
- chore: remove obsolete docs/process_all.py batch annotation script
- docs: add Layer-1 headers and Doxygen blocks to lib, bindings and standalone tests
- docs: add Layer-1 headers across examples, benchmarks, samples and tests
- docs: add Doxygen blocks and Layer-1 headers across drivers and tools
- docs: add Doxygen blocks and Layer-1 headers across memory and filesystem modules
- docs: add Doxygen blocks and Layer-1 headers across network modules
- docs: add Doxygen blocks and Layer-1 headers across all kernel modules
- docs: add Doxygen blocks and Layer-1 headers across all algorithm modules
- docs: add Doxygen blocks and Layer-1 headers across all security modules
- docs: add Doxygen blocks and Layer-1 headers to crypto C sources
- docs: add Layer-1 header to coverage.h
- docs: add Doxygen blocks and Layer-1 headers to drivers, model_zoo, network headers
- docs: add Doxygen blocks and Layer-1 headers to all kernel headers
- docs: add Doxygen blocks to all architecture headers
- docs: add Doxygen blocks and Layer-1 headers to all security headers
- docs: add Doxygen blocks to kyber and aes_gcm headers
- docs: add commenting standard reference to security architecture
- docs: add commenting standard reference to C API reference
- docs: add commenting standard to docs index
- docs: add commenting standard to contributing guide
- docs: add commenting standard reference to README
- docs: add commenting standard reference to doxygen guide
- docs: add commenting standards reference to architecture guide
- docs: add commenting standard to development guide
- docs: add commenting standards to code review checklist
- docs: add commenting standard reference to style guide
- docs: add commenting standards reference to index
- docs: add Layer-1 file headers, API docs, and commenting conventions
- Bump version to 1.1.1 and add v1.1.1 changelog
- Add C HTTP REST API, packaging improvements, quickstart, and dev-tools chain
- Fix unsafe C functions: replace gets/scanf/strcpy/sprintf/alloca with safe alternatives
- Fix spec: correct tensor header and source file paths
- Fix fuzz harness: register targets via static init, add NULL target check
- Phase 4-5: C HTTP server + middleware + API reference docs + CMake integration
- chore: reorganize documentation system
- chore: remove all CI/CD â€” manual-only workflow
- fix CI: install numpy for CMake, add build step for CodeQL analysis, fix pre-commit deps
- fix CI workflows: correct action versions, add CodeQL init, fix cross-platform shell compatibility
- update PR template to tier-based system
- chore: cleanup project structure

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v1.2.0...Unreleased){ .md-button }

## v1.2.0
**2026-07-30**

_No tagged changes; showing commits up to this tag._

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v1.1.1...v1.2.0){ .md-button }

## v1.1.1
**2026-08-01**

### Other

- Bump version to 1.1.1 and add v1.1.1 changelog
- Add C HTTP REST API, packaging improvements, quickstart, and dev-tools chain
- Fix unsafe C functions: replace gets/scanf/strcpy/sprintf/alloca with safe alternatives
- Fix spec: correct tensor header and source file paths
- Fix fuzz harness: register targets via static init, add NULL target check
- Phase 4-5: C HTTP server + middleware + API reference docs + CMake integration
- chore: reorganize documentation system

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v1.1.0...v1.1.1){ .md-button }

## v1.1.0
**2026-07-30**

### Features

- feat: add serving benchmark; update pyproject.toml with new CLIs; fix continuous_batching decode
- feat: integrate continuous batching + quantized serving into inference server; fix rlhf_cli
- feat: add sneppx-eval, sneppx-quantize, sneppx-rlhf CLI entry points
- feat: add synthetic data loaders for MMLU/GSM8K/HumanEval eval tasks
- feat: add FP8/INT4 quantized model serving module
- feat: add continuous batching scheduler for inference serving
- feat: complete v1.1.0 â€” LoRA/QLoRA, Eval Harness, FSDP
- feat: add serving profile to Docker Compose

### Bug Fixes

- fix: replace missing tensor.h include and wrong API names in attention_module.c
- fix: use .item() for numpy scalar conversion in ultra_trainer test
- fix: add model_implementations test file
- fix: correct API calls in LLM/vision test files (model names, create_model args)
- fix: align quantized_serve with QuantizedLinear API, fix quant_mode refs
- fix: implement DPO/GRPO trainer logprob methods (replaced stubs)

### Other

- chore: remove all CI/CD â€” manual-only workflow
- fix CI: install numpy for CMake, add build step for CodeQL analysis, fix pre-commit deps
- fix CI workflows: correct action versions, add CodeQL init, fix cross-platform shell compatibility
- update PR template to tier-based system
- chore: cleanup project structure
- docs: add contribution framework with tiers, branching strategy, CI workflows
- Fix test bugs found during regression suite: - test_crypto_asm.py: add missing import hashlib - test_eval_harness.py: update assertion for synthetic data fallback - test_key_vault.py: adapt to audit_logger.py API (FileAuditBackend, AuditAction enums) - test_s4_network.py: adapt to ddos_mitigation.py API (DDoSConfig, check_request) - test_trainer_v2.py: add missing import numpy as np - audit_logger.py: string-to-enum conversion in log(), hash computation fix in FileAuditBackend.write(), verify_chain sort order fix, add clear() - benchmarking.py: fix RNN/transformer weight shape mismatches - test_algo_wrappers.py: add __test__ = False to prevent pytest collection
- docs: update AGENTS.md with v1.2-1.3 features and new safe test list
- test: add tests for continuous batching, quantized serve, and CLI entry points
- docs: add .nojekyll for GitHub Pages serving
- test: add batch 3 test files (security, CLI, benchmark, trainer_v3, audit, firewall, DDoS)
- test: add batch 2 test files (data_loader, advanced_ops, augmentation, pruning, distillation)
- test: add 5 missing test files (AMP, grad_checkpoint, tokenizer, autograd_ops, schedulers) â€” 60 tests
- docs: add documentation system with Doxygen config, feature guides, and API references
- chore: update Arix-Site submodule to b62b5d2 (add Models, Benchmarks, Playground, API Reference)
- chore: bump version to 1.1.0 and prepare pre-release

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v1.0.0...v1.1.0){ .md-button }

## v1.0.0
**2026-07-26**

### Features

- feat: add 128K context extension and MHA forward pass

### Bug Fixes

- fix: pyproject.toml dependencies format for PEP 621 compatibility

### Other

- chore: update Arix-Site submodule to latest (v1.0.0 site rebuild)
- chore: update Arix-Site submodule to v1.0.0 site content
- repo SEO & infrastructure: badges, CITATION.cff, keywords, classifiers, docs refresh
- docs: comprehensive v1.0.0 documentation refresh
- v1.0.0: Stable Release
- 1.8: C + Python integration tests; weights.c buffer overrun fix
- 1.7: C++ Model Factory with RAII wrappers; model_card_to_json hang fix; 1.6 Python ModelHub
- 1.5: Python ModelConfig dataclass with JSON serialization, validation, C config conversion, registry integration; added to LlamaConfig, MistralConfig, Qwen2Config, DeepSeekV2Config
- 1.4: Model Cards - Metadata with JSON serialization, validation, file I/O; unit tests
- 1.3: Pretrained Weights - WeightCollection API, dtype conversion (f32/f16), INT8 quantization, safetensors/gguf/npz loaders (stubs); unit tests
- 1.2: Model Registry - Registration, discovery, search, save/load, deprecation; C API with tests
- 1.1: Model Config Schema - C config API with JSON serialize/parse, presets for LLaMA2/3, Mistral, Qwen2, BERT, ViT, SDXL; unit tests
- docs: Complete documentation overhaul for v0.5.0
- Update ROADMAP.md to reflect v0.5.0 completed features
- Add Python API wrappers for ARC/NPE/FM/Trainer features
- Add CUDA-accelerated training loop with optimizer bridge
- Add NPE JIT pipeline with fusion passes, auto-JIT in VM
- Add FM NCCL distributed sync bridge with callback pattern, tests
- Wire ARC adversarial training: FGSM/PGD/CW attack injection in training graph, config epsilon, test
- Add SER learned MLP gater: config flag, 2-layer MLP forward, autodiff subgraph, bindings, tests
- Enable HSS Blelloch parallel scan by default; update ARCHITECTURE.md
- Remove remaining CI/CD mentions (cliff.toml, ROADMAP)
- Remove CI/CD: drop GitHub workflows/dependabot/CODEOWNERS, strip CI mentions from docs and configs
- Docs: bump v0.9.7.890e, refresh stale content, add release/migration/HSS-training docs

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.7.890e...v1.0.0){ .md-button }

## v0.9.7.890e
**2026-07-19**

### Bug Fixes

- Fix: HSS backward corruption â€” layer_norm gamma/beta gradient pointer deref + missing math.h (sqrt)
- Fix: build lib/internal/*.c + vulkan typo + CI matrix + tape guard

### Other

- Phase E: E2E integration tests + gradient flow tests + CI l0-validation job
- Phase D: Layer 2 Memory cognitive architecture
- Phase C: realize 6 stub drivers with reference-compute + opt-in flags + tests

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.7.890b...v0.9.7.890e){ .md-button }

## v0.9.7.890b
**2026-07-18**

### Other

- Phase B: populate 4 kernel directories (activations, position, algorithms, drivers)

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.7.890a...v0.9.7.890b){ .md-button }

## v0.9.7.890a
**2026-07-18**

### Other

- Phase A: eliminate last C stubs (vmem evict + FM params/train_graph)
- Update VERSION with correct LOC/file/test counts
- Docs: document v0.9.7.890 â€” real implementations and opt-in Vulkan/TPU/HTTP/ZK backends

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.7.890...v0.9.7.890a){ .md-button }

## v0.9.7.890
**2026-07-18**

### Features

- feat: add S8/S9 security layer stubs (verification, hardware, c/asm)

### Bug Fixes

- fix: implement real C networking layer (replace skeletons with working code)
- fix: critical bugs, security hardening, and code quality improvements
- fix: replace unsafe strcpy/sprintf/sscanf with bounds-checked snprintf across 5 security modules
- fix: remove duplicate version param in FastAPI init

### Other

- Build v0.9.7.890: real implementations across formats, kernel, architectures, security, and backends
- Update files and add new kernel algorithms
- Rewrite README with accurate stats, correct links, updated architecture tree
- Create comprehensive skeleton framework for future planned features
- Update tests for real implementations + fix vmem.c field name
- Fill in remaining 4 stub files with real implementations
- Fill in remaining stubs with real implementations (+1,227 lines)
- Fix autodiff backward regressions: matmul, pow, sum, and add reduce_grad_to_shape
- Wire CUDA optimizer dispatch for all optimizer types, add LARS enum
- v0.9.5.937: Fix CUDA autodiff bugs, implement NCCL reduce/reduce_scatter, add RAdam and Schedule-Free AdamW C optimizers
- Implement all 26 missing autodiff forward+backward ops, add header declarations
- Fix package name in pyproject.toml, add Python audit_logger/container_security/ddos_mitigation modules, fix SyslogBackend default arg and Tuple import
- Fix autodiff ops.c deduplication, add missing helpers, implement all 25 forward+backward ops
- Re-enable 13 previously excluded C tests by implementing missing APIs and fixing test mismatches
- Fix pre-existing C compilation errors (CUDA guards, missing includes, duplicate symbols)
- Add scripts/README.md documenting sanitizer build/CI scripts
- Add include/README.md documenting C/C++ header structure and include conventions
- Add examples/README.md documenting demo programs
- Add drivers/README.md for CUDA, ROCm, and TPU hardware abstraction layers
- Add net/README.md documenting topology, socket, RDMA, gRPC, and NCCL layers
- Add docs/security_layers.txt with deep-dive into all 10 security layers (S0-S9), post-quantum crypto, and compliance mappings
- Add docs/build.txt (build instructions, options, troubleshooting) and docs/api_quickref.txt (C + Python API reference)
- Add docs/architecture.txt with system layers, data flow, memory model, distributed arch, and security stack
- Add tests/README.md documenting 31+ test suites, Python/C/CUDA tests, and excluded tests
- Add bindings/README.md documenting Python bindings architecture, 8 phases, build, and pure-Python fallback
- Add security/README.md documenting S0-S9 layers, post-quantum crypto, 4-ring firewall, and MASM routines
- Add algorithms/README.md covering HSS, SER, ARC, NPE, FM with configs and CUDA kernel details
- Add kernel/README.md documenting the entire computational substrate (tensor, autodiff, optimizer, CUDA, distributed, quantization)
- Expand root README.md with full project structure, C networking layer, post-quantum crypto, and build instructions
- Split bindings.cpp into modular include files
- docs: comprehensive README for v0.9.5.748 release

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.5.748...v0.9.7.890){ .md-button }

## v0.9.5.748
**2026-07-15**

### Features

- feat: add test_watermark (tests/python)
- feat: add test_security_middleware (tests/python)
- feat: add test_s5_safety (tests/python)
- feat: add test_onnx (tests/python)
- feat: add test_nccl (tests/python)
- feat: add test_inference_server (tests/python)
- feat: add test_hparams (tests/python)
- feat: add test_graph_compiler (tests/python)
- feat: add test_generation (tests/python)
- feat: add test_experiment_tracker (tests/python)
- feat: add test_distributed_wrapper (tests/python)
- feat: add test_differential_privacy (tests/python)
- feat: add test_cuda_kernels (tests/python)
- feat: add test_cuda (tests/python)
- feat: add test_checkpoint_manager (tests/python)
- feat: add test_autograd (tests/python)
- feat: add test_adversarial (tests/python)
- feat: add example (config/training)
- feat: add watermark (bindings/python/SneppX_ALG/interface_bindings)
- feat: add trainer_v3 (bindings/python/SneppX_ALG/interface_bindings)
- feat: add train_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add tokenizer (bindings/python/SneppX_ALG/interface_bindings)
- feat: add serve_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add security_middleware (bindings/python/SneppX_ALG/interface_bindings)
- feat: add s5_safety (bindings/python/SneppX_ALG/interface_bindings)
- feat: add nccl (bindings/python/SneppX_ALG/interface_bindings)
- feat: add inference_server (bindings/python/SneppX_ALG/interface_bindings)
- feat: add hparams (bindings/python/SneppX_ALG/interface_bindings)
- feat: add graph_compiler (bindings/python/SneppX_ALG/interface_bindings)
- feat: add generation (bindings/python/SneppX_ALG/interface_bindings)
- feat: add experiment_tracker (bindings/python/SneppX_ALG/interface_bindings)
- feat: add experiment_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add distributed_wrapper (bindings/python/SneppX_ALG/interface_bindings)
- feat: add differential_privacy (bindings/python/SneppX_ALG/interface_bindings)
- feat: add cuda_kernels (bindings/python/SneppX_ALG/interface_bindings)
- feat: add cuda_device (bindings/python/SneppX_ALG/interface_bindings)
- feat: add checkpoint_manager (bindings/python/SneppX_ALG/interface_bindings)
- feat: add autograd_ops (bindings/python/SneppX_ALG/interface_bindings)
- feat: add autograd (bindings/python/SneppX_ALG/interface_bindings)
- feat: add adversarial (bindings/python/SneppX_ALG/interface_bindings)

### Bug Fixes

- fix(N4): fix autocast duplicate import syntax error
- fix(N3): Optimizer saves lr, step() works without C backend, autocast is proper context manager
- fix(N2): add Trainer.train() method and fix LSTM benchmark shape
- fix(N1): MSELoss returns scalar tensor
- fix: resolve test import errors in test_trainer_v2 and test_ultra_trainer
- fix(C3): add fallback stub when C extension not available
- fix(C2): add missing Tensor import and fix matmul in benchmarking.py
- fix(C1): prevent SimpleTokenizer shadowing from .tokenizer over .data version
- fix(build): compile CUDA extension sources into existing exported targets via target_sources
- fix(build): link CUDA extension libs privately to avoid export-set errors
- fix(build): add CMakeLists.txt for algorithms/*/cuda CUDA extension libraries
- fix(ci): pin CUDA builds to ubuntu-22.04 for CUDA 12.4 repo compatibility
- fix(ci): install CUDA toolkit on runner (avoid container checkout break), mark macOS/Windows non-blocking
- fix(ci): build Linux via CUDA container, add net/distributed CMakeLists, mark macOS/Windows non-blocking
- fix: guard CUDA-only source files for CPU builds, fix CI workflow
- fix(ci): install numpy for CMake, fix markdownlint, cuda keyring
- fix(ci): remove remaining empty with block in security-scan
- fix(ci): remove submodules recursive (breaks on Arix-Site dir)
- fix(ci): remove duplicate runs-on, embedded dependabot, and other invalid YAML

### Refactor / Internal

- refactor: update advanced_ops, quantization, C tests
- refactor: update C source files (checkpoint_reader, dilithium)
- refactor: update test files and CMakeLists
- refactor: update __init__.py exports
- refactor: update benchmarks/export/distillation
- refactor: update training/infra modules
- refactor: update model/hf/vision modules
- refactor: update core tensor/nn/optim/data modules
- refactor(ci): restructure jobs, fix deps, make CUDA/security/docs non-blocking

### Other

- Release v0.9.5.748 â€” Unified Python Bindings (Phases 1-7), CLI commands, version bump
- Fix CLI entry points (console_scripts), experiment_cli.py import time bug, add missing deps, sync pyproject.toml
- bump: v0.9.4.467
- firewall: test files (24 tests), fix path traversal detection, lazy SSL context build
- firewall: CLI flags for serve_cli.py, kwargs safety in firewall constructors
- firewall: wire into inference_server.py middleware, set_security accepts firewall config
- firewall: integrate into SecurityMiddleware with check_firewall, release_concurrent
- firewall: orchestrator with YAML config, env/CLI overrides, 3-ring dispatch
- firewall: transport ring with TLS/mTLS, cert pinning, ALPN
- firewall: application ring with injection filter, path normalization, concurrent limiter
- firewall: network ring with IP CIDR filtering, rate limiting, port knock
- firewall: assembly routines for IP match, rate counter, conn track, port knock
- chore: bump version to 0.9.2.094 across all files
- ci: mark C/CUDA build jobs non-blocking (pre-existing source compile errors); keep Lint/Security/Docs as real gates
- chore(ci): add workflow_dispatch for manual runs

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.0-fix4...v0.9.5.748){ .md-button }

## v0.9.0-fix4
**2026-07-11**

### Bug Fixes

- fix: remove CUDA job (needs Docker Hub secrets not set)

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.0-fix3...v0.9.0-fix4){ .md-button }

## v0.9.0-fix3
**2026-07-11**

### Bug Fixes

- fix: wrap secrets check in template expr for job if

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.0-fix2...v0.9.0-fix3){ .md-button }

## v0.9.0-fix2
**2026-07-11**

### Bug Fixes

- fix: skip CUDA Docker if Docker Hub secrets not set

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.0-fix1...v0.9.0-fix2){ .md-button }

## v0.9.0-fix1
**2026-07-11**

_No tagged changes; showing commits up to this tag._

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.9.0...v0.9.0-fix1){ .md-button }

## v0.9.0
**2026-07-11**

### Features

- feat(packaging): add comprehensive packaging for PyPI, Docker, Conda, CI/CD
- feat(models): add complete LLaMA, Mistral, Qwen2, DeepSeek V2 architectures
- feat(vision): add Vision Transformer (ViT), DeiT, Swin, MAE architectures
- feat: add benchmarking suite + all 11 tests pass
- feat: massive expansion - distillation, pruning, advanced ops, augmentation, ONNX, model zoo
- feat: distillation, pruning, advanced ops, augmentation, ONNX export
- feat(augmentation): comprehensive data augmentation pipeline
- feat(train): add AMP, gradient checkpointing, advanced DataLoader, UltraTrainer
- feat(optim): add 7 extra optimizers + 10 schedulers + 6 advanced optimizers
- feat(tensor): add advanced ops - conv, pooling, RNN, attention, norms, tensor manip
- feat(core): add SIMD GEMM + tensor expression IR with operator fusion
- feat(crypto): Dilithium DRBG-based sampling, 10-bit packing, SHA512 challenge
- feat(zoo): model configs, weight converters, from_pretrained for LLaMA2/3, Mistral, Qwen2, DeepSeek V2 + 49 tests
- feat(profile): profiler, logger, NVTX markers, compute-sanitizer CI scripts + 13 tests
- feat(ft): async checkpointing, heartbeat, elastic training, fault tolerance (C + Python) + 23 tests
- feat(quant): INT8/FP8/AWQ/GPTQ quantization kernels (C, CUDA, Python) + 17 tests
- feat(py): self-contained Tensor with optional C backend, update exports and pip package config
- feat(py): implement neural network, optimizers, data, distributed, hf, model, train modules
- feat(arch): Mamba-2 selective SSM with HiPPO initialization, conv1d, discretized scan
- feat(arch): FlexAttention block-sparse with mask modulation, multi-modal cross-attention, MoD, gated activations, YaRN NTK-RoPE, ALiBi
- feat(arch): Differential Attention and Multi-head Latent Attention (DeepSeek MLA style)
- feat(arch): advanced architectures master header (DifferentialAttn, MLA, FlexAttn, Mamba2, MoD, YaRN, ALiBi)
- feat(dist): distributed sampler with epoch-based shuffling and gradient accumulation manager
- feat(dist): distributed checkpoint coordinator with async save and fault tolerance
- feat(dist): hierarchical all-reduce (NVLink+RDMA) and Top-K gradient compression with error feedback
- feat(dist): DDP with bucket-based gradient all-reduce and compute overlap
- feat(dist): expert parallelism all-to-all dispatch and FM distributed communication
- feat(dist): tensor parallelism row/column split linear with all-reduce
- feat(dist): pipeline parallelism 1F1B schedule with microbatches
- feat(dist): ZeRO-1/2/3 optimizer state partitioning and fused AdamW step
- feat(dist): distributed training config header (ZeRO, pipeline, tensor, expert parallel)
- feat(dist): NCCL dynamic loading, all-reduce, process group part 2/2
- feat(dist): NCCL dynamic loading, all-reduce, process group part 1/2
- feat(dist): NCCL communication primitives header
- feat(hss): extended SSM step, conv, selective scan kernels
- feat(arc): PGD/FGSM attacks, gradient obfuscation, smoothing
- feat(arc): adversarial robustness CUDA header
- feat(npe): differentiable program execution GPU kernel part 2/2
- feat(npe): differentiable program execution GPU kernel part 1/2
- feat(npe): neural VM instruction dispatch CUDA header
- feat(fm): ring/butterfly all-reduce, gradient quantization part 2/2
- feat(fm): ring/butterfly all-reduce, gradient quantization part 1/2
- feat(fm): all-reduce, quantization, federated avg CUDA header
- feat(ser): fused MoE forward, load balancing loss
- feat(ser): top-k gating, dispatch, combine kernels
- feat(ser): top-k gating and fused MoE CUDA header
- feat(hss): Mamba/S6 selective scan, SSM conv, HiPPO matrix CUDA kernels
- feat(hss): selective scan, S4, HiPPO CUDA header
- feat(cuda): RNG API header (Philox, distributions, init schemes)
- feat(cuda): random number generation kernels part 3/3
- feat(cuda): random number generation kernels part 2/3
- feat(cuda): random number generation kernels part 1/3
- feat(cuda): memory pool/stream/event API header
- feat(cuda): memory management (pool, streams, events) part 3/3
- feat(cuda): memory management (pool, streams, events) part 2/3
- feat(cuda): memory management (pool, streams, events) part 1/3
- feat(cuda): optimizer header with all step types
- feat(cuda): fused optimizer kernels part 3/3
- feat(cuda): fused optimizer kernels part 2/3
- feat(cuda): fused optimizer kernels part 1/3
- feat(cuda): autodiff backward header declarations
- feat(cuda): autodiff backward kernels part 4/4
- feat(cuda): autodiff backward kernels part 3/4
- feat(cuda): autodiff backward kernels part 2/4
- feat(cuda): autodiff backward kernels part 1/4
- feat(cuda): attention kernels part 5/5
- feat(cuda): attention kernels part 4/5
- feat(cuda): attention kernels part 3/5
- feat(cuda): attention kernels part 2/5
- feat(cuda): attention kernels part 1/5
- feat: expand assembly code to 3258 lines with constant-time, speculation-safe, cache-resistant security improvements across 15 files
- feat: security infrastructure expansion - compliance, PQ crypto, threat intel, IR, automation, ZT, chaos, AI guardrails, supply chain, fuzzing + docs
- feat(tokenizer): BPE tokenizer with train/encode/decode/save/load
- feat(autodiff): gradient checkpointing + view-aware storage
- feat(autodiff): min/max with selective gradient routing
- feat(autodiff): ref-counted backward lifecycle with ctx cleanup

### Bug Fixes

- fix(docker): make CPU Dockerfile self-contained, fix packages workflow
- fix(version): bump setup.py to 0.9.0 to match pyproject.toml and tag
- fix(crypto): Kyber uses DRBG for deterministic noise sampling instead of rand()
- fix: full build cleanup â€” fix 8 test files for API changes, stub sha512, fix audit_logger stdarg, fix secure_wipe ABI, exclude unimplemented tests
- fix(asm): rewrite poly1305_sse.asm with correct 5-limb scalar algorithm, fix 8 ABI signature mismatches in asm_exports.h
- fix(asm): fix 6 ABI signature mismatches in asm_exports.h header
- fix(asm): fix 5 assembly correctness bugs
- fix(crypto): fix 7 critical bugs - auth bypasses, memory overflows, X25519, Kyber
- fix: update metadata to align with sneppx-alg identity
- fix: update build.ps1 CMake flags to SNEPPX_ prefix
- fix: rename arix_algo package to SneppX_ALG, fix Python relative imports
- fix: sha3 finalize->finish, fm_forward 4 args, m.lib MSVC guard
- fix: add csignal, time.h, link m for math in kernel lib
- fix: move sys/prctl.h to __linux__ guard, ifdef s9_extensions _finddata for WIN32
- fix: s9_extensions io.h guard, blake3 finalize->finish across 4 files
- fix: use asm/unistd.h for __NR_* in seccomp filter
- fix: memory_hardening sys/syscall.h, hmac.c sha512 func name
- fix: memory_hardening linux headers, drbg sha512 func name, asm only on MSVC
- fix: ed25519 _umul128 compat + point_is_on_curve ordering; fix cache.c aarch64 prfm syntax
- fix: sign scheme S=r+h*a mod L with proper sc_reduce64
- fix: fe_to_bytes final p-subtraction via limb comparison
- fix: fe_mul rewrite with _umul128 128-bit arithmetic
- fix: point_add formula (X3=EÂ·F, Y3=GÂ·H, T3=EÂ·H, Z3=FÂ·G)
- fix: point_double formula (Y3=EÂ·G, T3=FÂ·G, Z3=DÂ·H) + identity guard
- fix: fe_sub per-limb bias constants
- fix: point_scalar_mult cswap mask uint8_t->uint64_t
- fix: add two-round carry chain to fe_to_bytes
- fix: add shebangs to build.sh, test.sh, clean.sh

### Other

- ci(packages): add GitHub Packages workflow for Docker and Python publishing
- chore: update Arix-Site submodule reference after cleanup
- test(crypto): add Kyber debug output for pk/sk/shared-secret bytes
- docs: update AGENTS.md with Phases 5-8, deprecate old lib/python bindings
- test(py): add 7 Python API test suites (82 tests)
- test(cuda): GPU kernel test suite part 3/3
- test(cuda): GPU kernel test suite part 2/3
- test(cuda): GPU kernel test suite part 1/3
- docs: update AGENTS.md with Phase 1 CUDA backend overview
- chore: bump version to 0.8.6 for 12 critical bug fixes across C and asm
- Fix ASM_MASM compiler (ml64.exe) and add asm_exports.h header
- Update README for algo0.8.2: assembly stats, version bump, S0 detail
- Fix MASM syntax errors: all 15 asm files assemble cleanly
- Fix assembly bugs across 8 files
- Remove CI/CD workflows, rebrand releases to SneppX-ALG
- Full rebrand: ARIX_Algo -> SneppX_ALG
- Fix ed25519 intrin.h, entropy_pool stdio.h, clang-tidy target order
- Fix ASM language, macOS func ptr types, clz64 order, skip broken MSVC tests
- Fix CI/CD: ASAN Debug build, macOS march, clang-tidy targets, Codecov v4, release changelog
- algo0.8.0: Infrastructure overhaul â€” build fixes, CI/CD, dev tooling, project organization
- Add Python package scaffolding, update Makefile/CI/gitignore/pre-commit [algo0.7.8 infra]
- Update README for algo0.7.8: Python bindings, PQ benchmarks, stats
- Bump version to algo0.7.8
- Add PQ crypto benchmark suite (Kyber, Dilithium, SPHINCS+) [algo0.7.8 #7]
- Enable Python bindings (pybind11) with linker + naming fixes [algo0.7.8 #6]
- Fix test projects infrastructure [algo0.7.8 #5]
- Add docs for 11 new security modules [algo0.7.8 #4]
- Fix C++ obfuscation layer errors (neural_security_cpp) [algo0.7.8 #3]
- Add CI/CD pipeline with GitHub Actions
- Add AGENTS.md for AI-assisted development workflow
- algo0.7.5: S4-S9 security additions - PQ crypto, ASM optimizations, DP, DDoS, container security, fuzzer, leak detector, breakout detection, RLHF safety
- Fix website link: aixsite -> arixsite.vercel.app
- Update README for algo0.7: S0-S9 complete, 64,589 total lines, full architecture details
- algo0.7: Complete S0-S9 security system (21,809 lines) + 64,589 total codebase
- bump version to algo0.5.4
- Fix README cd path (arix-algo -> arixalgo), update pyproject.toml version to 0.5.0
- Update VERSION file to reflect algo0.5 release
- #81 v0.5.0: ARC/NPE/FM training graphs, trainer fix, NPE training test
- #80 Fix loss computation in trainer, multi-module train graphs, and tests
- #79 Full attention training graph, reshape op, trainer fix
- Add 24 infrastructural files across 7 directories
- Remove CI/CD concepts
- README, VERSION, changelog, and 20 infra files
- Attention, inference, data pipeline, arch improvements
- Project-wide nomenclature restructuring: renamed all module identifiers to extended descriptive nomenclature for enhanced clarity and semantic precision
- Multi-head attention + RoPE + KV-cache + batched matmul
- test(gradient): conv2d finite-difference gradient verification
- L0.1 audit fixes: include paths, orphaned tests, pre-commit hook, ed25519 bit unpack, docs, gitignore
- docs: update README with skeleton infrastructure stats, fix links and back-to-top
- chore: add .gitkeep for papers/figures
- skeleton: samples directory with basic demo
- skeleton: generic library â€” rbtree, hashtable, pqueue, strutil, Rust, Python bindings
- skeleton: test suites â€” fuzz, unit, algorithm tests, HSS CUDA
- skeleton: tools â€” benchmark runner, CLI, fuzz harness, scripts
- skeleton: security language bindings â€” C, C++, C#, Go, Rust, secure allocator, integrity monitor
- skeleton: kernel internal implementations â€” autodiff, memory, optimizer, tensor, thread
- skeleton: checkpoint format, slab alloc, vmem, compression
- skeleton: network subsystem â€” socket, RDMA, gRPC, topology
- skeleton: ROCm and TPU drivers
- skeleton: CUDA driver interface
- chore: gitignore target/ and Cargo.lock
- chore: add target/ to .gitignore (Rust build artifacts)
- v0.2.0: rewrite all stubs into comprehensive implementations
- docs: update PROGRESS & README for restructure
- restructure: remove old src/ tree
- restructure: move security source to security/
- restructure: move algorithm sources to algorithms/
- restructure: move kernel source to kernel/
- restructure: move public headers to include/arix/
- T2-T5: implement stubs (inverse, det, conv1d/2d, pool1d/2d, save/load) + 54/56 tests pass, add specs
- Update README: 50/52 tests, 13 dtypes, 80+ ops
- T1: creation & shape tests + fixes
- T0: tensor audit & foundation
- Fix pre-existing benchmark errors: add seed arg to hss_model_create, fix npe instruction initializers
- Add backward gradient passes for sub, div, neg, pow and extend test coverage
- Update all GitHub URLs to ammar49-cyber/arixalgo
- Normalize all contact emails to algoarix@gmail.com
- Project infrastructure: git config, GitHub templates (no CI/CD), editorconfig, clang-format/tidy, Docker, pre-commit hooks, release scripts, security.txt, GOVERNANCE, DESIGN, STYLEGUIDE, AUTHORS, NEWS, INSTALL, COPYING; fix stale URLs
- Remove GitHub CI/CD workflow
- K0 foundation: extend NPE to 32 opcodes, expand autodiff tape/variable, optimizer factories, Python bindings (ARC/NPE/FM/SER), tensor/model/train Python API, checkpoint v2, API docs, benchmarks
- Remove black background from logo via CSS mix-blend-mode
- Resize logo to 30%
- Add ARIX logo to README
- Add back-to-top link at bottom of README
- âœ¨ Supercharged README with emojis, TOC, badges, and visual design
- Markdown documentation rewrite
- Phase 8: Training graphs (HSS multi-timestep + SER soft MoE), test suite hardening (edge cases, shared header, build config), benchmark infra (tensor + autodiff), CI workflow, CMakePresets, LTO support, Windows build scripts
- L3: HSS paper draft (LaTeX)
- Release system: scripts, VERSION, CHANGELOG, installers

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/v0.1.0...v0.9.0){ .md-button }

## v0.1.0
**2026-06-24**

_No tagged changes; showing commits up to this tag._

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.9.4.467...v0.1.0){ .md-button }

## algo0.9.4.467
**2026-07-14**

### Bug Fixes

- fix(N4): fix autocast duplicate import syntax error
- fix(N3): Optimizer saves lr, step() works without C backend, autocast is proper context manager
- fix(N2): add Trainer.train() method and fix LSTM benchmark shape
- fix(N1): MSELoss returns scalar tensor
- fix: resolve test import errors in test_trainer_v2 and test_ultra_trainer
- fix(C3): add fallback stub when C extension not available
- fix(C2): add missing Tensor import and fix matmul in benchmarking.py
- fix(C1): prevent SimpleTokenizer shadowing from .tokenizer over .data version

### Other

- bump: v0.9.4.467
- firewall: test files (24 tests), fix path traversal detection, lazy SSL context build
- firewall: CLI flags for serve_cli.py, kwargs safety in firewall constructors
- firewall: wire into inference_server.py middleware, set_security accepts firewall config
- firewall: integrate into SecurityMiddleware with check_firewall, release_concurrent
- firewall: orchestrator with YAML config, env/CLI overrides, 3-ring dispatch
- firewall: transport ring with TLS/mTLS, cert pinning, ALPN
- firewall: application ring with injection filter, path normalization, concurrent limiter
- firewall: network ring with IP CIDR filtering, rate limiting, port knock
- firewall: assembly routines for IP match, rate counter, conn track, port knock
- chore: bump version to 0.9.2.094 across all files

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.9.2.094...algo0.9.4.467){ .md-button }

## algo0.9.2.094
**2026-07-13**

### Features

- feat: add test_watermark (tests/python)
- feat: add test_security_middleware (tests/python)
- feat: add test_s5_safety (tests/python)
- feat: add test_onnx (tests/python)
- feat: add test_nccl (tests/python)
- feat: add test_inference_server (tests/python)
- feat: add test_hparams (tests/python)
- feat: add test_graph_compiler (tests/python)
- feat: add test_generation (tests/python)
- feat: add test_experiment_tracker (tests/python)
- feat: add test_distributed_wrapper (tests/python)
- feat: add test_differential_privacy (tests/python)
- feat: add test_cuda_kernels (tests/python)
- feat: add test_cuda (tests/python)
- feat: add test_checkpoint_manager (tests/python)
- feat: add test_autograd (tests/python)
- feat: add test_adversarial (tests/python)
- feat: add example (config/training)
- feat: add watermark (bindings/python/SneppX_ALG/interface_bindings)
- feat: add trainer_v3 (bindings/python/SneppX_ALG/interface_bindings)
- feat: add train_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add tokenizer (bindings/python/SneppX_ALG/interface_bindings)
- feat: add serve_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add security_middleware (bindings/python/SneppX_ALG/interface_bindings)
- feat: add s5_safety (bindings/python/SneppX_ALG/interface_bindings)
- feat: add nccl (bindings/python/SneppX_ALG/interface_bindings)
- feat: add inference_server (bindings/python/SneppX_ALG/interface_bindings)
- feat: add hparams (bindings/python/SneppX_ALG/interface_bindings)
- feat: add graph_compiler (bindings/python/SneppX_ALG/interface_bindings)
- feat: add generation (bindings/python/SneppX_ALG/interface_bindings)
- feat: add experiment_tracker (bindings/python/SneppX_ALG/interface_bindings)
- feat: add experiment_cli (bindings/python/SneppX_ALG/interface_bindings)
- feat: add distributed_wrapper (bindings/python/SneppX_ALG/interface_bindings)
- feat: add differential_privacy (bindings/python/SneppX_ALG/interface_bindings)
- feat: add cuda_kernels (bindings/python/SneppX_ALG/interface_bindings)
- feat: add cuda_device (bindings/python/SneppX_ALG/interface_bindings)
- feat: add checkpoint_manager (bindings/python/SneppX_ALG/interface_bindings)
- feat: add autograd_ops (bindings/python/SneppX_ALG/interface_bindings)
- feat: add autograd (bindings/python/SneppX_ALG/interface_bindings)
- feat: add adversarial (bindings/python/SneppX_ALG/interface_bindings)

### Bug Fixes

- fix(build): compile CUDA extension sources into existing exported targets via target_sources
- fix(build): link CUDA extension libs privately to avoid export-set errors
- fix(build): add CMakeLists.txt for algorithms/*/cuda CUDA extension libraries
- fix(ci): pin CUDA builds to ubuntu-22.04 for CUDA 12.4 repo compatibility
- fix(ci): install CUDA toolkit on runner (avoid container checkout break), mark macOS/Windows non-blocking
- fix(ci): build Linux via CUDA container, add net/distributed CMakeLists, mark macOS/Windows non-blocking
- fix: guard CUDA-only source files for CPU builds, fix CI workflow
- fix(ci): install numpy for CMake, fix markdownlint, cuda keyring
- fix(ci): remove remaining empty with block in security-scan
- fix(ci): remove submodules recursive (breaks on Arix-Site dir)
- fix(ci): remove duplicate runs-on, embedded dependabot, and other invalid YAML
- fix: remove CUDA job (needs Docker Hub secrets not set)
- fix: wrap secrets check in template expr for job if
- fix: skip CUDA Docker if Docker Hub secrets not set

### Refactor / Internal

- refactor: update advanced_ops, quantization, C tests
- refactor: update C source files (checkpoint_reader, dilithium)
- refactor: update test files and CMakeLists
- refactor: update __init__.py exports
- refactor: update benchmarks/export/distillation
- refactor: update training/infra modules
- refactor: update model/hf/vision modules
- refactor: update core tensor/nn/optim/data modules
- refactor(ci): restructure jobs, fix deps, make CUDA/security/docs non-blocking

### Other

- ci: mark C/CUDA build jobs non-blocking (pre-existing source compile errors); keep Lint/Security/Docs as real gates
- chore(ci): add workflow_dispatch for manual runs

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.9.0...algo0.9.2.094){ .md-button }

## algo0.9.0
**2026-07-11**

### Features

- feat(packaging): add comprehensive packaging for PyPI, Docker, Conda, CI/CD
- feat(models): add complete LLaMA, Mistral, Qwen2, DeepSeek V2 architectures
- feat(vision): add Vision Transformer (ViT), DeiT, Swin, MAE architectures
- feat: add benchmarking suite + all 11 tests pass
- feat: massive expansion - distillation, pruning, advanced ops, augmentation, ONNX, model zoo
- feat: distillation, pruning, advanced ops, augmentation, ONNX export
- feat(augmentation): comprehensive data augmentation pipeline
- feat(train): add AMP, gradient checkpointing, advanced DataLoader, UltraTrainer
- feat(optim): add 7 extra optimizers + 10 schedulers + 6 advanced optimizers
- feat(tensor): add advanced ops - conv, pooling, RNN, attention, norms, tensor manip
- feat(core): add SIMD GEMM + tensor expression IR with operator fusion
- feat(crypto): Dilithium DRBG-based sampling, 10-bit packing, SHA512 challenge
- feat(zoo): model configs, weight converters, from_pretrained for LLaMA2/3, Mistral, Qwen2, DeepSeek V2 + 49 tests
- feat(profile): profiler, logger, NVTX markers, compute-sanitizer CI scripts + 13 tests
- feat(ft): async checkpointing, heartbeat, elastic training, fault tolerance (C + Python) + 23 tests
- feat(quant): INT8/FP8/AWQ/GPTQ quantization kernels (C, CUDA, Python) + 17 tests
- feat(py): self-contained Tensor with optional C backend, update exports and pip package config
- feat(py): implement neural network, optimizers, data, distributed, hf, model, train modules
- feat(arch): Mamba-2 selective SSM with HiPPO initialization, conv1d, discretized scan
- feat(arch): FlexAttention block-sparse with mask modulation, multi-modal cross-attention, MoD, gated activations, YaRN NTK-RoPE, ALiBi
- feat(arch): Differential Attention and Multi-head Latent Attention (DeepSeek MLA style)
- feat(arch): advanced architectures master header (DifferentialAttn, MLA, FlexAttn, Mamba2, MoD, YaRN, ALiBi)
- feat(dist): distributed sampler with epoch-based shuffling and gradient accumulation manager
- feat(dist): distributed checkpoint coordinator with async save and fault tolerance
- feat(dist): hierarchical all-reduce (NVLink+RDMA) and Top-K gradient compression with error feedback
- feat(dist): DDP with bucket-based gradient all-reduce and compute overlap
- feat(dist): expert parallelism all-to-all dispatch and FM distributed communication
- feat(dist): tensor parallelism row/column split linear with all-reduce
- feat(dist): pipeline parallelism 1F1B schedule with microbatches
- feat(dist): ZeRO-1/2/3 optimizer state partitioning and fused AdamW step
- feat(dist): distributed training config header (ZeRO, pipeline, tensor, expert parallel)
- feat(dist): NCCL dynamic loading, all-reduce, process group part 2/2
- feat(dist): NCCL dynamic loading, all-reduce, process group part 1/2
- feat(dist): NCCL communication primitives header
- feat(hss): extended SSM step, conv, selective scan kernels
- feat(arc): PGD/FGSM attacks, gradient obfuscation, smoothing
- feat(arc): adversarial robustness CUDA header
- feat(npe): differentiable program execution GPU kernel part 2/2
- feat(npe): differentiable program execution GPU kernel part 1/2
- feat(npe): neural VM instruction dispatch CUDA header
- feat(fm): ring/butterfly all-reduce, gradient quantization part 2/2
- feat(fm): ring/butterfly all-reduce, gradient quantization part 1/2
- feat(fm): all-reduce, quantization, federated avg CUDA header
- feat(ser): fused MoE forward, load balancing loss
- feat(ser): top-k gating, dispatch, combine kernels
- feat(ser): top-k gating and fused MoE CUDA header
- feat(hss): Mamba/S6 selective scan, SSM conv, HiPPO matrix CUDA kernels
- feat(hss): selective scan, S4, HiPPO CUDA header
- feat(cuda): RNG API header (Philox, distributions, init schemes)
- feat(cuda): random number generation kernels part 3/3
- feat(cuda): random number generation kernels part 2/3
- feat(cuda): random number generation kernels part 1/3
- feat(cuda): memory pool/stream/event API header
- feat(cuda): memory management (pool, streams, events) part 3/3
- feat(cuda): memory management (pool, streams, events) part 2/3
- feat(cuda): memory management (pool, streams, events) part 1/3
- feat(cuda): optimizer header with all step types
- feat(cuda): fused optimizer kernels part 3/3
- feat(cuda): fused optimizer kernels part 2/3
- feat(cuda): fused optimizer kernels part 1/3
- feat(cuda): autodiff backward header declarations
- feat(cuda): autodiff backward kernels part 4/4
- feat(cuda): autodiff backward kernels part 3/4
- feat(cuda): autodiff backward kernels part 2/4
- feat(cuda): autodiff backward kernels part 1/4
- feat(cuda): attention kernels part 5/5
- feat(cuda): attention kernels part 4/5
- feat(cuda): attention kernels part 3/5
- feat(cuda): attention kernels part 2/5
- feat(cuda): attention kernels part 1/5
- feat: expand assembly code to 3258 lines with constant-time, speculation-safe, cache-resistant security improvements across 15 files
- feat: security infrastructure expansion - compliance, PQ crypto, threat intel, IR, automation, ZT, chaos, AI guardrails, supply chain, fuzzing + docs

### Bug Fixes

- fix(docker): make CPU Dockerfile self-contained, fix packages workflow
- fix(version): bump setup.py to 0.9.0 to match pyproject.toml and tag
- fix(crypto): Kyber uses DRBG for deterministic noise sampling instead of rand()
- fix: full build cleanup â€” fix 8 test files for API changes, stub sha512, fix audit_logger stdarg, fix secure_wipe ABI, exclude unimplemented tests
- fix(asm): rewrite poly1305_sse.asm with correct 5-limb scalar algorithm, fix 8 ABI signature mismatches in asm_exports.h
- fix(asm): fix 6 ABI signature mismatches in asm_exports.h header
- fix(asm): fix 5 assembly correctness bugs
- fix(crypto): fix 7 critical bugs - auth bypasses, memory overflows, X25519, Kyber
- fix: update metadata to align with sneppx-alg identity
- fix: update build.ps1 CMake flags to SNEPPX_ prefix
- fix: rename arix_algo package to SneppX_ALG, fix Python relative imports
- fix: sha3 finalize->finish, fm_forward 4 args, m.lib MSVC guard
- fix: add csignal, time.h, link m for math in kernel lib
- fix: move sys/prctl.h to __linux__ guard, ifdef s9_extensions _finddata for WIN32
- fix: s9_extensions io.h guard, blake3 finalize->finish across 4 files
- fix: use asm/unistd.h for __NR_* in seccomp filter
- fix: memory_hardening sys/syscall.h, hmac.c sha512 func name
- fix: memory_hardening linux headers, drbg sha512 func name, asm only on MSVC
- fix: ed25519 _umul128 compat + point_is_on_curve ordering; fix cache.c aarch64 prfm syntax

### Other

- ci(packages): add GitHub Packages workflow for Docker and Python publishing
- chore: update Arix-Site submodule reference after cleanup
- test(crypto): add Kyber debug output for pk/sk/shared-secret bytes
- docs: update AGENTS.md with Phases 5-8, deprecate old lib/python bindings
- test(py): add 7 Python API test suites (82 tests)
- test(cuda): GPU kernel test suite part 3/3
- test(cuda): GPU kernel test suite part 2/3
- test(cuda): GPU kernel test suite part 1/3
- docs: update AGENTS.md with Phase 1 CUDA backend overview
- chore: bump version to 0.8.6 for 12 critical bug fixes across C and asm
- Fix ASM_MASM compiler (ml64.exe) and add asm_exports.h header
- Update README for algo0.8.2: assembly stats, version bump, S0 detail
- Fix MASM syntax errors: all 15 asm files assemble cleanly
- Fix assembly bugs across 8 files
- Remove CI/CD workflows, rebrand releases to SneppX-ALG
- Full rebrand: ARIX_Algo -> SneppX_ALG
- Fix ed25519 intrin.h, entropy_pool stdio.h, clang-tidy target order
- Fix ASM language, macOS func ptr types, clz64 order, skip broken MSVC tests
- Fix CI/CD: ASAN Debug build, macOS march, clang-tidy targets, Codecov v4, release changelog

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.8.0...algo0.9.0){ .md-button }

## algo0.8.0
**2026-07-08**

### Other

- algo0.8.0: Infrastructure overhaul â€” build fixes, CI/CD, dev tooling, project organization
- Add Python package scaffolding, update Makefile/CI/gitignore/pre-commit [algo0.7.8 infra]
- Update README for algo0.7.8: Python bindings, PQ benchmarks, stats

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.7.8...algo0.8.0){ .md-button }

## algo0.7.8
**2026-07-08**

### Other

- Bump version to algo0.7.8
- Add PQ crypto benchmark suite (Kyber, Dilithium, SPHINCS+) [algo0.7.8 #7]
- Enable Python bindings (pybind11) with linker + naming fixes [algo0.7.8 #6]
- Fix test projects infrastructure [algo0.7.8 #5]
- Add docs for 11 new security modules [algo0.7.8 #4]
- Fix C++ obfuscation layer errors (neural_security_cpp) [algo0.7.8 #3]
- Add CI/CD pipeline with GitHub Actions
- Add AGENTS.md for AI-assisted development workflow

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.7.5...algo0.7.8){ .md-button }

## algo0.7.5
**2026-07-07**

### Other

- algo0.7.5: S4-S9 security additions - PQ crypto, ASM optimizations, DP, DDoS, container security, fuzzer, leak detector, breakout detection, RLHF safety
- Fix website link: aixsite -> arixsite.vercel.app
- Update README for algo0.7: S0-S9 complete, 64,589 total lines, full architecture details
- algo0.7: Complete S0-S9 security system (21,809 lines) + 64,589 total codebase
- bump version to algo0.5.4
- Fix README cd path (arix-algo -> arixalgo), update pyproject.toml version to 0.5.0
- Update VERSION file to reflect algo0.5 release

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/compare/algo0.5...algo0.7.5){ .md-button }

## algo0.5
**2026-07-01**

### Features

- feat(tokenizer): BPE tokenizer with train/encode/decode/save/load
- feat(autodiff): gradient checkpointing + view-aware storage
- feat(autodiff): min/max with selective gradient routing
- feat(autodiff): ref-counted backward lifecycle with ctx cleanup

### Bug Fixes

- fix: sign scheme S=r+h*a mod L with proper sc_reduce64
- fix: fe_to_bytes final p-subtraction via limb comparison
- fix: fe_mul rewrite with _umul128 128-bit arithmetic
- fix: point_add formula (X3=EÂ·F, Y3=GÂ·H, T3=EÂ·H, Z3=FÂ·G)
- fix: point_double formula (Y3=EÂ·G, T3=FÂ·G, Z3=DÂ·H) + identity guard
- fix: fe_sub per-limb bias constants
- fix: point_scalar_mult cswap mask uint8_t->uint64_t
- fix: add two-round carry chain to fe_to_bytes
- fix: add shebangs to build.sh, test.sh, clean.sh

### Other

- #81 v0.5.0: ARC/NPE/FM training graphs, trainer fix, NPE training test
- #80 Fix loss computation in trainer, multi-module train graphs, and tests
- #79 Full attention training graph, reshape op, trainer fix
- Add 24 infrastructural files across 7 directories
- Remove CI/CD concepts
- README, VERSION, changelog, and 20 infra files
- Attention, inference, data pipeline, arch improvements
- Project-wide nomenclature restructuring: renamed all module identifiers to extended descriptive nomenclature for enhanced clarity and semantic precision
- Multi-head attention + RoPE + KV-cache + batched matmul
- test(gradient): conv2d finite-difference gradient verification
- L0.1 audit fixes: include paths, orphaned tests, pre-commit hook, ed25519 bit unpack, docs, gitignore
- docs: update README with skeleton infrastructure stats, fix links and back-to-top
- chore: add .gitkeep for papers/figures
- skeleton: samples directory with basic demo
- skeleton: generic library â€” rbtree, hashtable, pqueue, strutil, Rust, Python bindings
- skeleton: test suites â€” fuzz, unit, algorithm tests, HSS CUDA
- skeleton: tools â€” benchmark runner, CLI, fuzz harness, scripts
- skeleton: security language bindings â€” C, C++, C#, Go, Rust, secure allocator, integrity monitor
- skeleton: kernel internal implementations â€” autodiff, memory, optimizer, tensor, thread
- skeleton: checkpoint format, slab alloc, vmem, compression
- skeleton: network subsystem â€” socket, RDMA, gRPC, topology
- skeleton: ROCm and TPU drivers
- skeleton: CUDA driver interface
- chore: gitignore target/ and Cargo.lock
- chore: add target/ to .gitignore (Rust build artifacts)
- v0.2.0: rewrite all stubs into comprehensive implementations
- docs: update PROGRESS & README for restructure
- restructure: remove old src/ tree
- restructure: move security source to security/
- restructure: move algorithm sources to algorithms/
- restructure: move kernel source to kernel/
- restructure: move public headers to include/arix/
- T2-T5: implement stubs (inverse, det, conv1d/2d, pool1d/2d, save/load) + 54/56 tests pass, add specs
- Update README: 50/52 tests, 13 dtypes, 80+ ops
- T1: creation & shape tests + fixes
- T0: tensor audit & foundation
- Fix pre-existing benchmark errors: add seed arg to hss_model_create, fix npe instruction initializers
- Add backward gradient passes for sub, div, neg, pow and extend test coverage
- Update all GitHub URLs to ammar49-cyber/arixalgo
- Normalize all contact emails to algoarix@gmail.com
- Project infrastructure: git config, GitHub templates (no CI/CD), editorconfig, clang-format/tidy, Docker, pre-commit hooks, release scripts, security.txt, GOVERNANCE, DESIGN, STYLEGUIDE, AUTHORS, NEWS, INSTALL, COPYING; fix stale URLs
- Remove GitHub CI/CD workflow
- K0 foundation: extend NPE to 32 opcodes, expand autodiff tape/variable, optimizer factories, Python bindings (ARC/NPE/FM/SER), tensor/model/train Python API, checkpoint v2, API docs, benchmarks
- Remove black background from logo via CSS mix-blend-mode
- Resize logo to 30%
- Add ARIX logo to README
- Add back-to-top link at bottom of README
- âœ¨ Supercharged README with emojis, TOC, badges, and visual design
- Markdown documentation rewrite
- Phase 8: Training graphs (HSS multi-timestep + SER soft MoE), test suite hardening (edge cases, shared header, build config), benchmark infra (tensor + autodiff), CI workflow, CMakePresets, LTO support, Windows build scripts
- L3: HSS paper draft (LaTeX)
- Release system: scripts, VERSION, CHANGELOG, installers
- Add release artifacts to gitignore
- Update docs with new vision: verifiable inference, on-device, federated contribution, safety guarantees
- S2: obfuscation engine (C++) â€” CFG flattening, string encryption, instruction substitution, opaque predicates, code VM, anti-debug, pipeline, tests, demo
- Add x86_64 assembly: ed25519 scalarmult, constant-time cmp, SC cmov ops
- Phase 7: S1 - Secure Memory & Side-Channel Resistance
- Phase 7: Security - AES-free crypto suite (ChaCha20-Poly1305 AEAD, Ed25519, Argon2, SHA-3, BLAKE3)
- Phase 6: Training Loop & Python Bindings
- Phase 5: FM - Federated Memory
- Phase 4: NPE - Neural Program Executor
- Phase 3: ARC - Adversarial Robustness Core - input guard with anomaly detection, gradient obfuscator with noise+clamp, output verifier with consistency history, attack simulation (FGSM/PGD/CW), multi-layer model, tests, integration with HSS+SER, demo
- Phase 2: SER - Sparse Expert Routing - expert create/destroy, routing with softmax+top-k, expert forward (ReLU/GELU/Swish), layer forward with gather/scatter, load balance loss, multi-layer model, tests, integration test with HSS, demo
- Phase 1: HSS forward pass - layer create/destroy, model create/destroy, discretization, single step, seq scan, hierarchical scan stub, forward pass with layer norm and input projection, test suite, demo
- Add .gitignore, remove build artifacts from tracking
- Fix CMakeLists.txt for MSVC compatibility, update README
- Initial foundation: tensor ops, memory allocator, thread pool stub, CMake build system, C test suite, Python/Rust/C security stubs, MIT license
- Initial commit

[:material-git-compare: `View diff vs previous`](https://github.com/SneppX/sneppx-alg/commits/algo0.5){ .md-button }

---

Each release header links to a **GitHub compare view** (`{prev}...{tag}`) which renders a side-by-side **visual diff** of the source tree between releases. Tag the repo (`git tag v1.2.0`) and re-run `python docs/changelog/generate.py` to refresh.

