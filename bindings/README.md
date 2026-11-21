# Python Bindings — `bindings/python/`

The Python bindings connect the C/CUDA engine with Python via **pybind11**. The compiled
extension module `_SNEPPX_c` is loaded by the `SneppX_ALG` package to expose tensor operations,
autograd, optimizers, models, security primitives, and algorithms to Python.

## Architecture

```
bindings/python/
├── bindings.cpp                  # PYBIND11_MODULE entry point (includes split module files)
├── bindings/                     # Split module files
│   ├── tensor_bindings.cpp       # PyTensor wrapper (factory, ops, reductions, NN, loss)
│   ├── autograd_bindings.cpp     # PyVariable + PyTape (autograd engine bindings)
│   ├── optim_model_bindings.cpp  # PyOptimizer, PyLRScheduler, PyModel, config structs
│   ├── npe_extra_bindings.cpp    # PyNPEVM, PyNPEProgram, PyFMController, PyTrainer, pools
│   └── crypto_bindings.cpp       # Crypto function lambdas (ChaCha20, SHA, Ed25519, etc.)
├── SneppX_ALG/                   # Python package directory
│   ├── __init__.py               # Package root
│   ├── tensor.py                 # Pure-Python Tensor fallback + C bridge
│   ├── nn.py                     # Neural network layers
│   ├── optim.py                  # Optimizers
│   ├── train.py                  # Training loop
│   ├── data.py                   # Data loading
│   ├── memory.py                 # Memory pool wrappers
│   ├── distributed.py            # Distributed training helpers
│   ├── quantization.py           # Quantization (INT8/FP8/AWQ/GPTQ)
│   ├── checkpoint.py             # Async checkpoint, heartbeat, elastic training
│   ├── profiler.py               # Profiling and timing
│   ├── model_zoo.py              # Model configs, from_pretrained(), weight converters
│   ├── crypto_sign.py            # Digital signature wrappers
│   ├── crypto_kem.py             # KEM wrappers
│   ├── crypto_symmetric.py       # Symmetric crypto wrappers
│   ├── crypto_hash.py            # Hash function wrappers
│   ├── crypto_kdf.py             # KDF wrappers
│   ├── crypto_util.py            # Utility wrappers
│   ├── secure_memory.py          # S1 memory hardening wrappers
│   ├── runtime_monitoring.py     # S3 monitoring wrappers
│   ├── network_security.py       # S4 network security wrappers
│   ├── ai_safety.py              # S5 AI safety wrappers
│   ├── container_security.py     # S7 container security wrappers
│   ├── key_vault.py              # UI key vault
│   ├── audit_logger.py           # UI audit logging
│   ├── formal_verification.py    # S8 formal verification wrappers
│   ├── penetration_testing.py    # S9 pentesting wrappers
│   ├── obfuscation_pipeline.py   # S2 obfuscation wrappers
│   ├── ser_model.py              # Sparse Expert Routing wrapper
│   ├── hss_model.py              # Hierarchical State Space wrapper
│   ├── arc_layer.py              # Adversarial Robustness wrapper
│   ├── npe_compiler.py           # Neural Program Engine wrapper
│   ├── fm_controller.py          # Fractal Memory wrapper
│   ├── attention.py              # Attention mechanism wrappers
│   ├── memory_pool.py            # Memory pool wrapper
│   ├── simd_gemm.py              # SIMD GEMM wrapper
│   ├── security_config.py        # Security configuration
│   ├── security_middleware.py    # Security middleware (4-ring firewall)
│   ├── interface_bindings/       # CLI interface modules
│   │   ├── train_cli.py          # sneppx-train command
│   │   ├── serve_cli.py          # sneppx-serve command
│   │   └── experiment_cli.py     # sneppx-experiment command
│   └── ... (additional modules)
├── setup.py                      # Package setup
└── CMakeLists.txt                # Build config
```

## Phases

The bindings were developed in 8 phases:

| Phase | Focus | Modules |
|-------|-------|---------|
| 1 | Foundation | c_loader, c_types, dispatch factory |
| 2 | Crypto | 6 families (sign, kem, symmetric, hash, kdf, util) |
| 3 | Security S1-S9 | 10 modules (memory, network, monitor, vault, updates, verify, pen-test, obfuscation, AI safety, middleware) |
| 4 | Algorithms | ARC, FM, HSS, NPE, SER |
| 5 | Kernels | attention, arch, memory, thread, tensor_expr, simd_gemm, logger |
| 6 | Infrastructure | net (topology, RDMA, gRPC, NCCL), drivers (CUDA/ROCm/TPU) |
| 7 | ASM | CPU features, crypto assembly (AES-NI, SHA-NI, AVX2), MASM build |
| 8 | Model Zoo | from_pretrained(), weight converters, JSON configs |

## Build

The bindings are built by the top-level CMake with `-DSNEPPX_BUILD_PYTHON=ON`:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_PYTHON=ON
cmake --build build --config Release --target _SNEPPX_c
```

Or via pip (uses pre-built extension or builds if C toolchain available):
```bash
pip install -e .
```

## Pure-Python Fallback

All modules in `SneppX_ALG/` have pure-Python fallbacks. When the C extension (`_SNEPPX_c`)
is not available, operations degrade gracefully using NumPy-based implementations.
This allows development and testing without a C compiler.

## API Design

The Python API mirrors PyTorch conventions where possible:
- `Tensor` — Multi-dimensional array with NumPy interop
- `Variable` — Autograd wrapper (requires_grad, backward)
- `nn.Linear`, `nn.Embedding`, `nn.Dropout`, etc. — Neural network layers
- `optim.AdamW`, `optim.SGD`, etc. — Optimizers
- `Trainer` / `TrainConfig` — Training pipeline
- `from_pretrained()` — HuggingFace-compatible model loading
