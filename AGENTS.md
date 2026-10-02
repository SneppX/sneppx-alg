# SNEPPX Algo — Agent Guide

## 🚫 Machine Constraints (CRITICAL — Read Before Any LLM Work)

- **No local LLM inference**: This machine has no GPU and limited RAM. NEVER run, test, or validate any LLM model tests (`test_llama_models.py`, `test_model_implementations.py`, `test_vision_transformers.py`, `test_inference_server.py`, `test_generation.py`) — they will hang or crash the machine.
- **No CUDA**: CUDA tests (`test_cuda.py`, `test_cuda_kernels.py`) cannot run here. Skip them.
- **No GPU-dependent ops**: Anything requiring a CUDA device or large model weights (>500MB) should not be executed.
- **Python-only testing**: Only run Python-level unit tests that use NumPy/Tensor backends. All tests under `tests/python/` that don't import LLM or CUDA modules are safe.
- **Safe test list** (always run these for regression): `test_tensor.py`, `test_nn.py`, `test_optim.py`, `test_data.py`, `test_quantization.py`, `test_checkpoint.py`, `test_profiler.py`, `test_model_zoo.py`, `test_autograd_ops.py`, `test_schedulers.py`, `test_amp.py`, `test_grad_checkpoint.py`, `test_tokenizer.py`, `test_data_loader.py`, `test_advanced_ops.py`, `test_augmentation.py`, `test_pruning.py`, `test_distillation.py`, `test_higher_order_ad.py`, `test_higher_order_ad2.py`, `test_higher_order_ad3.py`, `test_higher_order_ad4.py`, `test_higher_order_ad5.py`, `test_jvp_vjp.py`, `test_gap_fill.py`, `test_norm.py`, `test_functional_ns.py`, `test_state_dict.py`, `test_sparse.py`, `test_phase2_ops.py`, `test_losses.py`, `test_optimizers.py`, `test_nn_modules.py`, `test_ctc.py`, `test_init.py`, `test_grad_checkpoint.py`.
- **NEVER attempt**: `from transformers import ...`, `from llama_cpp import ...`, `torch.cuda.is_available()`, or any HuggingFace `from_pretrained()` call.

## 🚫 No CI/CD Policy (CRITICAL)

**Automated CI/CD is NOT allowed in this project. All builds, tests, and releases are done manually.**

- NEVER create or commit any CI/CD workflow files (`.github/workflows/*.yml`, `.github/CODEOWNERS`, CI configs)
- NEVER push CI/CD configuration of any kind
- All verification is local: build with `cmake --build`, test with `ctest`, lint with `pre-commit`
- Releases are tagged and pushed manually — no automated pipelines

## Python Execution

**Required env setup:**

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest <file> -q -p no:cacheprovider
```

- `pyproject.toml` discovers packages under `bindings/python`
- `sneppx-train`, `sneppx-serve`, `sneppx-eval`, `sneppx-quantize`, `sneppx-rlhf` CLI scripts are registered via `pyproject.toml [project.scripts]`
- `sx.functional` name is **RESERVED** by the pre-existing autograd functional module (JVP/VJP) — do NOT shadow; use `F` alias instead (`from SneppX_ALG.interface_bindings import nn_functional` if raw module needed)
- `sx.nn_functional` is **NOT available** — only `F` is exported in `__all__`; import `from SneppX_ALG.interface_bindings import nn_functional` for the raw module

## Key Entry Points (bindings/python/SneppX_ALG/interface_bindings/)

| Module | Purpose |
|---|---|
| `tensor.py` | Tensor class with autograd, requires_grad, backward, autograd ops |
| `nn.py` | NN layers (Linear, Conv1d/2d/3d, RMSNorm, LayerNorm, BatchNorm1d/2d/3d, InstanceNorm1d/2d/3d, etc.), Module, ModuleList, ModuleDict |
| `optim.py` | Optimizers: SGD, Adam, AdamW, Adamax, RMSprop, Adagrad, Adadelta, Adamax, NAdam, Rprop, ASGD, LBFGS, SparseAdam |
| `autograd.py` | Tape-based AD: `no_grad`, `set_grad_enabled`, `enable_grad`, `Function` base class, `Context` |
| `autograd_ops.py` | Differentiable Functions: Add, Sub, Mul, Div, MatMul, Transpose, Sum, Mean, Relu, Sigmoid, Tanh, Sqrt, Exp, Log, Pow, Reshape, GetItem, Square, Reciprocal, Sign, Clamp, Remainder, Prod, LogSumExp, Trace, Diagonal, MinDim, MaxDim, Neg, etc. |
| `nn_functional.py` | `F.*` functional namespace (~70 ops): activations, linear/conv/pool norm, interpolation, pixel/channel shuffle, one_hot, pad_sequence |
| `sparse.py` | `SparseTensor` (COO/CSR/CSC/BSR), `sparse_mm`, `sparse_addmm`, `sparse_sum`, `sparse_softmax` |
| `grad_checkpoint.py` | `checkpoint`, `checkpoint_sequential`, `CheckpointFunction`, `GradientCheckpointer` |
| `init.py` | `zeros_`, `ones_`, `constant_`, `eye`, `uniform`, `normal`, `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `orthogonal`, `calculate_gain` |
| `advanced_ops.py` | conv1d/conv2d/conv3d, max/avg pool2d, adaptive pool, linear, instance/batch/layer/rms norm |

## Test Commands (per-file only; full-suite HANGS)

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_tensor.py -q -p no:cacheprovider
python -m pytest tests/python/test_nn.py -q -p no:cacheprovider
python -m pytest tests/python/test_optim.py -q -p no:cacheprovider
python -m pytest tests/python/test_gap_fill.py -q -p no:cacheprovider
python -m pytest tests/python/test_losses.py -q -p no:cacheprovider
python -m pytest tests/python/test_optimizers.py -q -p no:cacheprovider
python -m pytest tests/python/test_nn_modules.py -q -p no:cacheprovider
python -m pytest tests/python/test_functional_ns.py -q -p no:cacheprovider
python -m pytest tests/python/test_state_dict.py -q -p no:cacheprovider
python -m pytest tests/python/test_sparse.py -q -p no:cacheprovider
python -m pytest tests/python/test_higher_order_ad.py -q -p no:cacheprovider
python -m pytest tests/python/test_higher_order_ad2.py -q -p no:cacheprovider
python -m pytest tests/python/test_higher_order_ad3.py -q -p no:cacheprovider
python -m pytest tests/python/test_higher_order_ad4.py -q -p no:cacheprovider
python -m pytest tests/python/test_higher_order_ad5.py -q -p no:cacheprovider
python -m pytest tests/python/test_jvp_vjp.py -q -p no:cacheprovider
python -m pytest tests/python/test_norm.py -q -p no:cacheprovider
python -m pytest tests/python/test_phase2_ops.py -q -p no:cacheprovider
python -m pytest tests/python/test_ctc.py -q -p no:cacheprovider
python -m pytest tests/python/test_init.py -q -p no:cacheprovider
python -m pytest tests/python/test_grad_checkpoint.py -q -p no:cacheprovider
```

## Build / CMake (optional; pure-Python path is default)

```powershell
# Configure (debug) — prefer repo-root ninja over venv (venv ninja crashes)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Configure (release)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build all
cmake --build build --config Release

# Build specific target
cmake --build build --config Release --target neural_security_c

# Run tests
cd build && ctest -C Release --output-on-failure

# Run a specific test
ctest -C Release -R test_kyber --output-on-failure
```

> **Windows + Ninja note.** This machine ships a Kitware *dev-build* Ninja in the venv (`1.13.0.git.kitware.jobserver-pipe`) which crashes with heap corruption. Prefer the stable repo-root `ninja.exe` (Ninja ≥ 1.12), e.g. `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`. CMake resolves the x64 MASM assembler automatically (`ml64` next to `cl.exe`) — no manual `-DCMAKE_ASM_MASM_COMPILER` is required. If venv Ninja must be used, fall back to the legacy Visual Studio generator (`cmake -B build -G "Visual Studio 17 2022"`).

## Coding Conventions

- `SNEPPX_` prefix for all public functions, types, and macros
- `void` in empty parameter lists: `int foo(void)` not `int foo()`
- Return `int` (0 success, -1 error) for most API functions
- `size_t` for lengths/counts, `uint8_t*` for byte buffers
- Python: `Tensor.data` is a **COPY** (storage in `self._data`); FD harness must rebuild perturbed Tensors, not mutate `.data`
- Python: `Module.__setattr__` auto-registers Tensor attrs as params with `requires_grad_(True)`
- Python: Conv1d `pad_width` axis-order bug fix: `pad_width = [(0,0)]*(ndim-npads) + list(reversed(pairs))` (torch tuple is last-dim-first)
- C: No VLAs (MSVC C11 doesn't support them — use `calloc`/`free`); No `__int128` on MSVC without `#ifndef NO_UINT128` guards
- Python: `sx.functional` is reserved — use `F` alias; `Tensor.reshape(*shape)` uses splat, not tuple

## Repo History (Last Committed State)

- `8dc8846` — repo relocated `C:\Users\PC\sneppx-ultra\ARIX_Algo` → `C:\Users\PC\SneppX\sneppx-alg` (git + uncommitted work preserved)
- `c9a0a9f` — P3 Phase-2 ops: `Square`/`Reciprocal`/`Sign`/`Clamp`/`Remainder`/`Prod` + `tensor.py` methods; scalar `(1,)` quirk fix
- `39cdb1a` — P3 sparse: N-D `from_tensor`, `sparse_conv2d`, `to_sparse_coe/csr/csc/bsr` methods, `SparseAdam` real sparse-step
- `9ba674d` — remote `origin/main` merge resolve (resolved `README.md` conflict + harmless dep bumps); all 235 regression tests green post-merge
- Last confirmed remote push: `c337ca3` (previous generation)
- PUSH PENDING — remote main owned by concurrent session; do not force-push. After session stops: `git fetch; git rebase origin/main; git push`

## Critical Workflow: Lint → Typecheck → Test

Order matters for regression safety:

1. Run per-file pytest on changed module (e.g., `test_gap_fill.py`, `test_norm.py`, `test_higher_order_ad.py`)
2. If module touches C/ctypes, verify `ctest` for the specific target only
3. Do NOT run full test suite — it hangs; per-file only is the gate