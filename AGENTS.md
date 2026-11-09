# SNEPPX Algo — Agent Guide

## 🚫 Machine Constraints (CRITICAL — Read Before Any LLM Work)

- **No local LLM inference**: This machine has no GPU and limited RAM. NEVER run, test, or validate any LLM model tests (`test_llama_models.py`, `test_model_implementations.py`, `test_vision_transformers.py`, `test_inference_server.py`, `test_generation.py`) — they will hang or crash the machine.
- **No CUDA**: CUDA tests (`test_cuda.py`, `test_cuda_kernels.py`) cannot run here. Skip them.
- **No GPU-dependent ops**: Anything requiring a CUDA device or large model weights (>500MB) should not be executed.
- **Python-only testing**: Only run Python-level unit tests that use NumPy/Tensor backends. All tests under `tests/python/` that don't import LLM or CUDA modules are safe.
- **Safe test list** (always run these for regression): `test_tensor.py`, `test_nn.py`, `test_optim.py`, `test_data.py`, `test_quantization.py`, `test_checkpoint.py`, `test_profiler.py`, `test_model_zoo.py`, `test_autograd_ops.py`, `test_schedulers.py`, `test_amp.py`, `test_grad_checkpoint.py`, `test_tokenizer.py`, `test_data_loader.py`, `test_advanced_ops.py`, `test_augmentation.py`, `test_pruning.py`, `test_distillation.py`, `test_higher_order_ad.py`, `test_higher_order_ad2.py`, `test_higher_order_ad3.py`, `test_higher_order_ad4.py`, `test_higher_order_ad5.py`, `test_jvp_vjp.py`, `test_gap_fill.py`, `test_norm.py`, `test_functional_ns.py`, `test_state_dict.py`, `test_sparse.py`, `test_phase2_ops.py`, `test_losses.py`, `test_optimizers.py`, `test_nn_modules.py`, `test_ctc.py`, `test_init.py`, `test_grad_checkpoint.py`, `test_scalar_ops.py`, `test_loss_diff.py`, `test_functional_diff.py`, `test_activation_autograd.py`, `test_trainable.py`, `test_conv2d_autograd.py`, `test_embedding_layernorm_autograd.py`.
- **NEVER attempt**: `from transformers import ...`, `from llama_cpp import ...`, `torch.cuda.is_available()`, or any HuggingFace `from_pretrained()` call.

## CI/CD & Pre-Commit Pipeline Gate (CRITICAL)

CI/CD pipelines are configured in `.github/workflows/`. **Before every commit in any repo/work session:**

1. Run the local checks that mirror CI (or the affected subset) and ensure **every required pipeline/check passes**.
2. If any check fails, **fix the failure** (code, tests, config, or docs) and re-run until green.
3. Only commit when required checks pass. Do not commit known-broken pipeline state.
4. Do not bypass failures with `--no-verify` unless the user explicitly requests it and the reason is documented.

**Local checks (match CI where feasible):**
```powershell
$env:PYTHONPATH = "bindings/python"
# Safe per-file Python tests (never full suite; never LLM/CUDA/inference-server tests on constrained machines)
py -3 -m pytest tests/python/test_readme_integrity.py tests/python/test_hf_safetensors.py tests/python/test_hf_integration.py tests/python/test_model_zoo.py tests/python/test_data_loader.py tests/python/test_tokenizer.py -q -p no:cacheprovider
# Convenience: .\scripts\precommit-gate.ps1
# Optional native build/test when toolchain is available (targeted):
# cmake --preset release; cmake --build --preset release; ctest --preset release -R <safe-target>
```

**Machine constraints still apply:** no local LLM inference, no HF model downloads, no GPU/CUDA-dependent execution, no large model weights (>500MB) on constrained machines. CI jobs must respect these constraints (skip or mock LLM/GPU paths; run CPU-safe tests only).

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

Machine constraint: no Ninja build program and no C/C++/MASM compiler are configured on this constrained machine, so `cmake -G Ninja ...` fails (`CMAKE_MAKE_PROGRAM`, `CMAKE_C_COMPILER`, `CMAKE_ASM_MASM_COMPILER` not set). Native C build/tests are **not runnable here** and must not block commits. Install a toolchain first if you need native C (HTTP server, crypto, formal):

```powershell
# Install Ninja + a C compiler (choose ONE):
#   winget install Ninja.Ninja
#   winget install LLVM.LLVM        # provides clang/clang++ (C/C++ compiler)
# or install "Visual Studio 17 2022" Build Tools (provides cl.exe + Ninja via VS)

# Configure (debug) — prefer repo-root stable ninja (not venv kitware ninja, which crashes)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Configure (release) — VS fallback when Ninja is absent but VS Build Tools exist
cmake -B build/vs -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_HTTP=ON -DSNEPPX_BUILD_CUDA=OFF

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

## C Build Reality (this constrained machine)

- `cmake` is present but **no Ninja build program and no C/C++/MASM compiler are configured**, so `cmake -G Ninja ...` fails (`CMAKE_MAKE_PROGRAM`, `CMAKE_C_COMPILER`, `CMAKE_ASM_MASM_COMPILER` not set).
- Native C build/tests are **not runnable here**. Keep `.github/workflows/ci.yml` `native-build-smoke` as best-effort (`continue-on-error`); do not let it block commits.
- Real C HTTP inference (`/v1/generate`, `/v1/completions`) is an **explicit reference scaffold** (`mode:"reference_scaffold"`, `runtime:"c_reference_no_weights"`). Making it real requires a C tokenizer + safetensors/bin weight loader + C model forward/logits + sampling — large C integration, not implementable/verifiable without a working C toolchain.

## Critical Workflow: Lint → Typecheck → Test → Commit (GATE)

Before **every** commit in any repo/work session:

1. Run `.\scripts\precommit-gate.ps1` (safe CPU per-file tests). Gate must pass (green).
2. If native C toolchain is available, run targeted `ctest` for changed C modules only.
3. If any check fails → fix and rerun until green → then commit. Do not `--no-verify` unless explicitly requested/documented.
4. Only commit on green. Do not force-push; sync with `git fetch; git rebase origin/main; git push` before pushing.

## Python Execution

## Docs Site (Material 9.5.3 + mike)

- **Dark/light toggle** lives in `theme.palette` (list of two schemes: `default`/`indigo`/`deep purple` and `slate`/`indigo`/`amber`). It MUST be under `theme:` — `extra.palette` is silently ignored (was the original bug). Persistence is automatic (localStorage `__palette`).
- **Custom assets** are wired via `extra_css: [stylesheets/extra.css]` and `extra_javascript: [javascripts/extra.js]`.
  - `docs/stylesheets/extra.css`: scroll-reveal, word-by-word heading animation, `.sx-hero` gradient, `.sx-stat`/`.sx-card` polish, `.sx-progress` bar, dark-mode logo, and a `prefers-reduced-motion` kill-switch block.
  - `docs/javascripts/extra.js`: `IntersectionObserver` scroll reveal, heading word-splitter, stat count-up (`[data-count]` / `data-suffix`), fixed scroll progress bar. No external deps.
- **Home page** (`docs/index.md`) opens with an `md_in_html` hero (`.sx-hero`), stat counters, and four `.sx-card` links.
- **Announcement banner**: Material 9.5.x has NO `extra.announcement` support — override `{% block announce %}` in `overrides/main.html`, enabled via `theme.custom_dir: overrides`. (base.html renders the banner only when `self.announce()` is non-empty.)
- **404**: MkDocs renders a theme `404.html` template to `site/404.html`; `docs/404.md` only produces `404/index.html` (which is NOT what GitHub Pages serves). Correct mechanism = override `overrides/404.html` (`{% extends "main.html" %}` + `{% block content %}`; add `<style>` in `extrahead` to hide `.md-sidebar`). **GitHub Pages serves the root `gh-pages:404.html`** for any 404 in the project; mike does NOT refresh it, so after each deploy copy the built `1.5.0/404.html` over the root `404.html` (checkout `gh-pages`, `Copy-Item`, commit, push, back to `main`). The built page uses absolute `/sneppx-alg/<version>/…` asset paths, so it works at any URL.
- Deploy: `mike deploy 1.5.0 latest --update-aliases` then `git push origin gh-pages`. `python -m mike` fails (no `__main__`) — use the `mike` entrypoint. Verify by grepping the built `site/index.html` or `curl` the live latest URL with a cache-buster.
- **Feedback widget / RSS / version pill / hero art / Pyodide playground** are now built (social cards still blocked — `cairosvg` cannot load `libcairo-2.dll`).
  - `docs/javascripts/extra.js`: also page feedback widget (`localStorage` key `sx-feedback:<path>`) + version pill (fetches `../versions.json`).
  - `overrides/404.html`: custom 404 (theme-template override); `overrides/main.html`: `{% block announce %}` banner. Reuses `.sx-cards`/`.sx-card`, `.sx-404-art` (compass doodle), `.sx-404-url` pill; `extra.js` fills the requested URL + wires `.sx-404-search`.
  - `docs/index.md`: scheme-aware inline-SVG pipeline hero (`#sxarrow`, `.sx-art-*` classes; colors via `var(--md-primary-fg-color)` so dark/light both work).
  - **Playground**: `docs/playground.md` + `docs/javascripts/playground.js` + `docs/assets/sneppx_py.zip` (regen with Python `zipfile`, forward-slash arcnames, exclude `__pycache__`/`*.pyd`). Pyodide `v0.26.2` from jsDelivr loads NumPy then extracts the zip to `/sneppx` and `sys.path.insert`. Verified `import SneppX_ALG` works with all heavy deps absent (blocked via `sys.modules[m]=None`). `.gitignore` has `!docs/assets/sneppx_py.zip` to override `*.zip`.