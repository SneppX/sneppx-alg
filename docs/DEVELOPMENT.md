# Development

## Prerequisites

- **CMake** 3.24+
- **C11 compiler** (MSVC 2022, GCC 11+, Clang 14+)
- **C++20** for S2 obfuscation (optional)
- **Python 3.11+** for bindings and test runner (optional)
- **Git** with GPG or Ed25519 signing configured

## Quick Start

```bash
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset debug
cmake --build build --config Debug -j$(nproc)
cd build && ctest --output-on-failure
```

## Branching Model

SNEPPX-Algo uses a **track-based Git Flow**. See `docs/BRANCHING_STRATEGY.md` for full details.

### Branch Types

| Type | Pattern | Base | Used for |
|------|---------|------|----------|
| Main | `main` | — | Production releases |
| Integration | `dev` | `main` | Feature integration |
| Feature | `feature/<track>-<name>` | `dev` | New features |
| Release | `release/v*.*.*` | `dev` | Release stabilization |
| Hotfix | `hotfix/<name>` | `main` | Urgent production fixes |
| Security | `security/<name>` | `main` | Coordinated security patches |
| Docs | `docs/<name>` | `dev` | Documentation changes |
| Experiment | `experiment/<name>` | `dev` | Research spikes |

### Track Prefixes

Features use track prefixes: `python`, `c-core`, `cuda`, `security`, `algo`, `infra`, `dist`

## Commenting Standard

All source files must follow the four-layer commenting standard in [COMMENTING.md](COMMENTING.md). PRs that add or modify source files must include:
- Layer 1 file header blocks (WHAT/CONCEPT/ROLE/REFERENCES)
- Layer 4 Doxygen `@brief/@param/@return` on all public `SNEPPX_*` functions
- Run `sneppx-format --docs` to verify before submitting

### Workflow

1. **Branch**: `git checkout -b feature/<track>-<name> dev`
2. **Develop**: write code, add tests, run locally
3. **Format**: `clang-format -i -style=file <files>`
4. **Test**: `ctest --output-on-failure`
5. **Commit**: `git commit -S -m "component: message"`
6. **Push**: `git push origin feature/<track>-<name>`
7. **PR**: Open PR to `dev` using template at `docs/PR_TEMPLATE.md`

## Project Layout

```
include/neural_core/     # Public headers (kernel, architecture, security)
kernel/                   # Core implementations (tensor, autodiff, train, optimizer, attention, distributed, quantization, cuda)
algorithms/               # Algorithm pipeline (hss, ser, arc, npe, fm)
tests/                    # Unit, integration, benchmark, security, python tests
examples/                 # Demo programs
bindings/python/          # Python wrappers (pure Python, no pybind11 needed)
scripts/                  # Build and development scripts
cmake/                    # CMake modules
docs/                     # Documentation
security/                 # S0-S9 security layer source
config/                   # Model zoo configs
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SNEPPX_BUILD_TESTS` | ON | Build test suite |
| `SNEPPX_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_CUDA` | OFF | Build CUDA kernels |
| `SNEPPX_USE_ASAN` | OFF | AddressSanitizer |
| `SNEPPX_USE_UBSAN` | OFF | UndefinedBehaviorSanitizer |
| `SNEPPX_USE_LTO` | OFF | Link-Time Optimization |

## Testing

- **C tests**: `ctest --output-on-failure`
- **Python tests**: `$env:PYTHONPATH = "bindings/python"; python tests/python/test_*.py`
- All new features must include tests
- Pre-existing failures: Argon2id (1 timing edge case), Ed25519 (2 verification edge cases)

## Build Targets

| Target | Description |
|--------|-------------|
| `neural_core_kernel` | Core tensor/memory/trainer library |
| `neural_architecture_layer` | Neural architecture algorithms |
| `neural_security_c` | C security library (S0-S1) |
| `neural_security_cpp` | C++ security library (S2-S3) |
| `neural_cuda_kernels` | CUDA kernels (conditional, `SNEPPX_BUILD_CUDA=ON`) |
| `neural_model_config` | Model config schema (C) |
| `neural_model_registry` | Model registry (C) |
| `neural_model_weights` | Weight collection and quantization (C) |
| `neural_model_card` | Model card metadata (C) |
| `neural_model_factory` | C++ RAII wrappers |

## Adding a New Algorithm Pipeline Component

1. Create `algorithms/<name>/core/<name>.c` and `include/neural_core/architecture/<name>.h`
2. Add public API with `SNEPPX_` prefix, `int` return codes, `SNEPPXTensor*` types
3. Write tests in `tests/unit/test_<name>.c`
4. Create Python wrapper in `bindings/python/SneppX_ALG/interface_bindings/algo_<name>.py`
5. Export from `interface_bindings/__init__.py`
6. Write Python tests in `tests/python/test_<name>.py`
7. Register in `CMakeLists.txt` (`.c` files picked up by `file(GLOB_RECURSE)`)

## Code Review

Patches are reviewed for:
- Correctness: does the code do what it claims?
- Style: does it follow STYLE_GUIDE.md?
- Safety: are all allocations checked? No buffer overflows?
- Tests: are new features adequately tested?

## SNEPPX Dev Tools

The 7 standalone SNEPPX developer tools are integrated into the workflow
(see `.sneppx-tools.json` and `scripts/dev-tools.{ps1,sh}`). Install:

```bash
pip install sneppx-toolkit[all]
```

Run the full quality gate before pushing a feature branch:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\dev-tools.ps1
# Linux/macOS
./scripts/dev-tools.sh
```

Individual checks:

```bash
sneppx-analyze --dirs kernel algorithms net security   # security scan
sneppx-format --lint kernel algorithms net             # style / standards lint
sneppx-deps --circular .                               # circular dependency check
sneppx-stats --save .sneppx/stats.json .               # code statistics
sneppx-test --build-dir build --exclude cuda           # enhanced test runner
sneppx-bench --build-dir build                         # benchmarks with regression tracking
```

The pre-commit hook (`scripts/install-hooks.sh`) also runs the analyzers when
installed.
