# SneppX-ALG Quick Start

Everything you need to build, test, run, and extend **SNEPPX-Alg (ARIX_Algo)** in one place.
For deeper docs, see [`docs/index.md`](docs/index.md).

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| CMake | ≥ 3.16 | `cmake --version` |
| C compiler | MSVC ≥ 19.44 / GCC ≥ 11 / Clang ≥ 14 | C11 required |
| C++ compiler | C++20 | For obfuscation layer only |
| Python | ≥ 3.9 | For Python bindings + CLI tools |
| Git | ≥ 2.30 | For source + submodule |

No other dependencies. The C core is dependency-free; Python bindings use
`numpy` + `pyyaml` only.

---

## 2. Clone & Build

```powershell
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg

# Configure (Release) — uses the Ninja generator
cmake --preset release

# Build everything (library + examples + tests)
cmake --build --preset release
```

> Windows tip: prefer the stable repo-root `ninja.exe` (Ninja ≥ 1.12). The
> Kitware *dev* Ninja in the venv can crash — see `AGENTS.md`.

### Opt-in backends (OFF by default)

```powershell
cmake -B build -G Ninja -DSNEPPX_BUILD_VULKAN=ON   # Vulkan reference compute
cmake -B build -G Ninja -DSNEPPX_BUILD_TPU=ON      # TPU reference compute
cmake -B build -G Ninja -DSNEPPX_BUILD_HTTP=ON     # HTTP transport (BSD sockets)
cmake -B build -G Ninja -DSNEPPX_BUILD_ZK=ON       # Zero-knowledge proofs
cmake -B build -G Ninja -DSNEPPX_BUILD_METAL=ON    # Apple Metal (macOS)
cmake -B build -G Ninja -DSNEPPX_BUILD_ONEAPI=ON   # Intel oneAPI/SYCL
```

### Everything at once

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON `
  -DSNEPPX_BUILD_VULKAN=ON -DSNEPPX_BUILD_TPU=ON -DSNEPPX_BUILD_HTTP=ON -DSNEPPX_BUILD_ZK=ON
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

---

## 3. Python Setup

### 3a. Install the package

```powershell
cd bindings/python
pip install -e .
# or
pip install -e ".[serve]"        # + uvicorn for the inference server
pip install -e ".[hf]"           # + huggingface_hub for model downloads
```

### 3b. Verify

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "from SneppX_ALG import *; print('ok')"
```

### 3c. CLI tools (installed by setup.py)

| Command | Purpose |
|---------|---------|
| `sneppx-train` | Run a training job |
| `sneppx-serve` | Start the inference server |
| `sneppx-experiment` | Run experiments |
| `sneppx-eval` | Evaluate on MMLU / GSM8K / HumanEval |
| `sneppx-quantize` | Quantize a model (INT8/INT4/FP8/AWQ/GPTQ) |
| `sneppx-rlhf` | Run RLHF (DPO/GRPO) |

---

## 4. Run a Demo

```powershell
# C demos (built into build/release)
build\release\examples\hss_demo.exe
build\release\examples\ser_demo.exe
build\release\examples\arc_demo.exe
build\release\examples\npe_demo.exe
build\release\examples\fm_demo.exe

# Python — run a regression test (safe, no GPU/LLM)
python tests/python/test_tensor.py
```

> ⚠️ This machine has **no GPU / limited RAM**. Never run LLM-inference or
> CUDA tests (`test_llama_models.py`, `test_cuda.py`, `test_inference_server.py`).
> Always run the safe suite: `test_tensor`, `test_nn`, `test_optim`, `test_data`,
> `test_quantization`, `test_checkpoint`, `test_profiler`, `test_model_zoo`.

---

## 5. Start the Inference Server

```powershell
# Start (FastAPI, defaults: 0.0.0.0:8080)
sneppx-serve start --port 8080 --admin-key <your-admin-key>

# Check health
sneppx-serve health

# Create an API key (needs a running server, or use the admin endpoints)
curl -X POST http://localhost:8080/v1/admin/keys \
  -H "Authorization: Bearer <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"name":"test-key","tier":"free"}'

# Generate
curl -X POST http://localhost:8080/v1/generate \
  -H "Authorization: Bearer sk-sneppx-<your-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","prompt":"Hello","max_tokens":16}'

# Stop
sneppx-serve stop
```

Full REST API docs (24 endpoints): [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md).

---

## 6. Developer CLI Tools

The 7 standalone tools live under `C:\Users\PC\SneppX\` as `sneppx-{analyze,bench,test,format,deps,stats,serve}`
or as a meta-package:

```powershell
pip install sneppx-toolkit[all]
```

| Command | Purpose |
|---------|---------|
| `sneppx-analyze` | Security vulnerability scanner for C/C++/CUDA |
| `sneppx-bench` | Benchmark runner with regression tracking |
| `sneppx-test` | Enhanced test runner (tags, parallel) |
| `sneppx-format` | Linter/formatter enforcing project standards |
| `sneppx-deps` | Dependency / circular-dependency analyzer |
| `sneppx-stats` | Code statistics dashboard with trends |
| `sneppx-serve` | Inference server lifecycle + API keys |

Quick usage:

```powershell
sneppx-analyze net/http/                # scan a dir for vulnerabilities
sneppx-format --lint kernel/            # lint a dir
sneppx-deps --dot deps.dot .           # export dependency graph
sneppx-stats --json .                  # code stats as JSON
```

---

## 7. Project Layout

| Path | Purpose |
|------|---------|
| `kernel/` | Core tensor / autodiff / optimizer / trainer |
| `algorithms/` | HSS, SER, ARC, NPE, FM, Transformer, ViT, GCN, RNN, GAN, Diffusion, RL |
| `drivers/` | Accelerator backends (CUDA, ROCm, Vulkan, TPU, HTTP, ZK, Metal*, oneAPI*) |
| `security/` | S0–S9 security layers |
| `net/` | Distributed + gRPC coordination |
| `bindings/python/` | Python API (`SneppX_ALG`) |
| `tests/` | Unit, integration, benchmark, security, python tests |
| `examples/` | Demo programs |
| `docs/` | All documentation |
| `config/` | Model zoo configs |
| `releases/` | Release signing tooling |

---

## 8. Common Tasks

### Add a new C source file

CMake uses `file(GLOB_RECURSE)`, so a new `.c` under any source dir is picked up
automatically. No CMakeLists edit needed.

### Run one specific test

```powershell
ctest -C Release -R test_kyber --output-on-failure
```

### Run Python unit tests

```powershell
$env:PYTHONPATH = "bindings/python"
python tests/python/test_tensor.py
```

### Load a model from Hugging Face

```python
from SneppX_ALG.model_zoo import from_pretrained
model = from_pretrained("meta-llama/Llama-2-7b-hf")
```

---

## 9. Where to Go Next

- [`docs/index.md`](docs/index.md) — full documentation home
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) — 24 REST endpoints
- [`docs/build.md`](docs/build.md) — build options and troubleshooting
- [`docs/security_layers.md`](docs/security_layers.md) — S0–S9 deep dive
- [`docs/CONTRIBUTOR_TIERS.md`](docs/CONTRIBUTOR_TIERS.md) — contribution framework
