# Build Instructions

## 1. Prerequisites

**Required:**
- CMake 3.16+
- C11/C++20 compiler (MSVC 19.44+, GCC 11+, Clang 14+)
- Python 3.9+ (for bindings)

**Optional:**
- CUDA 12.0+ (GPU optimizer acceleration, `SNEPPX_BUILD_CUDA=ON`)
- Doxygen (API docs)

## 2. Quick Build (All Targets)

This project is Ninja-based. Use the CMake presets (see `CMakePresets.json`),
which select the Ninja generator and write to per-config binary directories.

**Windows (MSVC):**
```powershell
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build --preset release
```

**Linux/Mac (GCC/Clang):**
```bash
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build --preset release -j$(nproc)
```

The `release` preset configures `Release` + LTO + tests; `debug`, `asan`,
`cuda`, `python`, and `minimal` presets are also available.

## 3. Build Individual Targets

Build into the active binary directory (e.g. `build/release` after the
`release` preset), then reference targets:

```powershell
# Core kernel library
cmake --build build/release --config Release --target neural_core_kernel

# Architecture algorithms
cmake --build build/release --config Release --target neural_architecture_layer

# Security C library
cmake --build build/release --config Release --target neural_security_c

# Security C++ obfuscation
cmake --build build/release --config Release --target neural_security_cpp

# Python bindings
cmake --build build/release --config Release --target _SNEPPX_c

# CUDA kernels (requires CUDA SDK; use the `cuda` preset)
cmake --preset cuda
cmake --build --preset cuda --target neural_cuda_kernels
```

## 4. Build Options

| CMake Variable | Default | Description |
|----------------|---------|-------------|
| `SNEPPX_BUILD_CUDA` | OFF | Enable CUDA kernel compilation |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_TESTS` | ON | Build test executables |
| `SNEPPX_BUILD_EXAMPLES` | ON | Build example programs |
| `SNEPPX_BUILD_VULKAN` | OFF | Vulkan reference backend |
| `SNEPPX_BUILD_TPU` | OFF | TPU reference backend |
| `SNEPPX_BUILD_HTTP` | OFF | HTTP transport backend |
| `SNEPPX_BUILD_ZK` | OFF | Zero-knowledge backend |
| `SNEPPX_ENABLE_ASAN` | OFF | AddressSanitizer |
| `SNEPPX_ENABLE_UBSAN` | OFF | UndefinedBehaviorSanitizer |

## 5. Debug Build

Use the `debug` preset (Ninja, `Debug` build type, tests+tools on):

```powershell
cmake --preset debug
cmake --build --preset debug
```

Debug builds disable optimization and enable assertions. Use `-DSNEPPX_S1_SECURE_MEMORY=OFF` for faster debug iteration.

## 6. Python Package Build

```powershell
# Editable install
pip install -e .

# With CUDA support
pip install -e .[cuda]

# Build wheel
pip install build
python -m build --wheel
```

## 7. Running Tests

Run from the binary directory of the active preset (e.g. `build/release`):

```powershell
# C tests
cd build\release
ctest -C Release --output-on-failure

# Specific test
ctest -C Release -R test_kyber

# Sanitizer tests (Windows)
powershell -File scripts/run_sanitizers.ps1
```

For Python-only tests (no build required):
```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/ -v
```

## 8. Common Issues

| Issue | Fix |
|-------|-----|
| `cuda_runtime.h: No such file or directory` | Build without CUDA: `-DSNEPPX_BUILD_CUDA=OFF` |
| `pybind11_add_module: Unknown CMake command` | Set `pybind11_DIR` or ensure FetchContent can reach GitHub |
| `LNK2001: unresolved external symbol` | Build `neural_core_kernel` and `neural_security_c` before `_SNEPPX_c` |
| MSVC "no VLAs" | Use `calloc`/`free` instead of stack-allocated VLAs |
| `__int128 undeclared` on MSVC | Guard with `#ifndef NO_UINT128` (see `x25519.c` pattern) |
| `CreateProcess failed: The system cannot find the file specified.` (MASM, Windows+Ninja) | Use the stable Ninja 1.12+ (see `scripts/` or repo root `ninja.exe`); the Kitware *dev* Ninja in the venv can crash. The x64 MASM path (`ml64`) is resolved automatically — no manual `-DCMAKE_ASM_MASM_COMPILER` needed. |

## 9. Build Directories

The project standardizes on CMake presets (`CMakePresets.json`), each using the
Ninja generator and writing to a per-config binary directory. Prefer the presets:

- `cmake --preset release` → `build/release`
- `cmake --preset debug` → `build/debug`
- `cmake --preset asan` → `build/asan`
- `cmake --preset cuda` → `build/cuda`
- `cmake --preset python` → `build/python`
- `cmake --preset minimal` → `build/min`

Avoid configuring the plain `build/` directory with the presets; it is
reserved for the legacy Visual Studio generator and is not Ninja-clean.
See [DEVELOPMENT.md](DEVELOPMENT.md) for the full development workflow.
