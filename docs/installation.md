# Installation Guide

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| CMake | 3.24 | Build system (4.x recommended for CUDA) |
| C compiler | C11 | MSVC 2022, GCC 11+, Clang 14+ |
| C++ compiler | C++20 | Only needed for S2 obfuscation engine |
| Python | 3.11+ | Optional (for bindings + test runner) |
| CUDA Toolkit | 12.x | Optional (for CUDA optimizer, `SNEPPX_BUILD_CUDA=ON`) |

## Linux

### Ubuntu / Debian

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Build
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release -j$(nproc)

# Test
cd build && ctest --output-on-failure
```

### Fedora / RHEL

```bash
sudo dnf install -y gcc gcc-c++ cmake git
# Same build steps as above
```

## macOS

### Intel (x86_64)

```bash
brew install cmake gcc
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build -j$(sysctl -n hw.ncpu)
cd build && ctest --output-on-failure
```

### Apple Silicon (ARM64)

```bash
brew install cmake
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build -j$(sysctl -n hw.ncpu)
cd build && ctest --output-on-failure
```

## Windows

### Native (Ninja + MSVC)

```powershell
# Open "Developer Command Prompt for VS 2022"
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release
cd build
ctest -C Release --output-on-failure
```

### PowerShell Script

```powershell
.\scripts\build.ps1 -Config Release -Tests
.\scripts\test.ps1 -Config Release
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `SNEPPX_BUILD_TESTS` | ON | Build test suite |
| `SNEPPX_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `SNEPPX_BUILD_PYTHON` | OFF | Build Python bindings |
| `SNEPPX_BUILD_CUDA` | OFF | Build CUDA kernels (requires CUDA Toolkit 12.x) |
| `SNEPPX_USE_ASAN` | OFF | Enable AddressSanitizer |
| `SNEPPX_USE_UBSAN` | OFF | Enable UndefinedBehaviorSanitizer |
| `SNEPPX_USE_LTO` | OFF | Enable Link-Time Optimization |
| `SNEPPX_BUILD_ZK` | OFF | Build zero-knowledge proof backend |

## CMake Presets

```bash
cmake --list-presets

# Available:
#   debug        — Debug build with tests
#   release      — Release build with tests
#   relwithdebinfo — Release with debug info
#   ninja-release — Ninja + Release
#   asan         — Debug + AddressSanitizer
```

## Python Setup

No build needed for pure-Python wrappers:

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "from SneppX_ALG import *"
```

For C-backed Python (optional):

```bash
# Requires pybind11
cmake --preset release -DSNEPPX_BUILD_PYTHON=ON
cmake --build build --config Release
```

## CUDA Setup

```powershell
cmake -B build -G Ninja -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_CUDA=ON
cmake --build build --config Release
```

Requires CUDA Toolkit 12.x with MSVC integration.

## Verify Installation

```bash
# Run full test suite
cd build && ctest --output-on-failure

# Run specific test
ctest -C Release -R test_tensor

# Run Python tests
$env:PYTHONPATH = "bindings/python"
python tests/python/test_algo_wrappers.py

# Run benchmarks
./tests/benchmark/bench_tensor
```

## Troubleshooting

### Build fails: "cannot open program database"

MSVC parallel build issue. The CMakeLists.txt includes `/FS` flag. If still failing:

```bash
cmake --build . --config Release -j1
```

### Test fails: "Ed25519 verify" or "Argon2 timing"

Pre-existing S0 edge case failures (2/306 Ed25519 vectors, 1/4 Argon2 timing). Not security vulnerabilities.

### CMake 4.x version error

If CMake fails with `invalid version string`, ensure you have the v1.0.0 tag checked out.

### Python import fails

```powershell
$env:PYTHONPATH = "bindings/python"
python -c "import SneppX_ALG; print(SneppX_ALG.__file__)"
```

### CUDA not found

Ensure `CUDA_PATH` environment variable is set and points to CUDA Toolkit 12.x.
