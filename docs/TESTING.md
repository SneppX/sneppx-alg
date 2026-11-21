# Testing

## Running Tests

```powershell
# From the active build directory (e.g. build/release after `cmake --preset release`):
cd build\release

# All C tests
ctest -C Release --output-on-failure

# Tensor tests only
ctest -C Release -R test_tensor

# JIT pipeline tests
ctest -C Release -R test_npe_jit

# CUDA optimizer tests
ctest -C Release -R test_trainer_cuda

# ARC + SER + FM tests
ctest -C Release -R "test_arc|test_ser|test_fm"
```

## Python Tests

```powershell
# Set Python path
$env:PYTHONPATH = "bindings/python"

# Run all Python tests
python -m pytest tests/python/

# Algorithm wrapper tests (8 tests)
python tests/python/test_algo_wrappers.py

# Quantization tests (17 tests)
python tests/python/test_quantization.py

# Checkpoint tests (23 tests)
python tests/python/test_checkpoint.py

# Profiler tests (13 tests)
python tests/python/test_profiler.py

# Model zoo tests (49 tests)
python tests/python/test_model_zoo.py

# ONNX toolkit tests (25 tests)
python -m pytest onnx/tests/
```

## Test Categories

| Directory | Contents |
|-----------|----------|
| `tests/unit/` | Per-component unit tests |
| `tests/integration/` | Multi-component integration tests |
| `tests/benchmark/` | Performance benchmarks |
| `tests/security/` | S0-S9 crypto and security tests |
| `tests/python/` | Python wrapper tests |
| `tests/quantization/` | INT8/FP8 quantization tests |
| `tests/fuzz/` | Fuzz harnesses |

## Adding a Test

### C Test

1. Create `tests/unit/test_<component>.c`
2. Include `test_common.h` for ASSERT macros
3. Define `main()` with `run_test()` calls and `RUN_ALL_TESTS()`
4. CMake auto-discovers via `file(GLOB_RECURSE)`

```c
#include "test_common.h"

static void test_foo(void) {
    ASSERT_EQ(SNEPPX_foo(2, 2), 4, "2+2 == 4");
}

int main(void) {
    run_test("foo", test_foo);
    RUN_ALL_TESTS();
}
```

### Python Test

```python
import numpy as np
from SneppX_ALG.interface_bindings.algo_npe import NPECompiler

def test_npe_compiler():
    compiler = NPECompiler()
    prog = compiler.compile([])
    opt = compiler.jit_optimize(prog)
    assert len(opt) >= 0
    print("PASS: npe_compiler_basic")
```

## Known Issues

- **test_argon2**: 1 pre-existing timing edge case (3/4 pass)
- **test_ed25519**: 2 verification edge cases (304/306 pass)
- **test_ser_train**: 1 pre-existing flaky assertion (2/3 pass)
- **test_kyber**: 1 pre-existing shared secret mismatch

These are accepted as pre-existing and do not indicate regression.

## Model Zoo Tests

### C Tests (all must pass before release)

```powershell
# Via ctest (recommended):
cd build\release
ctest -C Release -R "test_model_(config|registry|weights|card|factory|integration)" --output-on-failure

# ...or run the binaries directly:
.\build\release\algorithms\model_zoo\test_model_config.exe
.\build\release\algorithms\model_zoo\test_model_registry.exe
.\build\release\algorithms\model_zoo\test_model_weights.exe
.\build\release\algorithms\model_zoo\test_model_card.exe
.\build\release\algorithms\model_zoo\test_model_factory.exe
.\build\release\algorithms\model_zoo\test_integration.exe
```

### Python Tests

```powershell
$env:PYTHONPATH = "bindings/python"
python bindings/python/SneppX_ALG/interface_bindings/tests/test_model_config.py
python bindings/python/SneppX_ALG/interface_bindings/tests/test_integration.py
```

Expected: 34 C/C++ tests + 22 Python tests = 56 total, all pass.
