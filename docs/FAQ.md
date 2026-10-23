# Frequently Asked Questions

## General & Environment

### "What Python version is required?"
SNEPPX-Alg requires **Python 3.9+** (tested on Python 3.9, 3.10, 3.11, and 3.12).

### "Does SNEPPX-Alg require a GPU or CUDA?"
**No.** All core tensor operations, autograd, optimizers, neural layers, and quantization algorithms run on pure CPU with the NumPy-backed engine. GPU acceleration (CUDA, ROCm, Vulkan) is strictly opt-in.

### "How do I run tests safely on a local CPU machine?"
Run tests **per-file** using `pytest` without cache provider:
```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_tensor.py -q -p no:cacheprovider
```
Never execute the entire `tests/` directory in a single command on memory-constrained machines, and skip GPU/LLM tests (`test_llama_models.py`, `test_cuda.py`).

### "What is the difference between `sx.functional` and `F`?"
`sx.functional` is reserved by the core autograd engine for forward/reverse mode derivatives (`jvp`, `vjp`). For neural network activations and loss functions, import `F` instead:
```python
from SneppX_ALG import F
# F.relu, F.mse_loss, F.cross_entropy, F.layer_norm
```

### "Can I migrate models from PyTorch?"
Yes. SNEPPX-Alg supports `state_dict` weight loading and provides functional and module APIs deliberately aligned with standard deep learning semantics. See [`docs/from_pytorch.md`](from_pytorch.md).

## Installation & Build

### "mkdocs build fails — `doxygen: command not found`"
Install Doxygen 1.9+ (see [`docs/api/index.md`](api/index.md)). On Windows: `vcpkg install doxygen`; on Linux: `sudo apt-get install doxygen graphviz`.

### "CMake cannot find a C/C++ compiler on Windows"
Install **Visual Studio 2022** with the *Desktop development with C++* workload (includes MSVC 19.44 and MASM), or run from Developer PowerShell so `cl.exe`/`ml64` are on `PATH`:

```powershell
cmake -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### "ninja.exe crashes with heap corruption"
The Kitware dev-build Ninja in some environments can be unstable. Use the repo-root `ninja.exe` (Ninja ≥ 1.12), or fall back to the Visual Studio generator: `cmake -B build -G "Visual Studio 17 2022"`.

## Python API

### "ImportError: cannot import name 'X' from 'SneppX_ALG'"
Top-level `from SneppX_ALG import *` re-exports primary symbols (`Tensor`, `F`, `nn`, `optim`). Other tools live under `interface_bindings` and require setting `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "bindings/python"
from SneppX_ALG import Tensor, F, Linear, AdamW
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer
from SneppX_ALG.interface_bindings.data_loader import DataLoader
from SneppX_ALG.interface_bindings.generation import GenerationConfig, generate
```

### "RuntimeError: C backend not available"
This indicates compiled extension binaries (`_SNEPPX_c*.pyd` or `.so`) were not linked. The pure-Python/NumPy fallback handles tensors and basic modules seamlessly without needing C compilation:
```python
import SneppX_ALG as s
print(s._HAS_C_BACKEND)   # False = Pure-Python mode; True = Native C acceleration
```

### "CUDA not available"
If you do not have an NVIDIA GPU, keep `SNEPPX_BUILD_CUDA=OFF` (the default). The engine operates natively in CPU mode.

## Quantization

### "quantize OOM / runs forever on a 7B model"
- Use **per-channel** or **per-group** INT8 (`QuantGranularity.PER_CHANNEL`, group size 128).
- For weights-only quantization (W8A16), prefer `QuantizedLinear.from_float` over whole-tensor paths.
- On CPU, wrap the loop in `quantize_model_weights(...)` which skips `lm_head`/`embed_tokens` by default.

### "INT4 weights decode incorrectly"
Ensure the quantizer and dequantizer specify the exact `n` (total element count): `dequantize_int4_sym(qw, scale, n)` requires the original element count rather than packed byte size.

## Distributed Training

### "Training hangs after init_process_group"
On CPU machines without NCCL, verify that every rank specifies matching `MASTER_ADDR` and `MASTER_PORT`:
```powershell
$env:MASTER_ADDR="127.0.0.1"; $env:MASTER_PORT="29500"
$env:WORLD_SIZE=2; $env:RANK=0; $env:LOCAL_RANK=0
```

## Security

### "How do I report a security vulnerability?"
See [`SECURITY.md`](../SECURITY.md). Do **not** open public issues for security vulnerabilities — report them privately via GitHub Security Advisories or email `algoSNEPPX@gmail.com`.

## Still stuck?

- Read [Getting Started](getting-started.md) for quick setup.
- Read [Installation Guide](installation.md) for deep build notes.
- Open an issue at the [SneppX Issue Tracker](https://github.com/SneppX/sneppx-alg/issues).
