# Frequently Asked Questions

## Installation & build

### "mkdx build fails — `doxygen: command not found`"
Install Doxygen 1.9+ (see [`docs/api/index.md`](api/index.md)). On Windows,
`vcpkg install doxygen`; on Linux, `sudo apt-get install doxygen graphviz`.

### "CMake cannot find a C/C++ compiler on Windows"
Install **Visual Studio 2022** with the *Desktop development with C++*
workload (includes MSVC 19.44 and MASM), or run from the *Developer PowerShell*
so `cl.exe`/`ml64` are on `PATH`. Then:

```powershell
cmake -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### "ninja.exe crashes with heap corruption"
The Kitware dev-build Ninja in the venv is unstable. Use the repo-root
`ninja.exe` (Ninja ≥ 1.12), or fall back to the VS generator:
`cmake -B build -G "Visual Studio 17 2022"`.

## Python

### "ImportError: cannot import name 'X' from 'SneppX_ALG'"
The top-level `from SneppX_ALG import *` only re-exports names in the package
`__all__`. Some names live in sub-modules and must be imported by path, **and**
the `interface_bindings` package must be on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "bindings/python"
# ✅ works (top-level):
from SneppX_ALG import Tensor, Linear, AdamW, Transformer
# ✅ works (sub-module — names not in __all__):
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer
from SneppX_ALG.interface_bindings.data_loader import DataLoader
from SneppX_ALG.interface_bindings.generation import GenerationConfig, generate
```

> The shorthand `from SneppX_ALG.model_zoo import from_pretrained` shown in
> some older docs does **not** resolve — `model_zoo` is under
> `interface_bindings`. Use `from SneppX_ALG import from_pretrained` instead.

### "RuntimeError: C backend not available"
This means `_SNEPPX_c`/`_arix_c` did not load (the compiled extension is
absent). Pure-NumPy fallbacks exist for tensors and some layers, but
**algorithm stages require the C backend**. Build and install it:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# then ensure bindings/python/SneppX_ALG can import the .pyd/.so
$env:PYTHONPATH = "bindings/python"
```

Check the flag at runtime:
```python
import SneppX_ALG as s
print(s._HAS_C_BACKEND)   # True once the C bridge is linked
```

### "CUDA not available"
If you have no CUDA device, build with `SNEPPX_BUILD_CUDA=OFF` (the
default) — the stack runs on CPU. CUDA kernels are opt-in:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=ON
```

When CUDA is absent, `cuda_is_available()` returns `False` and all
`cuda_device` calls are no-ops; tensors stay on CPU.

### "pip install sneppx-alg fails to compile the extension"
The C extension needs a compiler. If you only need the pure-Python
fallbacks, install without building:

```bash
pip install --no-build-isolation sneppx-alg
# or, if wheels are available for your platform:
pip install sneppx-alg
```

## Quantization

### "quantize OOM / runs forever on a 7B model"
- Use **per-channel** or **per-group** INT8 (`QuantGranularity.PER_CHANNEL`,
  group size 128) — cheaper than per-token.
- For weights-only quantization (W8A16), prefer `QuantizedLinear.from_float`
  over whole-tensor path.
- On CPU, wrap the loop in `quantize_model_weights(...)` which skips
  `lm_head`/`embed_tokens` by default (`QuantizedModelConfig.skip_layers`).

```python
from SneppX_ALG import QuantizedLinear, QuantMode
ql = QuantizedLinear.from_float(my_linear, mode=QuantMode.INT8_SYM)
```

### "INT4 weights decode to garbage"
Ensure the quantizer and dequantizer agree on `n` (total element count):
`dequantize_int4_sym(qw, scale, n)` requires the *original* element count, not
the packed byte count.

## Distributed training

### "Training hangs after init_process_group"
On CPU-only machines without NCCL, the most common cause is a **master-port /
world-size mismatch** between ranks. Verify every rank sets the same
`MASTER_ADDR`/`MASTER_PORT` and that `WORLD_SIZE = dp×tp×pp`:

```powershell
$env:MASTER_ADDR="127.0.0.1"; $env:MASTER_PORT="29500"
$env:WORLD_SIZE=2; $env:RANK=$rank; $env:LOCAL_RANK=$rank
```

For single-node CPU-only dev, set `WORLD_SIZE=1` and call your training
function directly (`launch(...)` short-circuits when `num_nodes*num_gpus<=1`).

### "all_reduce is a no-op"
The NCCL layer loads `libnccl` dynamically. If it is not installed, the Python
`_NCCLBackend` falls back to a pass-through (returns data unchanged). Install
NCCL and point `LD_LIBRARY_PATH`/`PATH` at it, or run single-process.

## Generation & serving

### "generate() returns logits of shape (1,) not (batch, vocab)"
`Transformer.forward` returns **vocab logits** directly, but `generation.generate`
expects a model whose `forward(input_ids=..., past_key_values=..., use_cache=True)`
returns `{"logits", "past_key_values"}`. For a quick LM head wrapper, see
[tutorials/generation.md](tutorials/generation.md).

### "FastAPI server won't start: `pip install sneppx-alg[serve]`"
The inference server (`sneppx-serve`) requires optional deps:

```bash
pip install "sneppx-alg[serve]"
# or manually:
pip install fastapi uvicorn pydantic
```

## Checkpointing

### "checkpoint write fails with PermissionError"
Async checkpointing spawns a background I/O thread. On Windows ensure the
target path is not locked by another process and run with write access. The
checkpoint format is little-endian binary — do not hand-edit `.sneppx` files.

## Security

### "sneppx-analyze says my weights are unverified"
Run an **attestation** (`sneppx-analyze verify` checks the Ed25519
manifest signature). Unsigned/unsigned-mismatch checkpoints are quarantined by
S7 until re-signed with the key-vault signing key.

### "How do I report a vulnerability?"
See [`SECURITY.md`](https://github.com/SneppX/sneppx-alg/blob/main/../SECURITY.md). Do **not** open a public issue for
security bugs — email `algoSNEPPX@gmail.com`.

## Profiling

### "Timer shows 0.000s for my function"
The `@timeit` decorator only records when the global profiler is enabled.
Create and activate one:

```python
from SneppX_ALG import get_profiler
prof = get_profiler(); prof.enabled = True
# ...run work...
prof.print_summary()
```

## Still stuck?

- Browse the [cookbook](cookbook/index.md) for copy-paste recipes.
- Read [installation](installation.md) for platform notes.
- Open a non-security issue at the
  [tracker](https://github.com/SneppX/sneppx-alg/issues).
