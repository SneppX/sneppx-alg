# SNEPPX-Algo Cookbook

Copy-paste recipes for exploring **SneppX-ALG**. The snippets show documented
API usage, but availability and correctness vary by backend and operation.
Check each recipe's notes and run the relevant tests before relying on it.
Recipes are grouped by category and include intent, examples, and caveats.

> Set your path once per session:
>
> ```powershell
> $env:PYTHONPATH = "bindings/python"
> ```

## Categories

| Category | Recipes | When to reach for it |
|----------|--------:|----------------------|
| [Tensors](tensors.md) | 10 | Creating, reshaping, math, autograd |
| [Models & Layers](models_layers.md) | 8 | Building networks with `Module`/`nn` |
| [Training](training.md) | 6 | Loops, loss, checkpoints, `Trainer` |
| [Optimizers](optimizers.md) | 5 | SGD/AdamW/Lion/LAMB + schedulers |
| [Quantization](quantization.md) | 8 | INT8/INT4/FP8/AWQ/GPTQ, serving |
| [Distributed](distributed.md) | 2 | Local sampler utility and single-process training |
| [Serving & Inference](serving.md) | 4 | `sneppx-serve`, FastAPI, batching |
| [Data & Tokenization](data_tokenization.md) | 5 | `Dataset`, `DataLoader`, `Tokenizer` |
| [Security](../security.md) | 5 | Scan, PQ crypto, key vault, attestation |
| [Profiling](profiling.md) | 3 | `Profiler`, `timeit`, `MemoryTracker` |
| [Checkpointing](checkpointing.md) | 3 | Save/load, async, fault tolerance |
| [Conversion](conversion.md) | 3 | HF ↔ SNEPPX, safetensors, ONNX |
| [Generation](generation.md) | 4 | `generate`, sampling, beam search, streaming |

The listed counts describe the current recipe index; they are not feature-completeness or conformance claims.

## Legend

- :material-alert-decagram: **C backend required** — raises
  `RuntimeError: C backend not available` without `_SNEPPX_c`.
- :material-cpu-chip: **CPU-safe** — runs on pure NumPy, no build needed.
- :material-gpu: **GPU** — needs `SNEPPX_BUILD_CUDA=ON` and a CUDA device.

## Import quick-ref

```python
from SneppX_ALG import Tensor, TensorDataset, AdamW, Linear, Trainer, Transformer
# Sub-module (not re-exported via *):
from SneppX_ALG.interface_bindings.data_loader import DataLoader   # DataLoader
from SneppX_ALG.interface_bindings.tokenizer   import Tokenizer     # HuggingFace/byte-level
from SneppX_ALG.interface_bindings.generation  import generate, GenerationConfig, TextStreamer
from SneppX_ALG.interface_bindings.quantized_serve import quantize_model_weights, QuantizedModelConfig
```
