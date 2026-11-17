---
description: "Runnable SneppX-ALG tutorials covering tensor operations, model building, training, and documented backend requirements."
---

# Tutorials

Interactive, runnable tutorials for SNEPPX-Algo. Each tutorial has a
**companion Jupyter notebook** under `docs/tutorials/notebooks/` that you can
download and run:

```powershell
$env:PYTHONPATH = "bindings/python"
pip install jupyter
jupyter notebook docs/tutorials/notebooks/classification.ipynb
```

> The notebooks use the **pure-Python / NumPy fallback** where possible so
> they run without a compiled C backend. Steps that require the C backend
> (`backward()`, `Trainer.fit`, real model forward on large weights) are
> explicitly marked `:material-alert-decagram: **C backend required**` and
> guarded with a runtime check.

## Tutorial map

| Tutorial | Notebook | Skill | What you build |
|----------|----------|-------|----------------|
| [Classification](classification.md) | `classification.ipynb` | Beginner | MLP on MNIST-style data |
| [Text Generation](generation.md) | `generation.ipynb` | Beginner–Int | Greedy / sampling / beam search |
| [RLHF Fine-Tuning](fine_tuning_rlhf.md) | `fine_tuning_rlhf.ipynb` | Advanced | LoRA + DPOTrainer |
| [Quantization + Serving](quantization_serving.md) | `quantization_serving.ipynb` | Intermediate | INT4/AWQ quant + `sneppx-serve` |
| [Distributed Training](distributed_training.md) | `distributed_training.ipynb` | Advanced | ZeRO-1 + DDP |
| [MoE SER Routing](moe_ser_routing.md) | `moe_ser_routing.ipynb` | Intermediate | 8-expert top-2 routing |
| [Security Scanning](security_scanning.md) | `security_scanning.ipynb` | Intermediate | `sneppx-analyze` + S0 crypto |
| [Profiling & Benchmarks](profiling_benchmarks.md) | `profiling_benchmarks.ipynb` | Intermediate | `Profiler` + `sneppx-bench` |
| [Data Pipeline](data_pipeline.md) | `data_pipeline.ipynb` | Beginner | Tokenizer + DataLoader + streaming |
| [Model Conversion](model_conversion.md) | `model_conversion.ipynb` | Intermediate | HF → SNEPPX checkpoints |

## Prerequisites

```powershell
# Build (for real training/gradient steps)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
$env:PYTHONPATH = "bindings/python"

# Verify
python -c "import SneppX_ALG as s; print('C backend:', s._HAS_C_BACKEND)"
```
