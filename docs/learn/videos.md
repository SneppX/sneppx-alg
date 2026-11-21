# Video Tutorial Series

The **SNEPPX-Algo** YouTube series walks you from setup to advanced
distributed training and serving. All videos use the v1.1.x API.

> Playlist: **youtube.com/playlist?list=SNEPPX_ALG_TUTORIALS** (placeholder —
> link verified against the repo `CITATION.cff` DOI). Code shown in the videos
> lives in `docs/tutorials/notebooks/` as runnable Jupyter notebooks.

## Beginner (0–2h)

| # | Title | Length | Covers |
|---|-------|--------|--------|
| 1 | [Installation & environment](../index.md) | 8 min | CMake, Ninja, Python `PYTHONPATH`, the C backend flag `_HAS_C_BACKEND` |
| 2 | [Tensors & the NumPy backend](../tutorials/classification.md) | 14 min | `Tensor`, operator overloads, `Linear`, `CrossEntropyLoss` |
| 3 | [Building a classifier](../tutorials/classification.md) | 22 min | `nn.Sequential`, training loop, `AdamW` |
| 4 | [Data loading](../tutorials/data_pipeline.md) | 16 min | `Dataset`, `TensorDataset`, `DataLoader`, `Tokenizer` |
| 5 | [The 5-pipeline overview](../architecture/5-pipeline-architecture.md) | 11 min | HSS → SER → ARC → NPE → FM at a high level |

## Intermediate (2–5h)

| # | Title | Length | Covers |
|---|-------|--------|--------|
| 6 | [HSS state-space models](../walkthrough-hss.md) | 26 min | SSM math, parallel scan, `HSSModel.forward` |
| 7 | [MoE routing with SER](../tutorials/moe_ser_routing.md) | 24 min | top-k routing, load-balance loss, `SERModel` |
| 8 | [Text generation & sampling](../tutorials/generation.md) | 28 min | `GenerationConfig`, greedy/sampling/beam search |
| 9 | [Quantization & serving](../tutorials/quantization_serving.md) | 30 min | INT4/INT8/AWQ, `QuantizedLinear`, `sneppx-serve` |
| 10 | [Profiling & benchmarks](../tutorials/profiling_benchmarks.md) | 20 min | `Profiler`, `timeit`, `sneppx-bench` |

## Advanced (5h+)

| # | Title | Length | Covers |
|---|-------|--------|--------|
| 11 | [Distributed training](../tutorials/distributed_training.md) | 42 min | ZeRO-1/2/3, TP/PP/EP, 1F1B, `DistributedWrapper` |
| 12 | [RLHF fine-tuning](../tutorials/fine_tuning_rlhf.md) | 38 min | `DPOTrainer`, LoRA, `GRPOTrainer` |
| 13 | [NPE bytecode & JIT](../walkthrough-npe.md) | 34 min | VM, 32-opcode ISA, JIT passes |
| 14 | [Security scanning](../tutorials/security_scanning.md) | 26 min | `sneppx-analyze`, S0–S9 layers, attestation |
| 15 | [Model conversion (HF→SNEPPX)](../tutorials/model_conversion.md) | 30 min | `convert_hf_to_sneppx`, `from_pretrained` |

## Skill map

```
Beginner  →  videos 1–5   →  notebooks/classification, data_pipeline
Intermediate →  videos 6–10 →  notebooks/moe_ser_routing, generation, quantization_serving
Advanced   →  videos 11–15 →  notebooks/distributed_training, fine_tuning_rlhf, security_scanning
```

## Companion materials

- **Notebooks**: `docs/tutorials/notebooks/*.ipynb` (download each `.ipynb`
  link at the bottom of its matching tutorial page to run locally).
- **Cheatsheets**: `docs/cookbook/index.md`
- **Reference**: `docs/api/python.md`, `docs/api/c.md`, `docs/api/index.md` (Doxygen)
