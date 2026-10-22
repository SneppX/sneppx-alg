# Distributed Training Guide

SNEPPX-Alg provides distributed training primitives with NCCL + CPU fallback.

## DDP (Distributed Data Parallel)

```python
from SneppX_ALG.interface_bindings import distributed

# Initialize process group
pg = distributed.init_process_group(world_size=4, rank=0)
```

## ZeRO (Zero Redundancy Optimizer)

Three stages are supported:
- **ZeRO-1**: Partition optimizer states
- **ZeRO-2**: Partition optimizer states + gradients
- **ZeRO-3**: Partition optimizer states + gradients + parameters

## Tensor Parallelism

Split model layers across devices for large models.

## Pipeline Parallelism

Split model into stages across devices.

## Expert Parallelism (MoE)

Route tokens to experts on different devices.

## CPU Fallback

All collectives work on CPU for single-process testing and development.

## NCCL Binding

The NCCL binding (`net/distributed/nccl.c`) provides:
- `sneppx_pg_create` / `sneppx_pg_destroy`
- `sneppx_pg_all_reduce`
- `sneppx_nccl_all_gather`
- `sneppx_nccl_broadcast`
- `sneppx_nccl_send` / `sneppx_nccl_recv`

CPU fallback ensures correctness in single-process mode.