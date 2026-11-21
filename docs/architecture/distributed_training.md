# Distributed Training

SNEPPX-Algo ships a **full distributed-training stack** in
`kernel/distributed/`: data parallelism (ZeRO-1/2/3), pipeline parallelism
(1F1B), tensor parallelism (row/column split), expert parallelism (all-to-all
dispatch), gradient compression, and an NCCL backend layer.

```
  DP  ──┬── replica sharding (ZeRO optimizer-state/gradient/param partitioning)
  TP  ──┼── row/column linear split + partial-GEMM all-reduce
  PP  ──┼── 1F1B pipeline (micro-batch bubbles overlap compute)
  EP  ──┼── expert all-to-all (MoE / SER routing)
  + gradient topo-aware all-reduce, Top-K compression w/ error feedback
      async fault-tolerant checkpointing (heartbeat + elastic)
```

## Topology config

The C core exposes one unified struct, `SNEPPXDistributedConfig`
(`distributed.h`):

```c
SNEPPXDistributedConfig cfg = SNEPPX_distributed_config_default();
cfg.dp_size    = 4;      /* data-parallel world        */
cfg.tp_size    = 2;      /* tensor-parallel world      */
cfg.pp_size    = 2;      /* pipeline-parallel world    */
cfg.ep_size    = 2;      /* expert-parallel world      */
cfg.zero_stage = 2;      /* 0=none, 1=opt-state, 2+=grad+params */
cfg.nccl_dtype = SNEPPX_DTYPE_BF16;   /* NCCL reduce type             */
cfg.pipeline_microbatches = 4;        /* 1F1F micro-batches          */
cfg.expert_capacity = 32;
cfg.master_addr = "127.0.0.1";
cfg.master_port = 29500;
int world = cfg.dp_size * cfg.tp_size * cfg.pp_size; /* = 16 ranks */
```

**Rules of the mesh:**

- `dp × tp × pp = world_size`
- `ep` is derived from `num_experts / tp_size` (experts shard across the
  tensor-parallel group).

## ZeRO stages (S0/S1/S2 of *optimization state*)

| Stage | Optimizer state | Gradient | Parameters |
|------:|-----------------|----------|------------|
| 0 | replicated | replicated | replicated |
| 1 | **partitioned** (per-rank shard) | replicated | replicated |
| 2 | partitioned | **partitioned** | replicated |
| 3 | partitioned | partitioned | **partitioned** (param sharded) |

The fused CUDA kernels (`kernel/cuda/optim_cuda.cu`) implement ZeRO-1
partitioned AdamW. CPU fallback uses `kernel/distributed/zero.c`
(`SNEPPX_ZeROOptimizer`).

## Pipeline parallelism (1F1B)

`kernel/distributed/pipeline.c` implements the **1F1B (1-forward-1-backward)**
schedule: each micro-batch stage runs forward then immediately starts backward
on the *previous* micro-batch, keeping the pipeline saturated. A non-blocking
inter-stage `send`/`recv` (`SNEPPX_mpi_send_nb`/`recv_nb`) overlaps
communication with compute.

## Tensor parallelism

`kernel/distributed/tensor_parallel.c` provides **row** and **column** linear
layers that split weights across the TP group, run a local GEMM, then
all-reduce the partial result (`SNEPPX_tp_row_linear`, `SNEPPX_tp_col_linear`).
Head-count is divided evenly across TP ranks.

## Expert parallelism (all-to-all)

SER's MoE path (`algorithms/ser/core/`) runs a **Top-K dispatcher** that
exchanges tokens between ranks via the all-to-all collective
(`SNEPPX_alltoall`), ensuring each expert receives only the tokens routed to it.
`gradient_comm.c` adds Top-K gradient compression with error feedback on top.

## NCCL backend layer

`net/distributed/nccl.h/.c` provides dynamically-loaded NCCL wrappers:

```c
SNEPPXNCCLComm* comm = SNEPPX_nccl_create(rank, world_size, "nccl-pfn");
SNEPPX_nccl_all_reduce(comm, buf, count, SNEPPX_DTYPE_BF16, "sum");
SNEPPX_nccl_all_gather(comm, buf, count, SNEPPX_DTYPE_FP32);
SNEPPX_nccl_reduce_scatter(comm, recvbuf, sendbuf, count, SNEPPX_DTYPE_INT8, "sum");
SNEPPX_nccl_destroy(comm);
```

A pure-CPU fallback path is taken when NCCL is unavailable (single-node dev).

## Async checkpointing & fault tolerance

`kernel/distributed/checkpoint.c` provides double-buffered async save
(device→host I/O overlap), heartbeat liveness (`heartbeat.h`, UDP), and an
**elastic** trainer (`elastic.h`) that rebalances the topology when ranks join
or leave (`elastic.c`).

## Python API

The Python distributed layer mirrors the C API (`distributed.py` +
`distributed_wrapper.py`):

```python
from SneppX_ALG import (
    init_process_group, destroy_process_group, get_world_size, get_rank,
    barrier, all_reduce, DistributedSampler, DistributedDataParallel,
    DistributedWrapper, launch,
)

# Single-process multi-GPU launch (delegates to torchrun under the hood)
launch(train_fn=lambda: main(), num_nodes=1, num_gpus=2)

# Inside each rank:
init_process_group(backend="nccl")
dist = DistributedContext()                 # reads WORLD_SIZE / RANK / etc.
model = MyTransformer()
model = DistributedWrapper(model, device="cuda", sync_bn=False)

# Training loop with gradient sync
for x, y in DistributedSampler(dataset)(train_loader):
    loss = model(x); loss.backward()
    model.sync_gradients()                 # all-reduce grads / world_size
    optimizer.step()
barrier()
```

## ZeRO with FSDP

For very large models, `fsdp.py` exposes `FullyShardedDataParallel` with
`ShardingStrategy` (`FULL_SHARD`, `SHARD_GRAD_OS`, `NO_SHARD`) and `MixedPrecision`
bf16/fp16 controls — the Python equivalent of ZeRO-3.

```python
from SneppX_ALG import FullyShardedDataParallel, FSDPConfig, ShardingStrategy

cfg = FSDPConfig(sharding=ShardingStrategy.FULL_SHARD, mixed_precision=True)
model = FullyShardedDataParallel(base_model, cfg)
```

## Practical launch (multi-node)

```powershell
# rank 0
$env:MASTER_ADDR="10.0.0.1"; $env:MASTER_PORT="29500"
$env:WORLD_SIZE=8; $env:RANK=0; $env:LOCAL_RANK=0
sneppx-train --config configs/exp.yaml --zero-stage 2 --tp 2 --pp 2

# rank 1 (on the same or another node), RANK=1, LOCAL_RANK=1...
```

> **No GPU?** The stack still runs in eager CPU mode with `world_size=1`; set
> `SNEPPX_BUILD_CUDA=OFF` (the default) and the NCCL layer falls back to a
> no-op collective.

## Diagnostics

```powershell
cd build && ctest -C Release -R distributed --output-on-failure
```

| Test | Component | What it checks |
|------|-----------|----------------|
| `test_distributed` | `distributed.c` | DP/TP/PP/EP wiring, ZeRO partitioning |
| `test_nccl` | `net/distributed/nccl.c` | NCCL load + all-reduce correctness |
| `test_zero` | `zero.c` | ZeRO-1/2/3 optimizer state sharding |
| `test_pipeline` | `pipeline.c` | 1F1B bubble scheduling |
| `test_fsdp` | `fsdp.py` | FSDP shard/unshard round-trip |

## See also

- `docs/guide/model_zoo.md` for ZeRO-aware config presets
- `docs/architecture/security_layers_s0s9.md` (S2 obfuscation protects the
  communication layer's constants)
- `docs/tutorials/distributed_training.md` for a runnable walkthrough
