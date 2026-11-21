# Tutorial — Distributed Training

**Notebook:** [`distributed_training.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/distributed_training.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/distributed_training.ipynb))

## What you'll build

A single script that trains a model with **DDP-style gradient sync** and
**ZeRO-1** optimizer-state sharding, runnable as 1-process-per-GPU via
`SneppX_ALG.launch`. The notebook also covers `DistributedSampler` and the
FSDP (ZeRO-3) path.

## Setup

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np
from SneppX_ALG import (
    Transformer, DistributedWrapper, AdamW, CrossEntropyLoss,
    DistributedSampler, get_world_size, get_rank,
    init_process_group, destroy_process_group, FullyShardedDataParallel,
    FSDPConfig, ShardingStrategy, launch, TensorDataset, Tensor,
)
from SneppX_ALG.interface_bindings.data_loader import DataLoader
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. The training function (runs in each rank)

```python
def main():
    init_process_group(backend="nccl")       # reads WORLD_SIZE/RANK env
    rank = get_rank(); world = get_world_size()
    print(f"rank {rank}/{world}")

    model = Transformer(vocab_size=800, dim=256, num_heads=4, num_layers=4, ffn_dim=1024, max_seq_len=64)
    dp_model = DistributedWrapper(model, device="cuda" if world > 1 else "cpu")

    # synthetic sharded data
    X = Tensor.randn((256, 64)); y = Tensor(np.random.randint(0, 800, (256, 64)))
    ds = TensorDataset(X, y)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=32, sampler=sampler)

    opt = AdamW(dp_model.parameters(), lr=2e-4, weight_decay=0.01)
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for xb, yb in loader:
            logits = dp_model(xb)
            loss = CrossEntropyLoss()(logits.reshape((-1, 800)), yb.reshape((-1,)))
            opt.zero_grad(); loss.backward(); dp_model.sync_gradients(); opt.step()
        if rank == 0:
            print(f"epoch {epoch} loss={loss.item():.4f}")
    destroy_process_group()
```

## 2. Launch 2 GPUs (or 2 processes)

```python
if __name__ == "__main__":
    launch(main, num_nodes=1, num_gpus=2)
```

`launch` spawns `torch.distributed.run` under the hood when
`num_nodes * num_gpus > 1`; otherwise it calls `main()` directly — so the
notebook cell can be run as-is for a smoke test with `WORLD_SIZE=1`.

## 3. ZeRO-1 (shard optimizer state) with FSDP

For larger models, wrap in `FullyShardedDataParallel` (`FULL_SHARD` = ZeRO-3):

```python
base = Transformer(vocab_size=800, dim=512, num_heads=8, num_layers=12, ffn_dim=2048, max_seq_len=128)
fsdp_cfg = FSDPConfig(
    sharding=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(param="bf16", reduce="fp32"),
)
model = FullyShardedDataParallel(base, fsdp_cfg)
# model.parameters() are now sharded; grads all-reduce on backward (C backend)
```

## 4. ZeRO-2/3 directly (CPU/CUDA kernel)

```c
/* kernel/distributed/zero.c */
SNEPPX_ZeROOptimizer* zopt = SNEPPX_zero_optimizer_create(
    opt, world_size, rank, /*stage=*/3);
SNEPPX_zero_step(zopt, grads, n_grads, params, n_params);
```

## Key takeaways

- `DistributedWrapper.sync_gradients()` divides grads by `world_size` after
  the all-reduce — so use the *local* loss, not `loss / world_size`.
- `DistributedSampler.set_epoch()` must be called each epoch to re-shuffle
  across ranks.
- `launch` short-circuits for single-process runs — great for smoke tests.
- FSDP `FULL_SHARD` is the recommended way to fit 7B+ models on 4× GPUs.
- NCCL must be on `PATH`/`LD_LIBRARY_PATH`; without it the Python side
  falls back to no-op collectives (pass-throughs data).

## Next steps

- Combine with [RLHF](fine_tuning_rlhf.md) on LoRA adapters for PPO.
- Use [Profiling & Benchmarks](profiling_benchmarks.md) to spot
  communication bubbles.
