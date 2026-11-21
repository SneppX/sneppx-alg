# Cookbook — Distributed

## 1. Initialize a process group

**Intent:** Set up NCCL communication across ranks.

```python
from SneppX_ALG import init_process_group, destroy_process_group, get_world_size, get_rank

init_process_group(backend="nccl")      # reads WORLD_SIZE/RANK env vars
print(get_rank(), "/", get_world_size())
# ... train ...
destroy_process_group()
```

**Notes:** With `WORLD_SIZE=1`, `init_process_group` is a no-op. Needs
`libnccl` on `PATH`/`LD_LIBRARY_PATH`. CPU-safe as a pass-through.

## 2. Launch multi-GPU from a single script

**Intent:** One process per GPU, no `torchrun`.

```python
from SneppX_ALG import launch

def main():
    # your training function
    ...

if __name__ == "__main__":
    launch(main, num_nodes=1, num_gpus=2)
```

**Notes:** `launch` spawns `torch.distributed.run` under the hood when
`num_nodes * num_gpus > 1`; otherwise it calls `main()` directly.

## 3. Wrap a model for DDP

**Intent:** Gradient-sync data parallelism.

```python
from SneppX_ALG import Transformer, DistributedWrapper

model = Transformer(vocab_size=1000, dim=256, num_heads=4, num_layers=4, ffn_dim=1024, max_seq_len=128)
model = DistributedWrapper(model, device="cuda", sync_bn=False)
for x, y in loader:
    loss = model(x)
    loss.backward()
    model.sync_gradients()      # all-reduce grads / world_size
    opt.step()
```

**Notes:** `DistributedWrapper` keeps the inner `.module` accessible.
CPU-safe: `sync_gradients` is a no-op when `world_size <= 1`.

## 4. Shard data with DistributedSampler

**Intent:** Each rank sees a disjoint slice of the dataset.

```python
from SneppX_ALG import TensorDataset, DistributedSampler
from SneppX_ALG.interface_bindings.data_loader import DataLoader

sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
loader  = DataLoader(dataset, batch_size=32, sampler=sampler)
sampler.set_epoch(epoch)      # re-shuffle each epoch
```

**Notes:** `DataLoader` itself is in `interface_bindings.data_loader` (not
re-exported via `*`). CPU-safe.

## 5. ZeRO with FSDP (shard optimizer + grad + params)

**Intent:** Fit a 7B model on 4× GPUs by sharding everything.

```python
from SneppX_ALG import FullyShardedDataParallel, FSDPConfig, ShardingStrategy, MixedPrecision

cfg  = FSDPConfig(
    sharding=ShardingStrategy.FULL_SHARD,           # ZeRO-3 equivalent
    mixed_precision=MixedPrecision(param="bf16", reduce="fp32"),
)
model = FullyShardedDataParallel(base_transformer, cfg)
```

**Notes:** `ShardingStrategy` options: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OS`
(ZeRO-2), `NO_SHARD` (ZeRO-0). The C core mirror is `kernel/distributed/zero.c`.

## 6. Barrier + all-reduce

**Intent:** Synchronize or reduce across ranks.

```python
from SneppX_ALG import barrier, all_reduce, Tensor

barrier()                                   # all ranks rendezvous
acc = all_reduce(loss_tensor, op="sum")    # sum across ranks
```

## 7. Expert parallelism (MoE dispatch)

**Intent:** Route tokens to experts across devices.

```python
from SneppX_ALG import SERModel, SERConfig

cfg = SERConfig(); cfg.num_experts = 32; cfg.num_active = 2
# With EP, experts are sharded across the expert-parallel group; see
# algorithms/ser/core/ + net/distributed/nccl.c for the all-to-all dispatch.
moe = SERModel(cfg, seed=0, num_layers=1)
```

**Notes:** EP size is derived as `num_experts / tp_size`. The all-to-all is
handled by `SNEPPX_alltoall` in the NCCL layer. GPU + NCCL required.

## 8. Elastic training (node joins/leaves)

**Intent:** Elastic fault tolerance.

```python
from SneppX_ALG import ElasticTrainer

trainer = ElasticTrainer(max_restarts=3, restart_delay=5)
trainer.fit(train_fn, num_nodes=2, num_gpus=4)
# on a node failure: ranks re-shard, checkpoints roll forward
```
