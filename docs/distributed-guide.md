---
description: "Current support boundaries for SneppX-ALG distributed training, process groups, collectives, and sampler utilities."
---

# Distributed Training

SneppX-ALG includes distributed-training API foundations and a deterministic
DistributedSampler. **Validated multi-process training is not available in the
current Python backend.** Process-group initialization, gradient collectives,
FSDP sharding, and multi-node orchestration must not be treated as working
features.

The Python API now fails explicitly when WORLD_SIZE is greater than one. This
is intentional: a job that appears to train while gradients are not synchronized
is worse than an explicit unsupported-operation error.

## What works today

- Single-process training with the ordinary tensor, module, optimizer, and
  trainer APIs.
- DistributedSampler index partitioning when used as a local utility. It
  divides indices deterministically; it does not communicate between ranks.
- Single-process calls to all_reduce, all_gather, and reduce_scatter return the
  input unchanged because there is only one participant.

## What is not supported

| Capability | Current status |
|---|---|
| Multi-rank process-group initialization | Unsupported; fails closed |
| NCCL-backed Python collectives | Not implemented in the Python adapter |
| DDP gradient synchronization | Unsupported for multiple ranks |
| FSDP parameter/gradient sharding | Experimental wrapper; multi-rank construction is rejected |
| ZeRO stages 1–3 | Not supported as end-to-end distributed training |
| Pipeline, tensor, and expert parallelism | Not validated end to end |
| Elastic multi-node training | Not supported |

The presence of C/CUDA/NCCL source files or API names does not establish a
working distributed runtime. A capability will be promoted to supported only
after communicator setup, data movement, synchronization, error handling, and
multi-process integration tests pass on the advertised hardware.

## Single-process example

```python
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim

model = nn.Linear(4, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

inputs = Tensor.randn((8, 4))
targets = Tensor.randn((8, 2))

predictions = model(inputs)
loss = nn.functional.mse_loss(predictions, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Sampler utility example

```python
from SneppX_ALG import DistributedSampler

dataset = list(range(10))
rank_zero_indices = list(
    DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
)
rank_one_indices = list(
    DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)
)
assert rank_zero_indices == [0, 2, 4, 6, 8]
assert rank_one_indices == [1, 3, 5, 7, 9]
```

These index partitions are useful for testing input sharding logic locally, but
they do not make a multi-rank training loop valid.

## Roadmap

The implementation order is: establish a real process-group lifecycle, add
validated CPU collectives, integrate NCCL communicator creation and unique-ID
exchange, then test DDP gradient synchronization. FSDP/ZeRO and pipeline,
tensor, and expert parallelism depend on those foundations.
