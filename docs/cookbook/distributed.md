---
description: "Single-process training examples and current limitations of SneppX-ALG distributed utilities."
---

# Distributed Cookbook

The Python distributed API is currently limited to single-process execution
and local sampler utilities. Multi-rank process groups and collectives fail
closed; do not use the examples below as a recipe for distributed training.

## Partition dataset indices for local testing

```python
from SneppX_ALG import DistributedSampler

dataset = list(range(10))
rank_zero = list(DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False))
rank_one = list(DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False))

assert rank_zero == [0, 2, 4, 6, 8]
assert rank_one == [1, 3, 5, 7, 9]
```

This demonstrates deterministic index partitioning only. It does not perform
inter-process communication.

## Single-process training

```python
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn, optim

model = nn.Linear(4, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x = Tensor.randn((8, 4))
target = Tensor.randn((8, 2))
loss = nn.functional.mse_loss(model(x), target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Explicitly unsupported APIs

The following operations raise an error when configured for multiple ranks:
init_process_group, all_reduce, all_gather, reduce_scatter, and barrier.
launch(..., num_gpus=2) and multi-rank FSDP construction also raise. This
prevents silent local-only execution from being mistaken for synchronized
training.

See the [distributed support matrix](../distributed-guide.md) for the current
implementation boundary and roadmap.
