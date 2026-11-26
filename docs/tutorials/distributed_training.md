---
description: "Distributed training status for SneppX-ALG, including the single-process sampler utility and unsupported multi-rank APIs."
---

# Distributed Training: Support Status

This page records the current state of distributed training so examples do not
overstate what the implementation can execute.

> **Multi-rank training is unsupported.**

The Python distributed module currently provides environment metadata and a
deterministic sampler utility. It does not yet initialize a real multi-rank
process group, synchronize gradients, or gather sharded parameters. APIs that
would otherwise make a multi-rank run appear successful now fail closed.

## Safe local check

```python
from SneppX_ALG import DistributedSampler

dataset = list(range(12))
sampler = DistributedSampler(
    dataset, num_replicas=3, rank=1, shuffle=False
)
assert list(sampler) == [1, 4, 7, 10]
```

This checks local index partitioning only. It is not a distributed training
test.

## Required work before a distributed release

1. Implement a process-group lifecycle with validated rank membership.
2. Implement and test CPU collectives before adding accelerator transport.
3. Implement NCCL communicator creation, unique-ID exchange, and error checks.
4. Prove DDP gradient synchronization with deterministic multi-process tests.
5. Complete FSDP/ZeRO parameter and optimizer-state sharding, including
   checkpoint and failure-recovery semantics.
6. Test pipeline, tensor, and expert parallel execution across actual processes.

Do not use WORLD_SIZE greater than one, FSDP, ZeRO, or the launch helper for
production training until these milestones are complete.
