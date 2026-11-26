---
description: "Implementation status for SneppX-ALG distributed-training recipes; multi-rank Python collectives and FSDP are not currently supported."
---

# Cookbook — Distributed Training Status

> **These are not runnable multi-rank recipes.** The current Python distributed API fails closed when `WORLD_SIZE` is greater than 1. This page records the intended API surface without claiming that DDP, ZeRO, FSDP, or NCCL multi-rank execution works.

## Current supported boundary

- Single-process model training can run without initializing a distributed process group.
- `DistributedSampler` can partition an in-memory dataset's indices for local testing. It does not communicate or synchronize samples across workers.
- Multi-rank calls to `init_process_group()`, `all_reduce()`, `all_gather()`, `reduce_scatter()`, and `barrier()` raise an explicit error.
- The Python DDP/FSDP wrappers and ZeRO examples remain experimental and must not be used to claim distributed-training support.

## Before adding a distributed recipe

A recipe will be marked runnable only after it passes an end-to-end test with multiple processes, validates collective results against a reference implementation, verifies gradient and optimizer-state consistency, and tests failure handling. Until then, see the [distributed-training status](../distributed-guide.md) and [framework roadmap](../roadmap/tensorflow-pytorch-parity.md).
