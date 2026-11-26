---
description: "Current implementation status and limitations of SneppX-ALG distributed training, process groups, collectives, DDP, FSDP, and ZeRO."
---

# Distributed Training — Current Status

> **Status: experimental; multi-rank Python training is not currently supported.** The Python distributed API deliberately fails closed when `WORLD_SIZE` is greater than 1. Do not treat the DDP, FSDP, ZeRO, or NCCL interfaces as a working distributed-training stack.

## What the current Python API does

The Python package exposes distributed-related types and helpers, but it does not currently integrate a validated multi-rank collective backend. In particular:

- `init_process_group()` raises when `WORLD_SIZE > 1`; it does not report a successful multi-rank initialization.
- `all_reduce()`, `all_gather()`, `reduce_scatter()`, and `barrier()` raise for multi-rank contexts instead of returning local data or pretending ranks synchronized.
- `DistributedSampler` can partition a local dataset's indices. It does not transfer data or synchronize workers.
- DDP/FSDP and ZeRO code is experimental and is not a substitute for validated inter-process communication.

A single-process run (`WORLD_SIZE=1`) can use ordinary model training and local data utilities. That is not distributed training.

## Why the examples are limited

The repository contains native distributed code and API surfaces intended for future backend integration. The presence of NCCL symbols, process launch helpers, sharding classes, or architecture diagrams does not establish that a complete multi-rank training job is operational. Collective correctness, rank coordination, optimizer-state consistency, failure recovery, and multi-node interoperability require dedicated tests on real multi-process hardware.

## How to track progress

- [Framework parity roadmap](roadmap/tensorflow-pytorch-parity.md)
- [Distributed architecture notes](architecture/distributed_training.md)
- [Distributed cookbook status](cookbook/distributed.md)
- [Distributed tutorial status](tutorials/distributed_training.md)

Until the backend and end-to-end tests are complete, do not use these APIs for production training or publish performance claims based on multi-rank execution.
