---
description: "Status note for the SneppX-ALG distributed-training tutorial; the current Python multi-rank collective backend is not yet implemented."
---

# Distributed Training — Status Note

> **This page is not a runnable distributed-training tutorial yet.**

The repository contains an exploratory distributed-training notebook and API surfaces for DDP, FSDP, ZeRO, and process launching. The current Python package does not provide a validated multi-rank collective backend:

- `init_process_group()` raises when `WORLD_SIZE > 1`.
- Collective operations and barriers fail explicitly rather than simulating communication.
- FSDP and ZeRO paths are experimental and are not validated end-to-end across processes or GPUs.
- A local `DistributedSampler` test does not demonstrate distributed execution.

Do not run the historical notebook as proof of working multi-rank training or use it for performance benchmarks. The notebook remains an exploratory artifact until a real backend, reproducible multi-process tests, and optimizer/gradient correctness checks are available.

For the actual support boundary, see the [distributed-training status page](../distributed-guide.md) and the [framework parity roadmap](../roadmap/tensorflow-pytorch-parity.md).
