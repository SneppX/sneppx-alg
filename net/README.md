# Networking Layer — Distributed Communication

The `net/` directory implements low-level networking primitives for distributed
training and inference: topology discovery, TCP socket communication, RDMA
transfers, and gRPC services.

## Topology (`net/topology/`)

Ring, tree, and graph topology management with BFS-based routing.
- `topology.h / topology.c` — Topology creation, neighbor queries, route computation
- Supports 3 topology types: ring, tree, graph
- BFS shortest-path routing for graph topologies

## Socket Communication (`net/socket/`)

Real TCP communication layer with platform abstraction.
- `socket_comm.h / socket_comm.c` — TCP send/recv with message framing
- WinSock2 on Windows, POSIX sockets on Unix
- Message length-prefix framing for reliable delivery

## RDMA (`net/rdma/`)

Remote Direct Memory Access for high-throughput low-latency transfers.
- `rdma_manager.h / rdma_manager.c` — Memory registration, QP lifecycle, read/write operations
- Platform abstraction layer (ibverbs on Linux, WinOF on Windows)
- Supports both RDMA Read and RDMA Write operations

## gRPC (`net/grpc/`)

gRPC-based control plane for distributed coordination.
- `grpc_service.h / grpc_service.c` — Server/stub lifecycle, barrier, all_gather, tensor transfer
- Abstracted over platform gRPC implementation
- Collective operations: barrier, all_gather, tensor broadcast

## Distributed (`net/distributed/`)

NCCL integration for GPU-to-GPU communication.
- `nccl.h / nccl.c` — Dynamic NCCL loading (win/linux/mac)
- All-reduce, all-gather, reduce-scatter, broadcast, send/recv
- CPU fallback when NCCL unavailable
- Process group management

## Build

The networking layer is part of `neural_core_kernel` (auto-discovered via GLOB_RECURSE).
No additional dependencies required for TCP/rdma/gRPC abstractions.
NCCL support requires the NCCL SDK.

## Python Bindings

Available via `SneppX_ALG`:
```python
from SneppX_ALG import Topology, SocketComm, RDMA
```
