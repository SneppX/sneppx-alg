"""NCCL Python bindings — reference / single-process fallback.

.. warning::
   This module does **not** perform real multi-rank communication. The
   collectives below are identity or local CPU fallbacks: ``all_reduce``
   returns its input unchanged for ``world_size > 1``, ``all_gather`` replicates
   locally, and ``send``/``recv`` are no-ops. It is usable only for API-shape
   tests and single-process execution. For real distributed training use the
   NCCL/MPI-backed :mod:`distributed` module (which fails closed for multi-rank
   collectives when no real backend is loaded) or a supported multi-node stack.
"""

from typing import List, Optional, Tuple, Union
import ctypes
import os
import numpy as np
from .tensor import Tensor
from .nn import Module

# ============================================================================
# NCCL Constants
# ============================================================================


class NCCLRedOp:
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4


class NCCLDataType:
    FLOAT = 0
    HALF = 1
    INT = 2
    INT64 = 3
    FLOAT16 = 4
    BFLOAT16 = 5


# ============================================================================
# C Library Loading
# ============================================================================


class _NCCLBackend:
    def __init__(self):
        self._lib = None
        self._loaded = False
        self._load()

    def _load(self):
        if self._loaded:
            return
        paths = ["libnccl.so", "libnccl.so.2", "nccl.dll", "libnccl.dylib"]
        for p in paths:
            try:
                self._lib = ctypes.CDLL(p)
                self._loaded = True
                break
            except:
                continue
        if not self._loaded:
            print("[SNEPPX NCCL] Warning: NCCL library not found, using CPU fallback")

    @property
    def is_loaded(self):
        return self._loaded


_nccl_backend = _NCCLBackend()


# ============================================================================
# NCCL Communicator
# ============================================================================


class _NCCLComm:
    def __init__(self, world_size: int, rank: int, device_id: int):
        self.world_size = world_size
        self.rank = rank
        self.device_id = device_id
        self._nccl_comm = None
        self._use_nccl = _nccl_backend.is_loaded

    def all_reduce(self, data: np.ndarray, op: int = NCCLRedOp.SUM) -> np.ndarray:
        """Return ``data`` unchanged (identity fallback, no cross-rank reduce)."""
        if self._use_nccl and self._lib and self._lib.ncclAllReduce:
            # Real NCCL path is not wired up in this reference fallback.
            pass
        return data

    def all_gather(self, data: np.ndarray) -> List[np.ndarray]:
        """Return local copies of ``data`` (no cross-rank gather)."""
        if self._use_nccl:
            pass
        return [data.copy() for _ in range(self.world_size)]

    def broadcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Return a local copy of ``data`` (no cross-rank broadcast)."""
        if self._use_nccl:
            pass
        return data.copy()

    def send(self, data: np.ndarray, peer: int):
        """No-op: no data is sent to ``peer`` in the reference fallback."""

    def recv(self, data: np.ndarray, peer: int):
        """No-op: no data is received from ``peer`` in the reference fallback."""


# ============================================================================
# Process Group (High-level)
# ============================================================================


class ProcessGroup:
    """High-level process group for distributed communication."""

    def __init__(self, world_size: int, rank: int, device_id: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.device_id = device_id
        self._comm = _NCCLComm(world_size, rank, device_id)
        self._initialized = False

    def initialize(self):
        """Initialize process group."""
        if self._initialized:
            return
        # In real implementation, would call ncclCommInitRank
        self._initialized = True
        print(
            f"[Distributed] Rank {self.rank}/{self.world_size} initialized on device {self.device_id}"
        )

    def destroy(self):
        self._initialized = False

    @property
    def comm(self):
        return self._comm

    def all_reduce(self, tensor: Tensor, op: int = NCCLRedOp.SUM) -> Tensor:
        """Return ``tensor`` unchanged; no real multi-rank reduction occurs.

        For ``world_size > 1`` this is an identity fallback, not a reduction.
        Use :mod:`distributed` for a backend that fails closed rather than
        silently returning an un-reduced value.
        """
        if self.world_size <= 1:
            return tensor
        return tensor

    def broadcast(self, tensor: Tensor, root: int = 0) -> Tensor:
        """Return ``tensor`` unchanged; no real multi-rank broadcast occurs."""
        if self.world_size <= 1:
            return tensor
        return tensor

    def barrier(self):
        """No-op barrier: no cross-rank synchronisation is performed."""
        pass


# Global process group
_global_pg: Optional[ProcessGroup] = None


def init_process_group(world_size: int, rank: int, device_id: int = 0) -> ProcessGroup:
    """Initialize global process group."""
    global _global_pg
    _global_pg = ProcessGroup(world_size, rank, device_id)
    _global_pg.initialize()
    return _global_pg


def destroy_process_group():
    global _global_pg
    if _global_pg:
        _global_pg.destroy()
        _global_pg = None


def get_world_size() -> int:
    return _global_pg.world_size if _global_pg else 1


def get_rank() -> int:
    return _global_pg.rank if _global_pg else 0


def barrier():
    if _global_pg:
        _global_pg.barrier()


def all_reduce(tensor: Tensor, op: int = NCCLRedOp.SUM, comm=None) -> Tensor:
    if comm:
        return comm.all_reduce(tensor, op)
    if _global_pg:
        return _global_pg.all_reduce(tensor, op)
    return tensor


def broadcast(tensor: Tensor, root: int = 0, comm=None) -> Tensor:
    if comm:
        return comm.broadcast(tensor, root)
    if _global_pg:
        return _global_pg.broadcast(tensor, root)
    return tensor


# ============================================================================
# Distributed Sampler
# ============================================================================


class DistributedSampler:
    """Deterministic, evenly sized dataset shards for distributed ranks.

    By default, the sampler pads a short final shard by repeating initial
    indices so every rank receives the same number of samples. Set
    ``drop_last=True`` to discard the incomplete tail instead.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.num_replicas = get_world_size() if num_replicas is None else int(num_replicas)
        self.rank = get_rank() if rank is None else int(rank)
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        total = len(dataset)
        if self.drop_last:
            self.num_samples = total // self.num_replicas
        else:
            self.num_samples = (total + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
        if self.drop_last:
            indices = indices[: self.total_size]
        elif self.total_size > len(indices) and indices:
            padding = self.total_size - len(indices)
            repeats = (padding + len(indices) - 1) // len(indices)
            indices += (indices * repeats)[:padding]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples


# ============================================================================
# Distributed Data Parallel (DDP)
# ============================================================================


class DistributedDataParallel(Module):
    """DistributedDataParallel wrapper with gradient synchronization."""

    def __init__(self, module: Module, device_ids: List[int] = None, process_group=None):
        super().__init__()
        self.module = module
        self._modules["module"] = module
        self.device_ids = device_ids or [0]
        self.process_group = process_group
        self.bucket_size_mb = 25

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def sync_gradients(self):
        """Synchronize gradients across all ranks."""
        if not self.process_group or self.process_group.world_size <= 1:
            return
        for p in self.module.parameters():
            if p.grad is not None:
                pg = self.process_group
                pg.all_reduce(p.grad, NCCLRedOp.SUM)
                # Average
                p.grad = Tensor(p.grad.data / pg.world_size, device=p.grad.device)


# ============================================================================
# Model imports (lazy)
# ============================================================================


def _get_nn():
    from .nn import Linear, LayerNorm, TransformerBlock

    return Linear, LayerNorm, TransformerBlock


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "NCCLRedOp",
    "NCCLDataType",
    # Process Group
    "ProcessGroup",
    "init_process_group",
    "destroy_process_group",
    "get_world_size",
    "get_rank",
    "barrier",
    # Collective ops
    "all_reduce",
    "broadcast",
    # Data loading
    "DistributedSampler",
    # DDP
    "DistributedDataParallel",
]
