"""Distributed Training — Python API for multi-GPU/multi-node."""

from typing import Dict, List, Optional, Callable
from .tensor import Tensor
from .nn import Module
from .optim import Optimizer
import os
import socket
import json
import numpy as np


class _NCCLBackend:
    def __init__(self):
        self._lib = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            import ctypes

            paths = ["libnccl.so", "libnccl.so.2", "libnccl.dylib", "nccl.dll"]
            for p in paths:
                try:
                    self._lib = ctypes.CDLL(p)
                    self._loaded = True
                    break
                except:
                    continue
        except:
            pass

    def all_reduce(self, data, op="sum"):
        self._load()
        return data

    def broadcast(self, data, root=0):
        self._load()
        return data


_nccl = _NCCLBackend()


class DistributedContext:
    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        self.master_port = int(os.environ.get("MASTER_PORT", "29500"))
        self.initialized = False

    def init_process_group(self, backend="nccl"):
        if self.initialized:
            return
        if self.world_size <= 1:
            return
        if backend == "nccl":
            global _nccl
            _nccl._load()
        self.initialized = True
        print(f"[Distributed] Rank {self.rank}/{self.world_size} initialized")

    def destroy_process_group(self):
        self.initialized = False

    def barrier(self):
        pass

    def all_reduce(self, tensor: Tensor, op="sum") -> Tensor:
        if self.world_size <= 1:
            return tensor
        raise RuntimeError(
            "No validated multi-rank collective backend is available; "
            "all_reduce must not silently return unreduced data"
        )

    def all_gather(self, tensor: Tensor) -> List[Tensor]:
        if self.world_size <= 1:
            return [tensor]
        raise RuntimeError(
            "No validated multi-rank collective backend is available; "
            "all_gather must not fabricate copies of the local tensor"
        )

    def reduce_scatter(self, tensor: Tensor) -> Tensor:
        if self.world_size <= 1:
            return tensor
        raise RuntimeError(
            "No validated multi-rank collective backend is available; "
            "reduce_scatter must not silently return unreduced data"
        )


_global_context = DistributedContext()


def init_process_group(backend="nccl"):
    _global_context.init_process_group(backend)


def destroy_process_group():
    _global_context.destroy_process_group()


def get_world_size():
    return _global_context.world_size


def get_rank():
    return _global_context.rank


def barrier():
    _global_context.barrier()


def all_reduce(tensor: Tensor, op="sum") -> Tensor:
    return _global_context.all_reduce(tensor, op)


def all_gather(tensor: Tensor) -> List[Tensor]:
    return _global_context.all_gather(tensor)


def reduce_scatter(tensor: Tensor) -> Tensor:
    return _global_context.reduce_scatter(tensor)


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


class DistributedDataParallel(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._modules["module"] = module

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def all_reduce_grads(self):
        for p in self.module.parameters():
            if p.grad is not None:
                all_reduce(p.grad, "sum")
                p.grad = Tensor.from_numpy(p.grad.data / get_world_size())


def launch(train_fn: Callable, num_nodes: int = 1, num_gpus: int = 1):
    if num_nodes * num_gpus <= 1:
        train_fn()
        return
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--nnodes={num_nodes}",
        sys.argv[0],
    ] + sys.argv[1:]
    subprocess.run(cmd)
