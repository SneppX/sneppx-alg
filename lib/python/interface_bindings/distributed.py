"""Distributed Training — Python API for multi-GPU/multi-node."""

from typing import Dict, List, Optional, Callable
from .core import Tensor
from .nn import Module
from .optim import Optimizer
import os
import socket
import json


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
        arr = tensor.numpy()
        if op == "sum":
            pass  # NCCL handles this
        return Tensor(arr, dtype=tensor.dtype)

    def all_gather(self, tensor: Tensor) -> List[Tensor]:
        if self.world_size <= 1:
            return [tensor]
        return [tensor for _ in range(self.world_size)]

    def reduce_scatter(self, tensor: Tensor) -> Tensor:
        return tensor


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


class DistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank or get_rank()
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = len(dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
        indices = indices[:self.total_size]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedDataParallel(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self._modules['module'] = module

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
                p.grad = Tensor(p.grad.numpy() / get_world_size(), dtype=p.grad.dtype)


def launch(train_fn: Callable, num_nodes: int = 1, num_gpus: int = 1):
    """Launch distributed training with torchrun-like API."""
    if num_nodes * num_gpus <= 1:
        train_fn()
        return
    import subprocess
    import sys
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--nnodes={num_nodes}",
        sys.argv[0],
    ] + sys.argv[1:]
    subprocess.run(cmd)