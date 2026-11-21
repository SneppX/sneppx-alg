"""Fully Sharded Data Parallel (FSDP) — ZeRO-3 style sharding."""

from typing import Dict, List, Optional, Tuple, Union, Callable, Set
from enum import Enum
from dataclasses import dataclass, field

from .tensor import Tensor
from .nn import Module
import numpy as np


class ShardingStrategy(Enum):
    NO_SHARD = "no_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    SHARD_GRAD_OP_OPT = "shard_grad_op_opt"
    FULL_SHARD = "full_shard"


class MixedPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class FSDPConfig:
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    mixed_precision: MixedPrecision = MixedPrecision.FP32
    cpu_offload: bool = False
    backward_prefetch: bool = True
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    param_dtype: str = "float32"
    reduce_dtype: str = "float32"
    buffer_dtype: str = "float32"
    min_num_params: int = 1_000_000
    ignored_modules: List[str] = field(default_factory=list)


def _shard_tensor(t: Tensor, rank: int, world_size: int, dim: int = 0) -> Tensor:
    if world_size <= 1:
        return t
    numel = t.data.size
    shard_size = numel // world_size
    remainder = numel % world_size
    if rank < remainder:
        start = rank * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = rank * shard_size + remainder
        end = start + shard_size
    flat = t.data.ravel()
    shard_data = flat[start:end].copy()
    return Tensor.from_numpy(shard_data.reshape(-1))


def _gather_tensors(shards: List[np.ndarray], world_size: int) -> np.ndarray:
    return np.concatenate(shards)


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


class _ShardedParam:
    __slots__ = ("full_shape", "shard", "shard_size", "full", "dtype", "device", "grad_shard")

    def __init__(self, full_shape: tuple, shard: Tensor, dtype, device):
        self.full_shape = full_shape
        self.shard = shard
        self.shard_size = shard.data.size
        self.full: Optional[np.ndarray] = None
        self.dtype = dtype
        self.device = device
        self.grad_shard: Optional[np.ndarray] = None


class FullyShardedDataParallel(Module):
    def __init__(
        self,
        module: Module,
        config: Optional[FSDPConfig] = None,
    ):
        super().__init__()
        self._module = module
        self.config = config or FSDPConfig()
        self._world_size = 1
        self._rank = 0
        self._initialized = False
        self._sharded_params: Dict[str, _ShardedParam] = {}
        self._flat_param_names: List[str] = []
        self._param_name_to_shard: Dict[str, str] = {}

    def _try_init_dist(self):
        if self._initialized:
            return
        try:
            from .distributed import get_world_size, get_rank
            self._world_size = get_world_size()
            self._rank = get_rank()
        except Exception:
            self._world_size = 1
            self._rank = 0
        self._initialized = True

    def _should_shard(self, name: str, param: Tensor) -> bool:
        if param.data.size < self.config.min_num_params:
            return False
        for ign in self.config.ignored_modules:
            if ign in name:
                return False
        return True

    def _shard_parameters(self):
        self._try_init_dist()
        if self._world_size <= 1:
            return
        for name, param in self._module.named_parameters():
            if not self._should_shard(name, param):
                continue
            shard = _shard_tensor(param, self._rank, self._world_size)
            sp = _ShardedParam(
                full_shape=param.shape,
                shard=shard,
                dtype=param.dtype_name,
                device=param.device,
            )
            self._sharded_params[name] = sp
            self._flat_param_names.append(name)
            self._param_name_to_shard[name] = name

    def all_gather(self) -> Set[str]:
        gathered: Set[str] = set()
        if self._world_size <= 1:
            for name, param in self._module.named_parameters():
                gathered.add(name)
            return gathered
        from .distributed import all_gather as dist_all_gather
        for name, sp in self._sharded_params.items():
            shard_t = sp.shard
            try:
                gathered_tensors = dist_all_gather(shard_t)
                full_np = np.concatenate([t.data for t in gathered_tensors])
            except Exception:
                full_np = sp.shard.data.copy()
            sp.full = full_np.reshape(sp.full_shape).astype(np.float32).copy()
            gathered.add(name)
        for name, param in self._module.named_parameters():
            if name not in self._sharded_params:
                gathered.add(name)
        return gathered

    def reduce_gradients(self):
        if self._world_size <= 1:
            return
        from .distributed import all_reduce as dist_all_reduce
        for name, sp in self._sharded_params.items():
            for p_name, param in self._module.named_parameters():
                if p_name == name and hasattr(param, 'grad') and param.grad is not None:
                    grad_shard = _shard_tensor(param.grad, self._rank, self._world_size)
                    try:
                        reduced = dist_all_reduce(grad_shard, op="sum")
                        grad_data = reduced.data / self._world_size
                    except Exception:
                        grad_data = grad_shard.data
                    sp.grad_shard = grad_data
                    param.grad = None

    def unscale_params(self, gathered: Set[str]):
        if self._world_size <= 1:
            return
        for name, sp in self._sharded_params.items():
            if name not in gathered:
                continue
            flat_full = sp.full.ravel()
            ws = self._world_size
            rem = flat_full.size % ws
            if self._rank < rem:
                start = self._rank * (flat_full.size // ws + 1)
                end = start + flat_full.size // ws + 1
            else:
                start = self._rank * (flat_full.size // ws) + rem
                end = start + flat_full.size // ws
            sp.shard = Tensor.from_numpy(flat_full[start:end].copy())
            sp.full = None

    def forward(self, x: Tensor) -> Tensor:
        gathered = self.all_gather()
        self._restore_params(gathered)
        try:
            out = self._module(x)
        finally:
            self._free_full_params(gathered)
        return out

    def _restore_params(self, gathered: Set[str]):
        if self._world_size <= 1:
            return
        for name, sp in self._sharded_params.items():
            if name in gathered and sp.full is not None:
                for p_name, param in self._module.named_parameters():
                    if p_name == name:
                        param.data = sp.full.astype(param.dtype_name)

    def _free_full_params(self, gathered: Set[str]):
        if self._world_size <= 1:
            return
        for name, sp in self._sharded_params.items():
            sp.full = None

    def parameters(self):
        return self._module.parameters()

    def named_parameters(self):
        return self._module.named_parameters()

    def state_dict(self) -> dict:
        if self._world_size <= 1:
            return self._module.state_dict()
        sd = self._module.state_dict()
        from .distributed import all_gather as dist_all_gather
        for name, sp in self._sharded_params.items():
            try:
                gathered_tensors = dist_all_gather(sp.shard)
                all_shards = [t.data for t in gathered_tensors]
                full = np.concatenate(all_shards)
                sd[name] = full.reshape(sp.full_shape)
            except Exception:
                sd[name] = sp.shard.data.reshape(sp.full_shape)
        return sd

    def load_state_dict(self, state_dict: dict):
        self._module.load_state_dict(state_dict)

    def train(self):
        self._module.train()

    def eval(self):
        self._module.eval()

    def to(self, device: str):
        self._module.to(device)
        return self

    @property
    def module(self) -> Module:
        return self._module


def shard_model(
    model: Module,
    config: Optional[FSDPConfig] = None,
) -> FullyShardedDataParallel:
    fsdp_model = FullyShardedDataParallel(model, config)
    fsdp_model._shard_parameters()
    return fsdp_model


def fsdp_save_checkpoint(
    fsdp_model: FullyShardedDataParallel,
    path: str,
    optimizer_state: Optional[dict] = None,
):
    import pickle
    state = {
        "model": fsdp_model.state_dict(),
        "optimizer": optimizer_state,
        "world_size": fsdp_model._world_size,
        "rank": fsdp_model._rank,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def fsdp_load_checkpoint(
    fsdp_model: FullyShardedDataParallel,
    path: str,
    load_optimizer: bool = True,
) -> Optional[dict]:
    import pickle
    with open(path, "rb") as f:
        state = pickle.load(f)
    fsdp_model.load_state_dict(state.get("model", {}))
    return state.get("optimizer") if load_optimizer else None
