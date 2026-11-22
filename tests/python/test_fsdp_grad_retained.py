"""Regression: FSDP gradient reduction must leave a usable local grad."""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings import fsdp as fsdp_mod
from SneppX_ALG.interface_bindings.nn import Module
from SneppX_ALG.interface_bindings.tensor import Tensor


class _Toy(Module):
    def __init__(self):
        super().__init__()
        self.p = Tensor(np.arange(8, dtype=np.float32))


def test_reduce_gradients_keeps_shard_grad():
    wrapped = fsdp_mod.FullyShardedDataParallel(
        _Toy(), fsdp_mod.FSDPConfig(min_num_params=1)
    )
    wrapped._initialized = True
    wrapped._world_size = 2
    wrapped._rank = 0
    wrapped._shard_parameters()

    wrapped._module.p.grad = Tensor(np.ones(8, dtype=np.float32))
    wrapped.reduce_gradients()

    assert wrapped._module.p.grad is not None
    assert np.allclose(np.asarray(wrapped._module.p.grad.data), 0.5)
