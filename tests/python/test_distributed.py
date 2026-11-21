"""Tests for distributed.py — distributed training utilities."""

import os
import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    Module,
    DistributedContext,
    DistributedSampler,
    DistributedDataParallel,
    init_process_group,
    destroy_process_group,
    get_world_size,
    get_rank,
    all_reduce,
    _HAS_C_BACKEND,
)


def test_distributed_context_defaults():
    ctx = DistributedContext()
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    print("  test_distributed_context_defaults PASS")


def test_init_process_group_single():
    init_process_group("nccl")
    assert get_world_size() == 1
    assert get_rank() == 0
    destroy_process_group()
    print("  test_init_process_group_single PASS")


def test_all_reduce_single():
    t = Tensor.ones((4,))
    result = all_reduce(t, "sum")
    assert np.allclose(result.data, 1.0)
    print("  test_all_reduce_single PASS")


def test_distributed_sampler_single():
    data = list(range(10))
    sampler = DistributedSampler(data, num_replicas=1, rank=0, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 10
    assert indices == list(range(10))
    print("  test_distributed_sampler_single PASS")


def test_distributed_sampler_multi():
    data = list(range(10))
    sampler_0 = DistributedSampler(data, num_replicas=2, rank=0, shuffle=False)
    sampler_1 = DistributedSampler(data, num_replicas=2, rank=1, shuffle=False)
    indices_0 = list(sampler_0)
    indices_1 = list(sampler_1)
    assert len(indices_0) == 5
    assert len(indices_1) == 5
    all_indices = sorted(indices_0 + indices_1)
    assert all_indices == list(range(10))
    print("  test_distributed_sampler_multi PASS")


def test_distributed_sampler_shuffle():
    data = list(range(100))
    sampler = DistributedSampler(data, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert sorted(epoch0) == list(range(100))
    assert sorted(epoch1) == list(range(100))
    print("  test_distributed_sampler_shuffle PASS")


def test_ddp_wrapper():
    if not _HAS_C_BACKEND:
        print("  test_ddp_wrapper SKIP (no C backend)")
        return
    linear = Linear(4, 3)
    ddp = DistributedDataParallel(linear)
    x = Tensor.randn((2, 4))
    out = ddp(x)
    assert out.shape == (2, 3)
    assert len(ddp.parameters()) == 2
    ddp.all_reduce_grads()
    print("  test_ddp_wrapper PASS")


def test_context_env_override():
    os.environ["WORLD_SIZE"] = "4"
    os.environ["RANK"] = "2"
    os.environ["LOCAL_RANK"] = "1"
    ctx = DistributedContext()
    assert ctx.world_size == 4
    assert ctx.rank == 2
    assert ctx.local_rank == 1
    del os.environ["WORLD_SIZE"]
    del os.environ["RANK"]
    del os.environ["LOCAL_RANK"]
    print("  test_context_env_override PASS")


if __name__ == "__main__":
    test_distributed_context_defaults()
    test_init_process_group_single()
    test_all_reduce_single()
    test_distributed_sampler_single()
    test_distributed_sampler_multi()
    test_distributed_sampler_shuffle()
    test_ddp_wrapper()
    test_context_env_override()
    print("\nAll distributed tests passed!")
