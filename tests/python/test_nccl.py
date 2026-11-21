"""Tests for NCCL distributed communication."""

import numpy as np
import pytest
from SneppX_ALG.interface_bindings.nccl import (
    NCCLRedOp,
    ProcessGroup,
    init_process_group,
    get_world_size,
    get_rank,
    destroy_process_group,
    all_reduce,
    broadcast,
    DistributedSampler,
    DistributedDataParallel,
)
from SneppX_ALG.interface_bindings.nn import Linear, LayerNorm, TransformerBlock
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_process_group_init():
    """Test process group initialization."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()
    assert pg._initialized
    pg.destroy()


def test_process_group():
    """Test ProcessGroup high-level API."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()
    assert pg.world_size == 2
    assert pg.rank == 0
    pg.destroy()


def test_all_reduce_cpu():
    """Test all_reduce with CPU tensors."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=-1)
    pg.initialize()

    t = Tensor([1.0, 2.0, 3.0])
    result = pg.all_reduce(t, NCCLRedOp.SUM)

    # CPU fallback just returns same tensor
    np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])
    pg.destroy()


def test_all_reduce_cuda():
    """Test all_reduce with CUDA tensors (fallback)."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    t = Tensor([1.0, 2.0, 3.0], device="cuda")
    result = pg.all_reduce(t, NCCLRedOp.SUM)

    # Falls back to no-op in test environment
    assert result.device.startswith("cuda")
    pg.destroy()


def test_broadcast():
    """Test broadcast operation."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    t = Tensor([1.0, 2.0, 3.0])
    result = pg.broadcast(t, root=0)
    np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])
    pg.destroy()


def test_barrier():
    """Test barrier synchronization."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()
    pg.barrier()  # Should not hang
    pg.destroy()


def test_init_process_group():
    """Test global process group initialization."""
    pg = init_process_group(world_size=4, rank=2, device_id=0)
    assert get_world_size() == 4
    assert get_rank() == 2
    destroy_process_group()
    assert get_world_size() == 1


def test_distributed_sampler():
    """Test DistributedSampler for multi-GPU data loading."""
    dataset = list(range(100))

    sampler = DistributedSampler(dataset, num_replicas=4, rank=0, shuffle=False)
    assert len(sampler) == 25

    # Test sharding
    indices = list(sampler)
    assert len(indices) == 25
    assert indices[0] == 0
    assert indices[-1] == 96  # 0, 4, 8, ..., 96


def test_distributed_sampler_shuffle():
    """Test DistributedSampler with shuffling."""
    dataset = list(range(100))

    sampler = DistributedSampler(dataset, num_replicas=4, rank=0, shuffle=True)
    sampler.set_epoch(42)
    indices = list(sampler)
    assert len(indices) == 25

    # Different epoch should give different order
    sampler.set_epoch(43)
    indices2 = list(sampler)
    assert indices != indices2


def test_ddp_linear():
    """Test DDP with Linear layer."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    model = Linear(10, 5)
    ddp = DistributedDataParallel(model, device_ids=[0], process_group=pg)

    x = Tensor.randn((4, 10), dtype="float32", device="cuda")
    out = ddp(x)
    assert out.shape == (4, 5)

    # Test gradient sync
    out.sum().backward()
    ddp.sync_gradients()

    pg.destroy()


def test_ddp_layernorm():
    """Test DDP with LayerNorm."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    model = LayerNorm(32)
    ddp = DistributedDataParallel(model, device_ids=[0], process_group=pg)

    x = Tensor.randn((4, 16, 32), dtype="float32", device="cuda")
    out = ddp(x)
    assert out.shape == (4, 16, 32)

    out.sum().backward()
    ddp.sync_gradients()

    pg.destroy()


def test_ddp_transformer_block():
    """Test DDP with TransformerBlock."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    model = TransformerBlock(dim=64, num_heads=4, ffn_dim=256)
    ddp = DistributedDataParallel(model, device_ids=[0], process_group=pg)

    x = Tensor.randn((2, 8, 64), dtype="float32", device="cuda")
    out = ddp(x)
    assert out.shape == (2, 8, 64)

    out.sum().backward()
    ddp.sync_gradients()

    pg.destroy()


def test_global_operations():
    """Test global NCCL functions."""
    pg = ProcessGroup(world_size=2, rank=0, device_id=0)
    pg.initialize()

    t = Tensor([1.0, 2.0, 3.0])
    result = all_reduce(t, NCCLRedOp.SUM)
    np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])

    result = broadcast(t, root=0)
    np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])

    result = broadcast(t, root=1)
    np.testing.assert_allclose(result.data, [1.0, 2.0, 3.0])

    pg.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
