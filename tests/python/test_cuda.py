"""Tests for CUDA device abstraction and tensor GPU support."""

import pytest
from SneppX_ALG.interface_bindings.cuda_device import (
    cuda_is_available,
    current_device,
    device_count,
    set_device,
    get_device_name,
    get_device_capability,
    CUDADevice,
    DeviceContext,
    CUDAMemoryPool,
    cuda_memory_summary,
    synchronize,
)
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_cuda_is_available():
    assert cuda_is_available()


def test_device_count():
    assert device_count() >= 2


def test_set_device():
    set_device(1)
    assert current_device() == 1

    set_device(0)
    assert current_device() == 0

    with pytest.raises(ValueError, match="not found"):
        set_device(device_count())


def test_get_device_properties():
    props0 = get_device_name(0)
    assert "Simulated CUDA Device 0" in props0
    assert get_device_capability(0) == (8, 0)


def test_cuda_device_class():
    dev = CUDADevice(0)
    assert dev.device_id == 0
    assert "Simulated CUDA Device 0" in dev.name
    assert dev.compute_capability == (8, 0)
    assert dev.total_memory > 0


def test_device_context():
    set_device(0)
    assert current_device() == 0

    with DeviceContext(1):
        assert current_device() == 1

    assert current_device() == 0


def test_memory_pool():
    pool = CUDAMemoryPool(0)
    assert pool.total_bytes == 16 * 1024**3
    assert pool.free_bytes == 16 * 1024**3

    allocated = pool.allocate(1024 * 1024)
    assert allocated > 0

    assert pool.allocated_bytes == 1024 * 1024
    assert pool.free_bytes == (16 * 1024**3 - 1024 * 1024)

    pool.free(1024 * 1024)
    assert pool.allocated_bytes == 0
    assert pool.free_bytes == pool.total_bytes


def test_memory_summary():
    summary = cuda_memory_summary()
    assert len(summary) >= 2
    for did in summary:
        assert "device" in summary[did]
        assert "allocated_bytes" in summary[did]
        assert "free_bytes" in summary[did]
        assert "total_bytes" in summary[did]


def test_tensors_on_cpu():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t1.device == "cpu"
    assert t1.numel == 4


def test_tensors_clone_cpu():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    t2 = t1.clone()
    assert t1.device == t2.device
    assert t1.device == "cpu"
    assert t1.data.tolist() == t2.data.tolist()


def test_tensors_to_cpu():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    t2 = t1.cpu()
    assert t1 is t2
    assert t2.device == "cpu"


def test_tensor_creation_with_device():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    assert t.device == "cpu"


def test_operations_require_same_device_error():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = Tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu")

    t3 = Tensor([[9.0, 10.0], [11.0, 12.0]], device="cpu")
    t4 = t3 + t1
    assert t4.device == "cpu"


def test_memory_pool_reset():
    pool = CUDAMemoryPool(0)
    pool.allocate(1024)
    assert pool.allocated_bytes == 1024

    CUDAMemoryPool.reset()
    assert pool.free_bytes == pool.total_bytes


def test_synchronize():
    synchronize()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
