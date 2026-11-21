"""CUDA device abstraction — simulated CUDA device management for Python backend."""

import threading
from typing import Optional, Dict, List

_SIMULATED_CUDA = True

_current_device = threading.local()
_current_device.device_id = 0

_device_properties = {
    0: {
        "name": "Simulated CUDA Device 0",
        "compute_capability": (8, 0),
        "total_memory": 16 * 1024**3,
        "free_memory": 16 * 1024**3,
        "multi_processor_count": 80,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_block_dim": (1024, 1024, 64),
        "max_grid_dim": (2147483647, 65535, 65535),
        "max_shared_memory_per_block": 49152,
    },
    1: {
        "name": "Simulated CUDA Device 1",
        "compute_capability": (8, 0),
        "total_memory": 24 * 1024**3,
        "free_memory": 24 * 1024**3,
        "multi_processor_count": 100,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_block_dim": (1024, 1024, 64),
        "max_grid_dim": (2147483647, 65535, 65535),
        "max_shared_memory_per_block": 49152,
    },
}


def cuda_is_available() -> bool:
    return _SIMULATED_CUDA


def set_device(device_id: int) -> None:
    if not _SIMULATED_CUDA:
        return
    if device_id >= device_count():
        raise ValueError(f"Device {device_id} not found (have {device_count()})")
    _current_device.device_id = device_id


def current_device() -> int:
    return getattr(_current_device, "device_id", 0)


def device_count() -> int:
    return len(_device_properties) if _SIMULATED_CUDA else 0


def get_device_capability(device_id: Optional[int] = None) -> tuple:
    did = device_id if device_id is not None else current_device()
    props = _device_properties.get(did, {})
    return props.get("compute_capability", (0, 0))


def get_device_name(device_id: Optional[int] = None) -> str:
    did = device_id if device_id is not None else current_device()
    props = _device_properties.get(did, {})
    return props.get("name", "Unknown")


class CUDADevice:
    def __init__(self, device_id: int):
        if device_id >= device_count():
            raise ValueError(f"Device {device_id} not found (have {device_count()})")
        self._device_id = device_id
        self._props = _device_properties[device_id]

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def name(self) -> str:
        return self._props["name"]

    @property
    def compute_capability(self) -> tuple:
        return self._props["compute_capability"]

    @property
    def total_memory(self) -> int:
        return self._props["total_memory"]

    @property
    def free_memory(self) -> int:
        return self._props["free_memory"]

    @property
    def multi_processor_count(self) -> int:
        return self._props["multi_processor_count"]

    @property
    def warp_size(self) -> int:
        return self._props["warp_size"]

    @property
    def max_threads_per_block(self) -> int:
        return self._props["max_threads_per_block"]

    def __repr__(self) -> str:
        return (
            f"CUDADevice({self._device_id}: {self._props['name']}, "
            f"CC={'.'.join(map(str, self.compute_capability))}, "
            f"Mem={self.total_memory / 1024**3:.0f}GiB)"
        )


class DeviceContext:
    def __init__(self, device_id: int):
        self._device_id = device_id
        self._prev_id = None

    def __enter__(self):
        self._prev_id = current_device()
        set_device(self._device_id)
        return self

    def __exit__(self, *args):
        if self._prev_id is not None:
            set_device(self._prev_id)


_simulated_allocations: Dict[int, List[int]] = {}
_alloc_lock = threading.Lock()


class CUDAMemoryPool:
    def __init__(self, device_id: Optional[int] = None):
        self._device_id = device_id if device_id is not None else current_device()
        if self._device_id not in _simulated_allocations:
            with _alloc_lock:
                if self._device_id not in _simulated_allocations:
                    _simulated_allocations[self._device_id] = []

    def allocate(self, nbytes: int) -> int:
        with _alloc_lock:
            _simulated_allocations[self._device_id].append(nbytes)
            props = _device_properties[self._device_id]
            props["free_memory"] = max(0, props["free_memory"] - nbytes)
        return id(self)

    def free(self, nbytes: int) -> None:
        with _alloc_lock:
            allocs = _simulated_allocations.get(self._device_id, [])
            if nbytes in allocs:
                allocs.remove(nbytes)
            props = _device_properties[self._device_id]
            props["free_memory"] = min(
                props["total_memory"], props["free_memory"] + nbytes
            )

    @property
    def allocated_bytes(self) -> int:
        return sum(_simulated_allocations.get(self._device_id, []))

    @property
    def free_bytes(self) -> int:
        return _device_properties[self._device_id]["free_memory"]

    @property
    def total_bytes(self) -> int:
        return _device_properties[self._device_id]["total_memory"]

    @staticmethod
    def reset() -> None:
        with _alloc_lock:
            _simulated_allocations.clear()
            for did, props in _device_properties.items():
                props["free_memory"] = props["total_memory"]


def cuda_memory_summary() -> Dict[int, dict]:
    result = {}
    for did in range(device_count()):
        pool = CUDAMemoryPool(did)
        result[did] = {
            "device": get_device_name(did),
            "allocated_bytes": pool.allocated_bytes,
            "free_bytes": pool.free_bytes,
            "total_bytes": pool.total_bytes,
        }
    return result


def synchronize() -> None:
    pass


__all__ = [
    "cuda_is_available",
    "set_device",
    "current_device",
    "device_count",
    "get_device_capability",
    "get_device_name",
    "CUDADevice",
    "DeviceContext",
    "CUDAMemoryPool",
    "cuda_memory_summary",
    "synchronize",
]
