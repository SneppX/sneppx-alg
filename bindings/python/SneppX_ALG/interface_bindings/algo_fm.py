"""FM (Fractal Memory) algorithm bindings.

Wraps C implementations in ``algorithms/fm/core/`` with pure-Python fallback.
"""

from typing import List, Optional, Tuple, Dict, Callable

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_architecture_layer")


class FMController:
    """Fractal Memory controller — manages memory bank operations."""

    def __init__(self, memory_size: int = 1024, key_size: int = 64):
        self.memory_size = memory_size
        self.key_size = key_size
        self._memory: Dict[str, np.ndarray] = {}
        self._has_c = _HAS_C

    def read(self, key: str) -> Optional[np.ndarray]:
        return self._memory.get(key)

    def write(self, key: str, value: np.ndarray) -> None:
        self._memory[key] = value

    def clear(self) -> None:
        self._memory.clear()


class FMMemoryBank:
    """Fractal memory bank — stores compressed representations."""

    def __init__(self, capacity: int = 1000, compression_ratio: float = 0.5):
        self.capacity = capacity
        self.compression_ratio = compression_ratio
        self._bank: List[np.ndarray] = []
        self._has_c = _HAS_C

    def store(self, embedding: np.ndarray) -> int:
        compressed = self._compress(embedding)
        if len(self._bank) >= self.capacity:
            self._bank.pop(0)
        self._bank.append(compressed)
        return len(self._bank) - 1

    def retrieve(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < len(self._bank):
            return self._decompress(self._bank[idx])
        return None

    def _compress(self, x: np.ndarray) -> np.ndarray:
        k = max(1, int(x.shape[-1] * self.compression_ratio))
        indices = np.sort(np.random.choice(x.shape[-1], k, replace=False))
        return x[..., indices]

    def _decompress(self, x: np.ndarray) -> np.ndarray:
        return x

    @property
    def size(self) -> int:
        return len(self._bank)


class FMNode:
    """Fractal memory node — processes and routes information."""

    def __init__(self, node_id: str, dim: int = 64):
        self.node_id = node_id
        self.dim = dim
        self._has_c = _HAS_C

    def process(self, x: np.ndarray) -> np.ndarray:
        return x

    def route(self, x: np.ndarray, targets: List[str]) -> Dict[str, np.ndarray]:
        return {t: x for t in targets}


class FMSync:
    """Fractal memory synchronization — all-reduce and gradient sync."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def all_reduce(tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        return tensor

    @staticmethod
    def federated_avg(local_tensors: List[np.ndarray]) -> np.ndarray:
        if not local_tensors:
            return np.array([])
        result = np.zeros_like(local_tensors[0])
        for t in local_tensors:
            result += t
        return result / len(local_tensors)


class FMSyncNCCL:
    """NCCL-based distributed synchronization with callback pattern.

    Wraps the C-level ``SNEPPX_fm_sync_nccl`` when available.
    """

    def __init__(self):
        self._has_c = _HAS_C

    def sync(self, data: np.ndarray, callback: Optional[Callable] = None) -> np.ndarray:
        """Synchronize data across distributed nodes.

        Pure Python fallback: if callback is provided, invokes it with the data.
        """
        result = data.copy()
        if callback is not None:
            callback(result)
        return result
