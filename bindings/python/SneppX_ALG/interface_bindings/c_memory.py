"""Memory allocator kernel bindings — allocator, pool.

Wraps C implementations in ``kernel/memory/`` with pure-Python fallback.
"""

from typing import Optional, Dict, List

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class MemoryAllocator:
    """General-purpose memory allocator with pooling."""

    def __init__(self, pool_size: int = 1 << 24):
        self._pool = bytearray(pool_size)
        self._allocations: Dict[int, int] = {}
        self._free_list: List[int] = list(range(0, pool_size, 4096))
        self._has_c = _HAS_C

    def malloc(self, size: int, alignment: int = 64) -> Optional[int]:
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        for i, offset in enumerate(self._free_list):
            if offset + aligned_size <= len(self._pool):
                self._free_list.pop(i)
                self._allocations[offset] = aligned_size
                return offset
        return None

    def free(self, offset: int) -> None:
        if offset in self._allocations:
            del self._allocations[offset]
            self._free_list.append(offset)
            self._free_list.sort()

    def get_buffer(self, offset: int, size: int) -> Optional[memoryview]:
        if offset in self._allocations:
            return memoryview(self._pool)[offset:offset + size]
        return None

    @property
    def used_bytes(self) -> int:
        return sum(self._allocations.values())

    @property
    def free_bytes(self) -> int:
        return len(self._pool) - self.used_bytes


class MemoryPool:
    """Fixed-size memory pool for fast allocation of same-sized blocks."""

    def __init__(self, block_size: int = 256, n_blocks: int = 1024):
        self._block_size = block_size
        self._pool = bytearray(block_size * n_blocks)
        self._free_blocks = list(range(n_blocks))
        self._has_c = _HAS_C

    def alloc(self) -> Optional[int]:
        if not self._free_blocks:
            return None
        return self._free_blocks.pop()

    def free(self, block_idx: int) -> None:
        if 0 <= block_idx < len(self._pool) // self._block_size:
            self._free_blocks.append(block_idx)

    def get_block(self, block_idx: int) -> Optional[memoryview]:
        if 0 <= block_idx < len(self._pool) // self._block_size:
            start = block_idx * self._block_size
            return memoryview(self._pool)[start:start + self._block_size]
        return None

    @property
    def available(self) -> int:
        return len(self._free_blocks)

    @property
    def total_blocks(self) -> int:
        return len(self._pool) // self._block_size
