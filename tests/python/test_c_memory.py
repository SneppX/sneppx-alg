"""Tests for c_memory.py — memory allocator kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import MemoryAllocator, MemoryPool


def test_allocator_malloc_free():
    a = MemoryAllocator(pool_size=1 << 20)
    offset = a.malloc(1024)
    assert offset is not None
    assert offset >= 0
    a.free(offset)
    assert offset not in a._allocations


def test_allocator_get_buffer():
    a = MemoryAllocator(pool_size=4096)
    offset = a.malloc(64)
    buf = a.get_buffer(offset, 64)
    assert buf is not None
    assert len(buf) == 64


def test_allocator_get_buffer_freed():
    a = MemoryAllocator()
    offset = a.malloc(64)
    a.free(offset)
    assert a.get_buffer(offset, 64) is None


def test_allocator_out_of_memory():
    a = MemoryAllocator(pool_size=128)
    assert a.malloc(256) is None


def test_allocator_used_bytes():
    a = MemoryAllocator(pool_size=4096)
    a.malloc(1024)
    assert a.used_bytes >= 1024
    assert a.free_bytes > 0


def test_memory_pool_alloc_free():
    p = MemoryPool(block_size=128, n_blocks=4)
    b1 = p.alloc()
    b2 = p.alloc()
    assert b1 is not None
    assert b2 is not None
    assert b1 != b2
    p.free(b1)
    assert p.available >= 1


def test_memory_pool_exhaustion():
    p = MemoryPool(block_size=64, n_blocks=2)
    assert p.alloc() is not None
    assert p.alloc() is not None
    assert p.alloc() is None


def test_memory_pool_get_block():
    p = MemoryPool(block_size=32, n_blocks=4)
    b = p.alloc()
    block = p.get_block(b)
    assert block is not None
    assert len(block) == 32


def test_memory_pool_invalid_block():
    p = MemoryPool(block_size=32, n_blocks=2)
    assert p.get_block(999) is None


def test_memory_pool_stats():
    p = MemoryPool(block_size=64, n_blocks=10)
    assert p.total_blocks == 10
    assert p.available == 10
    p.alloc()
    assert p.available == 9


if __name__ == "__main__":
    test_allocator_malloc_free()
    test_allocator_get_buffer()
    test_allocator_get_buffer_freed()
    test_allocator_out_of_memory()
    test_allocator_used_bytes()
    test_memory_pool_alloc_free()
    test_memory_pool_exhaustion()
    test_memory_pool_get_block()
    test_memory_pool_invalid_block()
    test_memory_pool_stats()
    print("All c_memory tests passed.")
