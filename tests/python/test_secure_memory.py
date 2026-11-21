"""Tests for secure_memory.py — S1 memory hardening."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import SecureAllocator, StackCanary, ASLR, MemoryHardening, MemoryLeakDetector


def test_allocator_malloc_free():
    a = SecureAllocator()
    addr = a.malloc(64)
    buf = a.get_buffer(addr)
    assert buf is not None
    assert len(buf) == 64
    a.free(addr)
    assert a.get_buffer(addr) is None


def test_allocator_secure_free():
    a = SecureAllocator()
    addr = a.malloc(32)
    a.secure_free(addr)
    assert a.get_buffer(addr) is None


def test_allocator_multiple():
    a = SecureAllocator()
    addrs = [a.malloc(16) for _ in range(10)]
    for addr in addrs:
        assert a.get_buffer(addr) is not None
    for addr in addrs:
        a.free(addr)


def test_canary_value():
    c = StackCanary()
    assert len(c.value) == 8


def test_canary_check():
    c = StackCanary()
    assert c.check(c.value)
    assert not c.check(b'\x00' * 8)
    assert not c.check(b'\xff' * 8)


def test_canary_refresh():
    c = StackCanary()
    old = c.value
    c.refresh()
    assert c.value != old


def test_aslr_randomize():
    base = 0x1000
    r = ASLR.randomize_address(base, 4096)
    assert base <= r < base + 4096


def test_memory_hardening():
    m = MemoryHardening()
    assert m.w_xor_x(True)


def test_leak_detector():
    d = MemoryLeakDetector()
    d.track(0x1234, "test")
    d.track(0x5678, "test2")
    assert d.leak_count == 2
    d.untrack(0x1234)
    assert d.leak_count == 1
    report = d.report()
    assert 0x5678 in report


if __name__ == "__main__":
    test_allocator_malloc_free()
    test_allocator_secure_free()
    test_allocator_multiple()
    test_canary_value()
    test_canary_check()
    test_canary_refresh()
    test_aslr_randomize()
    test_memory_hardening()
    test_leak_detector()
    print("All secure_memory tests passed.")
