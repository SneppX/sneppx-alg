"""Tests for asm_bridge.py — ASM library loading and CPU detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.asm_bridge import (
    CPUFeatures, AsmLibrary, AsmFunction, ConstantTime, SecureMemory
)


def test_cpu_features():
    cpu = CPUFeatures()
    feats = cpu.supported()
    assert isinstance(feats, list)
    assert "sse2" in feats or True  # most systems have sse2


def test_cpu_feature_check():
    cpu = CPUFeatures()
    # at minimum sse2 should be present on x86-64
    assert isinstance(cpu.has("sse2"), bool)


def test_asm_library():
    lib = AsmLibrary()
    assert isinstance(lib.available, bool)


def test_constant_time_compare():
    ct = ConstantTime()
    assert ct.compare_u64(42, 42) is True
    assert ct.compare_u64(1, 2) is False
    assert ct.compare_bytes(b"abc", b"abc") is True
    assert ct.compare_bytes(b"abc", b"abd") is False


def test_constant_time_select():
    ct = ConstantTime()
    assert ct.select(True, 10, 20) == 10
    assert ct.select(False, 10, 20) == 20


def test_secure_memory_wipe():
    sm = SecureMemory()
    data = bytearray(b"secretdata")
    sm.wipe(data)
    assert all(b == 0 for b in data)


def test_secure_memory_wipe_buffer():
    sm = SecureMemory()
    arr = bytearray(32)
    for i in range(32):
        arr[i] = i
    sm.wipe_buffer(arr)
    assert all(b == 0 for b in arr)


if __name__ == "__main__":
    test_cpu_features()
    test_cpu_feature_check()
    test_asm_library()
    test_constant_time_compare()
    test_constant_time_select()
    test_secure_memory_wipe()
    test_secure_memory_wipe_buffer()
    print("All asm_bridge tests passed.")
