"""Tests for driver_bindings.py — hardware driver interfaces."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings.driver_bindings import (
    CUDADriver, CUDADeviceProps, ROCmDriver, TPUDriver
)


def test_cuda_available():
    assert CUDADriver.is_available() is False


def test_cuda_device_count():
    d = CUDADriver()
    assert d.get_device_count() == 0


def test_cuda_device_props():
    d = CUDADriver()
    props = d.get_device_props(0)
    assert isinstance(props, CUDADeviceProps)
    assert props.warp_size == 32


def test_cuda_set_get_device():
    d = CUDADriver()
    d.set_device(1)
    assert d.get_device() == 1
    d.set_device(0)


def test_cuda_context():
    d = CUDADriver()
    ctx = d.create_context(0)
    assert ctx is not None
    assert ctx["device_id"] == 0
    assert d.context_error(ctx) == 0
    d.destroy_context(ctx)


def test_cuda_stream():
    d = CUDADriver()
    s = d.stream_create()
    assert s is not None
    assert d.stream_synchronize(s) == 0
    d.stream_destroy(s)


def test_cuda_event():
    d = CUDADriver()
    e = d.event_create()
    assert e is not None
    s = d.stream_create()
    assert d.event_record(e, s) == 0
    assert d.event_synchronize(e) == 0
    d.event_destroy(e)
    d.stream_destroy(s)


def test_cuda_memory():
    d = CUDADriver()
    ptr = d.mem_alloc(1024)
    assert ptr is not None
    arr = np.zeros(256, dtype=np.float32)
    assert d.mem_htod(ptr, arr, arr.nbytes) == 0
    out = np.zeros(256, dtype=np.float32)
    assert d.mem_dtoh(out, ptr, out.nbytes) == 0
    assert d.mem_set(ptr, 0, 1024) == 0
    d.mem_free(ptr)


def test_cuda_launch_kernel():
    d = CUDADriver()
    assert d.launch_kernel("test_kernel", (1, 1, 1), (32, 1, 1)) == 0


def test_rocm_available():
    assert ROCmDriver.is_available() is False


def test_rocm_lifecycle():
    r = ROCmDriver()
    assert r.get_device_count() == 0
    r.set_device(0)
    ctx = r.create_context(0)
    assert ctx is not None
    r.destroy_context(ctx)
    s = r.stream_create()
    assert s is not None
    assert r.stream_synchronize(s) == 0
    r.stream_destroy(s)


def test_rocm_memory():
    r = ROCmDriver()
    ptr = r.mem_alloc(512)
    assert ptr is not None
    arr = np.ones(128, dtype=np.float32)
    assert r.mem_htod(ptr, arr, arr.nbytes) == 0
    r.mem_free(ptr)


def test_tpu_available():
    assert TPUDriver.is_available() is False


def test_tpu_lifecycle():
    t = TPUDriver()
    assert t.get_device_count() == 0
    ctx = t.create_context(0)
    assert ctx is not None
    t.destroy_context(ctx)


def test_tpu_memory():
    t = TPUDriver()
    ctx = t.create_context(0)
    ptr = t.mem_alloc(256, ctx)
    assert ptr is not None
    arr = np.ones(64, dtype=np.float32)
    assert t.mem_htod(ptr, arr, arr.nbytes, ctx) == 0
    t.mem_free(ptr, ctx)
    t.destroy_context(ctx)


def test_tpu_all_reduce():
    t = TPUDriver()
    a = np.ones(16, dtype=np.float32)
    b = np.zeros(16, dtype=np.float32)
    assert t.all_reduce(a, b, 0, 0) == 0
    assert np.allclose(b, a)


def test_tpu_compile_execute():
    t = TPUDriver()
    ctx = t.create_context(0)
    exec = t.compile("dummy_hlo", ctx)
    assert exec is not None
    t.executable_destroy(exec)
    t.destroy_context(ctx)


if __name__ == "__main__":
    test_cuda_available()
    test_cuda_device_count()
    test_cuda_device_props()
    test_cuda_set_get_device()
    test_cuda_context()
    test_cuda_stream()
    test_cuda_event()
    test_cuda_memory()
    test_cuda_launch_kernel()
    test_rocm_available()
    test_rocm_lifecycle()
    test_rocm_memory()
    test_tpu_available()
    test_tpu_lifecycle()
    test_tpu_memory()
    test_tpu_all_reduce()
    test_tpu_compile_execute()
    print("All driver_bindings tests passed.")
