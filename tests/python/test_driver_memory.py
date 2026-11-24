"""Regression tests for Python driver device-memory ownership."""

import gc

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.driver_bindings import (
    CUDADriver,
    ROCmDriver,
    TPUDriver,
)


@pytest.mark.parametrize("driver_factory", [CUDADriver, ROCmDriver])
def test_host_copy_round_trip(driver_factory):
    driver = driver_factory()
    ptr = driver.mem_alloc(64)
    gc.collect()  # the backing buffer must stay alive after collection
    src = np.arange(16, dtype=np.float32)
    assert driver.mem_htod(ptr, src, src.nbytes) == 0
    out = np.zeros_like(src)
    assert driver.mem_dtoh(out, ptr, out.nbytes) == 0
    np.testing.assert_array_equal(out, src)
    driver.mem_free(ptr)


def test_tpu_host_copy_round_trip():
    driver = TPUDriver()
    ctx = driver.create_context(0)
    ptr = driver.mem_alloc(64, ctx)
    gc.collect()
    src = np.arange(16, dtype=np.float32)
    assert driver.mem_htod(ptr, src, src.nbytes, ctx) == 0
    out = np.zeros_like(src)
    assert driver.mem_dtoh(out, ptr, out.nbytes, ctx) == 0
    np.testing.assert_array_equal(out, src)
    driver.mem_free(ptr, ctx)
    driver.destroy_context(ctx)


def test_cuda_device_to_device_copy_and_set():
    driver = CUDADriver()
    src = driver.mem_alloc(16)
    dst = driver.mem_alloc(16)
    payload = np.arange(4, dtype=np.int32)
    assert driver.mem_htod(src, payload, payload.nbytes) == 0
    assert driver.mem_dtod(dst, src, payload.nbytes) == 0
    out = np.zeros_like(payload)
    assert driver.mem_dtoh(out, dst, out.nbytes) == 0
    np.testing.assert_array_equal(out, payload)
    assert driver.mem_set(dst, 7, 16) == 0
    assert driver.mem_dtoh(out, dst, out.nbytes) == 0
    np.testing.assert_array_equal(out, np.full(4, 0x07070707, dtype=np.int32))
    driver.mem_free(src)
    driver.mem_free(dst)


def test_free_releases_pointer():
    driver = CUDADriver()
    ptr = driver.mem_alloc(32)
    driver.mem_free(ptr)
    with pytest.raises(ValueError, match="invalid or freed device pointer"):
        driver.mem_htod(ptr, np.zeros(8, dtype=np.float32), 32)
    # A second free is harmless and must not release unrelated memory.
    driver.mem_free(ptr)


def test_allocation_validation():
    driver = CUDADriver()
    with pytest.raises(ValueError, match="non-negative"):
        driver.mem_alloc(-1)
    with pytest.raises(TypeError, match="integer"):
        driver.mem_alloc(1.5)


def test_allocation_sizes_are_checked():
    driver = CUDADriver()
    ptr = driver.mem_alloc(4)
    assert driver.mem_htod(ptr, np.zeros(8, dtype=np.float32), 8) == -1
    assert driver.mem_dtoh(np.zeros(8, dtype=np.float32), ptr, 8) == -1
    driver.mem_free(ptr)


def test_live_pointers_are_unique():
    driver = CUDADriver()
    first = driver.mem_alloc(8)
    second = driver.mem_alloc(8)
    assert first != second
    driver.mem_free(first)
    driver.mem_free(second)
