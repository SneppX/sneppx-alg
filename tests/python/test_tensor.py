"""Comprehensive tests for Tensor operations (pure NumPy or C backend)."""

import numpy as np
from SneppX_ALG.interface_bindings import Tensor, Dtype, Device, _HAS_C_BACKEND
from SneppX_ALG.interface_bindings.tensor import DTYPE_MAP


def test_dtype_and_device_enum_contract():
    expected_dtype_names = (
        "FLOAT32", "FLOAT64", "FLOAT16", "BFLOAT16", "FLOAT8",
        "INT32", "INT64", "INT16", "INT8", "UINT8", "BOOL",
        "COMPLEX64", "COMPLEX128",
    )
    assert all(hasattr(Dtype, name) for name in expected_dtype_names)
    assert all(hasattr(Device, name) for name in ("CPU", "CUDA", "METAL", "VULKAN", "TPU", "NPU"))
    if not _HAS_C_BACKEND:
        assert [getattr(Dtype, name) for name in expected_dtype_names] == list(range(13))
        assert [getattr(Device, name) for name in ("CPU", "CUDA", "METAL", "VULKAN", "TPU", "NPU")] == list(range(6))


def test_complex_scalar_construction():
    value = 2.5 - 3.0j
    tensor = Tensor(value, dtype="complex64")
    assert tensor.shape == (1,)
    assert tensor.data.dtype == np.complex64
    assert np.allclose(tensor.data, np.array([value], dtype=np.complex64))


def test_float8_storage_roundtrip():
    assert DTYPE_MAP["float8"][1] == 1
    values = np.array([0, 1, 127, 255], dtype=np.uint8)
    tensor = Tensor(values, dtype="float8")
    assert tensor.dtype_name == "float8"
    assert tensor.data.dtype == np.uint8
    assert np.array_equal(tensor.data, values)


def test_create():
    t = Tensor([2, 3])
    assert t.shape == (2,), f"Expected (2,), got {t.shape}"
    print("  test_create PASS")


def test_numpy_roundtrip():
    arr = np.random.randn(3, 4).astype(np.float32)
    t = Tensor.from_numpy(arr)
    out = t.data
    assert np.allclose(arr, out), "Roundtrip mismatch"
    print("  test_numpy_roundtrip PASS")


def test_factory_zeros():
    t = Tensor.zeros((2, 3))
    assert t.shape == (2, 3)
    assert np.all(t.data == 0)
    print("  test_factory_zeros PASS")


def test_factory_ones():
    t = Tensor.ones((2, 3))
    assert t.shape == (2, 3)
    assert np.all(t.data == 1)
    print("  test_factory_ones PASS")


def test_factory_eye():
    t = Tensor.eye(3)
    assert t.shape == (3, 3)
    assert np.allclose(t.data, np.eye(3))
    print("  test_factory_eye PASS")


def test_factory_arange():
    t = Tensor.arange(0, 5, 1)
    assert t.shape == (5,)
    assert np.allclose(t.data, [0, 1, 2, 3, 4])
    print("  test_factory_arange PASS")


def test_factory_full():
    t = Tensor.full((2, 3), 3.14)
    assert np.allclose(t.data, 3.14)
    print("  test_factory_full PASS")


def test_factory_randn():
    t = Tensor.randn((1000,))
    assert t.shape == (1000,)
    assert abs(t.data.mean()) < 0.1
    assert abs(t.data.std() - 1.0) < 0.1
    print("  test_factory_randn PASS")


def test_list_construction():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    print("  test_list_construction PASS")


def test_scalar_construction():
    t = Tensor(42.0)
    assert t.shape == (1,)
    assert abs(t.data[0] - 42.0) < 1e-5
    print("  test_scalar_construction PASS")


def test_arithmetic_add():
    a = Tensor.ones((3,))
    b = Tensor.ones((3,))
    c = a + b
    assert np.allclose(c.data, 2.0)
    print("  test_arithmetic_add PASS")


def test_arithmetic_sub():
    a = Tensor.full((3,), 5.0)
    b = Tensor.full((3,), 3.0)
    c = a - b
    assert np.allclose(c.data, 2.0)
    print("  test_arithmetic_sub PASS")


def test_arithmetic_mul():
    a = Tensor.full((3,), 4.0)
    b = Tensor.full((3,), 2.0)
    c = a * b
    assert np.allclose(c.data, 8.0)
    print("  test_arithmetic_mul PASS")


def test_arithmetic_div():
    a = Tensor.full((3,), 10.0)
    b = Tensor.full((3,), 2.0)
    c = a / b
    assert np.allclose(c.data, 5.0)
    print("  test_arithmetic_div PASS")


def test_arithmetic_neg():
    a = Tensor.full((3,), 5.0)
    c = -a
    assert np.allclose(c.data, -5.0)
    print("  test_arithmetic_neg PASS")


def test_matmul():
    a = Tensor.eye(3)
    b = Tensor.full((3,), 2.0).reshape(3, 1)
    c = a @ b
    assert c.shape == (3, 1)
    assert np.allclose(c.data.flatten(), 2.0)
    print("  test_matmul PASS")


def test_activations():
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(x.relu().data, [0, 0, 0, 1, 2])
    assert np.allclose(x.sigmoid().data, 1 / (1 + np.exp(-x.data)))
    assert np.allclose(x.tanh_act().data, np.tanh(x.data))
    assert np.allclose(x.exp().data, np.exp(x.data))
    assert np.allclose(x.abs().data, [2, 1, 0, 1, 2])
    print("  test_activations PASS")


def test_softmax():
    x = Tensor([[1.0, 2.0, 3.0]])
    sm = x.softmax(-1)
    assert np.allclose(sm.data.sum(axis=1), [1.0])
    print("  test_softmax PASS")


def test_sum_mean():
    x = Tensor.ones((2, 3))
    assert np.allclose(x.sum(0).data, [2, 2, 2])
    assert np.allclose(x.mean(0).data, [1, 1, 1])
    print("  test_sum_mean PASS")


def test_transpose():
    x = Tensor.ones((2, 3))
    xt = x.transpose(0, 1)
    assert xt.shape == (3, 2)
    print("  test_transpose PASS")


def test_reshape():
    x = Tensor.ones((12,))
    y = x.reshape(3, 4)
    assert y.shape == (3, 4)
    print("  test_reshape PASS")


def test_clone():
    x = Tensor.ones((3,))
    y = x.clone()
    assert np.allclose(y.data, 1.0)
    y.data[0] = 99
    assert x.data[0] == 1.0
    print("  test_clone PASS")


def test_item():
    x = Tensor([3.14])
    assert abs(x.item() - 3.14) < 1e-5
    print("  test_item PASS")


def test_getitem():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    assert x[0].data[0] == 1
    row = x[0]
    assert row.shape == (3,), f"Expected (3,), got {row.shape}"
    val = x[1, 2]
    assert abs(val.item() - 6) < 1e-5, f"Expected 6, got {val.item()}"
    print("  test_getitem PASS")


def test_fill():
    x = Tensor.zeros((3,))
    x.fill_(5.0)
    assert np.allclose(x.data, 5.0)
    print("  test_fill PASS")


def test_data_setter():
    x = Tensor.zeros((3,))
    x.data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(x.data, [1.0, 2.0, 3.0])
    print("  test_data_setter PASS")


def test_losses():
    pred = Tensor.ones((3,))
    target = Tensor.zeros((3,))
    mse = pred.mse_loss(target)
    assert mse.item() > 0
    mae = pred.mae_loss(target)
    assert mae.item() > 0
    print("  test_losses PASS")


def test_broadcasting():
    a = Tensor.ones((3, 4))
    b = Tensor.ones((4,))
    c = a + b
    assert c.shape == (3, 4)
    print("  test_broadcasting PASS")


if __name__ == "__main__":
    test_create()
    test_list_construction()
    test_scalar_construction()
    test_numpy_roundtrip()
    test_factory_zeros()
    test_factory_ones()
    test_factory_eye()
    test_factory_arange()
    test_factory_full()
    test_factory_randn()
    test_arithmetic_add()
    test_arithmetic_sub()
    test_arithmetic_mul()
    test_arithmetic_div()
    test_arithmetic_neg()
    test_matmul()
    test_activations()
    test_softmax()
    test_sum_mean()
    test_transpose()
    test_reshape()
    test_clone()
    test_item()
    test_getitem()
    test_fill()
    test_data_setter()
    test_losses()
    test_broadcasting()
    print("\nAll tensor tests passed!")
