import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import SneppX_ALG.interface_bindings.init as init
from SneppX_ALG.interface_bindings.tensor import Tensor


def _stats(t):
    a = t.data
    return a.mean(), a.std(), a.min(), a.max()


def test_basic_fillers():
    t = Tensor(np.zeros((10, 10), dtype="float32"))
    init.ones_(t)
    assert np.allclose(t.data, 1.0)
    init.zeros_(t)
    assert np.allclose(t.data, 0.0)
    init.constant_(t, 3.5)
    assert np.allclose(t.data, 3.5)


def test_eye_():
    t = Tensor(np.zeros((4, 5), dtype="float32"))
    init.eye_(t)
    assert np.allclose(t.data, np.eye(4, 5))
    try:
        init.eye_(Tensor(np.zeros((3,), dtype="float32")))
        assert False, "eye_ on 1D should raise"
    except ValueError:
        pass


def test_uniform_bounds():
    t = Tensor(np.zeros((20000, 1), dtype="float32"))
    init.uniform_(t, -2.0, 2.0)
    _, _, lo, hi = _stats(t)
    assert lo >= -2.0 - 1e-5 and hi <= 2.0 + 1e-5


def test_normal_mean_std():
    t = Tensor(np.zeros((50000,), dtype="float32"))
    init.normal_(t, 1.0, 0.5)
    mean, std, _, _ = _stats(t)
    assert abs(mean - 1.0) < 0.05
    assert abs(std - 0.5) < 0.05


def test_xavier_uniform_variance():
    # For a square weight the uniform bound is gain*sqrt(3/(fan_in+fan_out));
    # check the realised std ~ gain/sqrt(fan_in+fan_out).
    t = Tensor(np.zeros((100, 100), dtype="float32"))
    init.xavier_uniform_(t, gain=1.0)
    _, std, _, _ = _stats(t)
    expected_std = math.sqrt(1.0 / (100 + 100))  # gain*sqrt(2/(fan_in+fan_out)) for normal-equiv
    # uniform(-b,b) has std = b/sqrt(3); b = sqrt(3/(fan_in+fan_out))
    expected_std = math.sqrt(3.0 / (100 + 100)) / math.sqrt(3.0)
    assert abs(std - expected_std) < 0.02


def test_xavier_normal_variance():
    t = Tensor(np.zeros((200, 200), dtype="float32"))
    init.xavier_normal_(t, gain=1.0)
    _, std, _, _ = _stats(t)
    expected_std = math.sqrt(2.0 / (200 + 200))
    assert abs(std - expected_std) < 0.02


def test_kaiming_uniform_fan_in():
    t = Tensor(np.zeros((64, 128), dtype="float32"))
    init.kaiming_uniform_(t, nonlinearity="relu")
    fan_in = 128
    gain = init.calculate_gain("relu")
    bound = gain * math.sqrt(3.0 / fan_in)
    _, std, _, _ = _stats(t)
    expected_std = bound / math.sqrt(3.0)
    assert abs(std - expected_std) < 0.02


def test_kaiming_normal_fan_out():
    t = Tensor(np.zeros((64, 128), dtype="float32"))
    init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu")
    fan_out = 64
    gain = init.calculate_gain("relu")
    expected_std = gain / math.sqrt(fan_out)
    _, std, _, _ = _stats(t)
    assert abs(std - expected_std) < 0.02


def test_calculate_gain():
    assert abs(init.calculate_gain("relu") - math.sqrt(2.0)) < 1e-6
    assert abs(init.calculate_gain("tanh") - 5.0 / 3.0) < 1e-6
    assert abs(init.calculate_gain("leaky_relu", 0.1) - math.sqrt(2.0 / (1.0 + 0.01))) < 1e-6


def test_orthogonal():
    t = Tensor(np.zeros((8, 8), dtype="float32"))
    init.orthogonal_(t)
    m = t.data
    # check orthogonality: W @ W.T ~ I
    prod = m @ m.T
    assert np.allclose(prod, np.eye(8), atol=1e-4)


import math

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL INIT TESTS PASSED")
