"""Tests for algo_hss.py — HSS algorithm bindings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import HSSDiscretize, HSSScan, HSSStep
from SneppX_ALG.interface_bindings import AlgoHSSModel as HSSModel


def test_discretize_zoh():
    d = HSSDiscretize()
    A = np.eye(4).astype(np.float32)
    B = np.ones((4, 1), dtype=np.float32)
    delta = np.array([0.1], dtype=np.float32)
    Ad, Bd = d.zoh(A, B, delta)
    assert Ad.shape == (4, 4)
    assert Bd.shape == (4, 1)


def test_discretize_bilinear():
    d = HSSDiscretize()
    A = np.eye(4).astype(np.float32) * -0.5
    B = np.ones((4, 1), dtype=np.float32)
    delta = np.array([0.1], dtype=np.float32)
    Ad, Bd = d.bilinear(A, B, delta)
    assert Ad.shape == (4, 4)
    assert Bd.shape == (4, 1)


def test_scan_parallel():
    s = HSSScan()
    batch, seq, d = 2, 8, 4
    x = np.random.randn(batch, seq, d).astype(np.float32)
    A_bar = np.random.randn(batch, seq, d).astype(np.float32)
    B_bar_x = np.random.randn(batch, seq, d).astype(np.float32)
    out = s.parallel_scan(x, A_bar, B_bar_x)
    assert out.shape == (batch, seq, d)


def test_model_forward():
    m = HSSModel(d_model=8, n_layers=2)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    out = m.forward(x)
    assert out.shape == (2, 4, 8)


def test_model_default():
    m = HSSModel()
    assert m.d_model == 64
    assert m.n_layers == 2


def test_step():
    s = HSSStep()
    batch, d = 2, 4
    h = np.zeros((batch, d), dtype=np.float32)
    x_t = np.random.randn(batch, d).astype(np.float32)
    A_bar = np.ones((batch, d), dtype=np.float32) * 0.9
    B_bar = np.ones((batch, d), dtype=np.float32) * 0.5
    C = np.ones((batch, d), dtype=np.float32)
    h_new, y_t = s.step(h, x_t, A_bar, B_bar, C)
    assert h_new.shape == (batch, d)
    assert y_t.shape == (batch,)


if __name__ == "__main__":
    test_discretize_zoh()
    test_discretize_bilinear()
    test_scan_parallel()
    test_model_forward()
    test_model_default()
    test_step()
    print("All algo_hss tests passed.")
