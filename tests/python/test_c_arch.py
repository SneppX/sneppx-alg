"""Tests for c_arch.py — architecture kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import ArchOps, Mamba2


def test_arch_activation_silu():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    out = ArchOps.activation(x, "silu")
    assert out.shape == x.shape
    assert np.all(out >= -1)


def test_arch_activation_gelu():
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out = ArchOps.activation(x, "gelu")
    assert out.shape == x.shape


def test_arch_activation_relu():
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    out = ArchOps.activation(x, "relu")
    assert np.allclose(out, [0.0, 0.0, 2.0])


def test_arch_activation_tanh():
    x = np.array([0.0, 1.0], dtype=np.float32)
    out = ArchOps.activation(x, "tanh")
    assert np.allclose(out, np.tanh(x))


def test_arch_layer_norm():
    x = np.random.randn(4, 32).astype(np.float32)
    w = np.ones(32, dtype=np.float32)
    b = np.zeros(32, dtype=np.float32)
    out = ArchOps.layer_norm(x, w, b)
    assert out.shape == (4, 32)
    assert np.allclose(np.mean(out, axis=-1), 0, atol=1e-4)
    assert np.allclose(np.std(out, axis=-1), 1, atol=1e-4)


def test_mamba2_init():
    m = Mamba2(d_model=16, d_state=8)
    assert m.d_model == 16
    assert m.d_state == 8


def test_mamba2_hippo():
    A = Mamba2.hippo_init(4)
    assert A.shape == (4, 4)
    assert np.all(A <= 0)  # non-positive
    assert np.allclose(np.diag(A), -np.arange(1, 5, dtype=np.float32))


def test_mamba2_forward():
    m = Mamba2(d_model=8, d_state=4)
    batch, seq = 2, 6
    x = np.random.randn(batch, seq, m.d_model).astype(np.float32)
    dt = np.random.randn(batch, seq, 1).astype(np.float32)
    A = np.eye(m.d_state, dtype=np.float32) * -0.1
    B = np.ones((m.d_state, 1), dtype=np.float32)
    C = np.ones((batch, seq, m.d_state), dtype=np.float32)
    out = m.forward(x, dt, A, B, C)
    assert out.shape == (batch, seq, m.d_model)


if __name__ == "__main__":
    test_arch_activation_silu()
    test_arch_activation_gelu()
    test_arch_activation_relu()
    test_arch_activation_tanh()
    test_arch_layer_norm()
    test_mamba2_init()
    test_mamba2_hippo()
    test_mamba2_forward()
    print("All c_arch tests passed.")
