"""Tests for algo_ser.py — SER algorithm bindings."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import SERExpert, SERRoute, SERLayer, SERLoss
from SneppX_ALG.interface_bindings import AlgoSERModel as SERModel


def test_expert_forward():
    e = SERExpert(0, input_dim=8, hidden_dim=16)
    x = np.random.randn(3, 8).astype(np.float32)
    out = e.forward(x)
    assert out.shape == x.shape


def test_route_topk_gating():
    route = SERRoute(n_experts=4, k=2)
    x = np.random.randn(5, 8).astype(np.float32)
    w = np.random.randn(4, 8).astype(np.float32)
    indices, weights = route.topk_gating(x, w)
    assert indices.shape == (5, 2)
    assert weights.shape == (5, 2)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)


def test_route_load_balance():
    route = SERRoute(n_experts=4, k=2)
    scores = np.random.randn(10, 4).astype(np.float32)
    indices = np.random.randint(0, 4, (10, 2))
    loss = route.load_balance_loss(scores, indices)
    assert loss >= 0.0


def test_layer_forward():
    layer = SERLayer(n_experts=4, input_dim=8, hidden_dim=16, k=2)
    x = np.random.randn(2, 3, 8).astype(np.float32)
    w = np.random.randn(4, 8).astype(np.float32)
    expert_outs = [np.random.randn(8) for _ in range(4)]
    out = layer.forward(x, w, expert_outs)
    assert out.shape == (2, 3, 8)


def test_loss_load_balancing():
    loss = SERLoss()
    gate = np.random.randn(10, 4).astype(np.float32)
    idx = np.random.randint(0, 4, (10, 2))
    lb = loss.load_balancing(gate, idx, 4)
    assert lb >= 0.0


def test_loss_z_loss():
    loss = SERLoss()
    logits = np.random.randn(5, 8).astype(np.float32)
    zl = loss.z_loss(logits, z_weight=0.001)
    assert zl >= 0.0


def test_model_forward():
    m = SERModel(n_layers=2, n_experts=4, d_model=16, k=2)
    x = np.random.randn(2, 4, 16).astype(np.float32)
    out = m.forward(x)
    assert out.shape == (2, 4, 16)


if __name__ == "__main__":
    test_expert_forward()
    test_route_topk_gating()
    test_route_load_balance()
    test_layer_forward()
    test_loss_load_balancing()
    test_loss_z_loss()
    test_model_forward()
    print("All algo_ser tests passed.")
