"""Tests for optim.py — pure Python optimizers."""

import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    SGD,
    AdamW,
    Lion,
    LAMB,
    CosineAnnealingLR,
)


def test_sgd_step():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = SGD(params, lr=0.01)
    x = Tensor.randn((2, 4))
    out = linear(x)
    loss = out.sum()
    for p in params:
        p.grad = Tensor.ones(p.shape)
    opt.step()
    print("  test_sgd_step PASS")


def test_adamw_step():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = AdamW(params, lr=0.001)
    for p in params:
        p.grad = Tensor.ones(p.shape)
    opt.step()
    print("  test_adamw_step PASS")


def test_lion_step():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = Lion(params, lr=0.0001)
    for p in params:
        p.grad = Tensor.ones(p.shape)
    opt.step()
    print("  test_lion_step PASS")


def test_lamb_step():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = LAMB(params, lr=0.001)
    for p in params:
        p.grad = Tensor.ones(p.shape)
    opt.step()
    print("  test_lamb_step PASS")


def test_zero_grad():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = SGD(params, lr=0.01)
    for p in params:
        p.grad = Tensor.randn(p.shape)
    opt.zero_grad()
    for p in params:
        if p.grad is not None:
            assert np.all(p.grad.data == 0), "grad not zeroed"
    print("  test_zero_grad PASS")


def test_cosine_annealing_lr():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = SGD(params, lr=0.1)
    scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    initial_lr = opt.lr
    for _ in range(5):
        scheduler.step()
    assert opt.lr < initial_lr, "LR should decrease"
    for _ in range(5):
        scheduler.step()
    assert opt.lr < 0.01, "LR should be near 0 after full cycle"
    print("  test_cosine_annealing_lr PASS")


def test_optimizer_parameters_tracked():
    linear = Linear(4, 3)
    params = linear.parameters()
    opt = AdamW(params, lr=0.001)
    assert len(opt.params) == 2
    print("  test_optimizer_parameters_tracked PASS")


if __name__ == "__main__":
    test_sgd_step()
    test_adamw_step()
    test_lion_step()
    test_lamb_step()
    test_zero_grad()
    test_cosine_annealing_lr()
    test_optimizer_parameters_tracked()
    print("\nAll optim tests passed!")
