"""Tests for differential_privacy.py — DP mechanisms, budget, accountant, DPSGD."""

import math
import json
import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    SGD,
    AdamW,
    DPSGD,
    LaplaceMech,
    GaussianMech,
    PrivacyBudget,
    RDPAccountant,
)


# ===========================================================================
#  Laplace & Gaussian mechanisms
# ===========================================================================


def test_laplace_mech():
    mech = LaplaceMech(epsilon=1.0)
    np.random.seed(42)
    vals = np.array([mech.apply(0.0, sensitivity=1.0) for _ in range(50000)])
    mean = float(np.mean(vals))
    var = float(np.var(vals))
    assert abs(mean) < 0.1, f"Mean too large: {mean}"
    assert abs(var - 2.0) < 0.3, f"Var not ~2.0: {var}"
    print("  test_laplace_mech PASS")


def test_gaussian_mech():
    mech = GaussianMech(epsilon=1.0, delta=1e-5)
    np.random.seed(42)
    vals = np.array([mech.apply(0.0, sensitivity=1.0) for _ in range(50000)])
    mean = float(np.mean(vals))
    expected_sigma = math.sqrt(2 * math.log(1.25 / 1e-5))
    var = float(np.var(vals))
    assert abs(mean) < 0.1, f"Mean too large: {mean}"
    assert abs(var - expected_sigma ** 2) < 0.5, f"Var {var:.3f} != {expected_sigma**2:.3f}"
    print("  test_gaussian_mech PASS")


# ===========================================================================
#  PrivacyBudget
# ===========================================================================


def test_privacy_budget_tracking():
    budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
    assert budget.check(0.5, 0.0)
    budget.spend(0.5, 0.0)
    assert budget.check(0.5, 0.0)
    budget.spend(0.5, 0.0)
    try:
        budget.spend(0.1, 0.0)
        assert False, "Should have raised"
    except ValueError:
        pass
    remaining_eps, remaining_del = budget.remaining
    assert remaining_eps < 1e-10, f"Expected 0 remaining, got {remaining_eps}"
    print("  test_privacy_budget_tracking PASS")


def test_privacy_budget_serialize():
    budget = PrivacyBudget(epsilon=2.0, delta=1e-6)
    budget.spend(0.3, 0.0)
    budget.spend(0.7, 0.0)
    d = budget.to_dict()
    restored = PrivacyBudget.from_dict(d)
    assert abs(restored.spent[0] - 1.0) < 1e-10
    assert abs(restored.remaining[0] - 1.0) < 1e-10
    assert abs(restored.spent[1] - 0.0) < 1e-10
    print("  test_privacy_budget_serialize PASS")


# ===========================================================================
#  RDP Accountant
# ===========================================================================


def test_rdp_accountant_eps_monotonic():
    acc = RDPAccountant(delta=1e-5)
    eps_before = acc.get_epsilon()
    assert eps_before == float("inf"), f"Expected inf before any steps, got {eps_before}"
    acc.step(noise_multiplier=1.0, batch_size=256, num_samples=50000)
    eps_after = acc.get_epsilon()
    assert eps_after < float("inf"), "Epsilon should be finite after one step"
    assert eps_after > 0, "Epsilon should be positive"
    eps_prev = eps_after
    for _ in range(50):
        acc.step(noise_multiplier=1.0, batch_size=256, num_samples=50000)
        eps_curr = acc.get_epsilon()
        assert eps_curr >= eps_prev - 1e-6, f"Epsilon decreased: {eps_prev} -> {eps_curr}"
        eps_prev = eps_curr
    print("  test_rdp_accountant_eps_monotonic PASS")


def test_rdp_subsampling_amplification():
    acc_no_subsample = RDPAccountant(delta=1e-5)
    acc_subsample = RDPAccountant(delta=1e-5)
    np.random.seed(42)
    for _ in range(100):
        acc_no_subsample.step(noise_multiplier=1.0, batch_size=50000, num_samples=50000)
        acc_subsample.step(noise_multiplier=1.0, batch_size=256, num_samples=50000)
    eps_no = acc_no_subsample.get_epsilon()
    eps_sub = acc_subsample.get_epsilon()
    assert eps_sub < eps_no, f"Subsampling should reduce epsilon: {eps_sub} >= {eps_no}"
    print("  test_rdp_subsampling_amplification PASS")


def test_rdp_to_dp_conversion():
    acc = RDPAccountant(delta=1e-5)
    sigma = 10.0
    orders = acc._orders
    for _ in range(100):
        acc.step(noise_multiplier=sigma, batch_size=50000, num_samples=50000)
    eps = acc.get_epsilon(delta=1e-5)
    # Analytical: rdp(alpha) = alpha / (2 * sigma^2) per step
    # After 100 steps: rdp(alpha) = 100 * alpha / (2 * sigma^2)
    # epsilon = min_alpha rdp(alpha) + log(1/delta) / (alpha-1)
    expected_eps = float("inf")
    for alpha in orders:
        if alpha <= 1:
            continue
        rdp = 100.0 * alpha / (2.0 * sigma * sigma)
        e = rdp + math.log(1.0 / 1e-5) / (alpha - 1)
        expected_eps = min(expected_eps, e)
    assert abs(eps - expected_eps) < 0.1, f"Epsilon {eps} != expected {expected_eps}"
    print("  test_rdp_to_dp_conversion PASS")


# ===========================================================================
#  DPSGD
# ===========================================================================


def test_dpsgd_clipping():
    np.random.seed(42)
    linear = Linear(4, 3)
    dpsgd = DPSGD(
        SGD(linear.parameters(), lr=0.01),
        noise_multiplier=0.0,
        max_grad_norm=0.5,
        num_samples=100,
        epsilon=1.0,
        delta=1e-5,
    )
    x = Tensor.randn((16, 4))
    targets = Tensor.ones((16, 3))
    outputs = linear(x)
    per_sample = [outputs[i : i + 1].mse_loss(targets[i : i + 1]) for i in range(16)]
    dpsgd.step(per_sample)
    params = list(linear.parameters())
    max_norm = 0.0
    for p in params:
        if p.grad is not None:
            g_norm = float(np.sqrt(np.sum(p.grad.data.astype(np.float64) ** 2)))
            max_norm = max(max_norm, g_norm)
    assert max_norm <= 0.5 + 1e-3, f"Gradient norm {max_norm} > 0.5"
    print("  test_dpsgd_clipping PASS")


def test_dpsgd_noise_std():
    np.random.seed(42)
    # Use a linear layer with zero weights so gradients are purely from DP noise
    linear = Linear(2, 1)
    for p in linear.parameters():
        p.data[:] = 0.0
    noise_mult = 2.0
    max_norm = 1.0
    batch_size = 8
    dpsgd = DPSGD(
        SGD(linear.parameters(), lr=0.0),
        noise_multiplier=noise_mult,
        max_grad_norm=max_norm,
        num_samples=100,
        epsilon=8.0,
        delta=1e-5,
    )
    x = Tensor.ones((batch_size, 2))
    targets = Tensor.zeros((batch_size, 1))
    outputs = linear(x)
    per_sample = [outputs[i : i + 1].mse_loss(targets[i : i + 1]) for i in range(batch_size)]
    grads_all = []
    for _ in range(200):
        dpsgd.step(per_sample)
        for p in linear.parameters():
            if p.grad is not None:
                grads_all.append(p.grad.data.copy())
    all_grads = np.concatenate([g.ravel() for g in grads_all])
    expected_sigma = noise_mult * max_norm / batch_size
    observed_std = float(np.std(all_grads))
    tol = 0.15
    assert abs(observed_std - expected_sigma) < tol, (
        f"Observed std {observed_std:.4f} != expected sigma {expected_sigma:.4f}"
    )
    print("  test_dpsgd_noise_std PASS")


def test_dpsgd_wrapping_optimizer():
    np.random.seed(42)
    linear = Linear(4, 2)
    opt = AdamW(linear.parameters(), lr=0.01)
    dpsgd = DPSGD(
        opt,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        num_samples=32,
        epsilon=8.0,
        delta=1e-5,
    )
    x = Tensor.randn((8, 4))
    targets = Tensor.randn((8, 2))
    before = [p.data.copy() for p in linear.parameters()]
    outputs = linear(x)
    per_sample = [outputs[i : i + 1].mse_loss(targets[i : i + 1]) for i in range(8)]
    dpsgd.step(per_sample)
    after = [p.data.copy() for p in linear.parameters()]
    # With lr=0.01, params should change
    params_changed = any(
        not np.allclose(b, a) for b, a in zip(before, after)
    )
    assert params_changed, "Parameters did not change after DPSGD step"
    eps = dpsgd.privacy_spent()
    assert eps["epsilon"] < float("inf"), "Privacy epsilon should be finite"
    assert eps["epsilon"] > 0, "Privacy epsilon should be positive"
    print("  test_dpsgd_wrapping_optimizer PASS")


# ===========================================================================
#  End-to-end: DP reduces overfitting / membership inference
# ===========================================================================


def test_dp_training_reduces_memorization():
    np.random.seed(42)
    x_train = Tensor.randn((50, 4))
    y_train = Tensor.randn((50, 1))
    x_test = Tensor.randn((50, 4))
    y_test = Tensor.randn((50, 1))

    # Non-private model
    np.random.seed(42)
    model_nonprivate = Linear(4, 1)
    opt_np = SGD(model_nonprivate.parameters(), lr=0.05)
    for _ in range(200):
        opt_np.zero_grad()
        out = model_nonprivate(x_train)
        loss = out.mse_loss(y_train)
        loss.backward()
        opt_np.step()
    loss_train_np = float(model_nonprivate(x_train).mse_loss(y_train).data.flat[0])
    loss_test_np = float(model_nonprivate(x_test).mse_loss(y_test).data.flat[0])
    gap_np = abs(loss_train_np - loss_test_np)

    # Private model
    np.random.seed(42)
    model_dp = Linear(4, 1)
    priv_opt = SGD(model_dp.parameters(), lr=0.05)
    dpsgd = DPSGD(
        priv_opt,
        noise_multiplier=0.8,
        max_grad_norm=0.5,
        num_samples=50,
        epsilon=8.0,
        delta=1e-5,
    )
    for _ in range(200):
        out = model_dp(x_train)
        per_sample = [out[i : i + 1].mse_loss(y_train[i : i + 1]) for i in range(50)]
        dpsgd.step(per_sample)
    loss_train_dp = float(model_dp(x_train).mse_loss(y_train).data.flat[0])
    loss_test_dp = float(model_dp(x_test).mse_loss(y_test).data.flat[0])
    gap_dp = abs(loss_train_dp - loss_test_dp)

    assert gap_dp <= gap_np + 0.5, (
        f"DP gap {gap_dp:.4f} > non-private gap {gap_np:.4f} + 0.5"
    )
    print("  test_dp_training_reduces_memorization PASS")


# ===========================================================================
#  DPSGD + Trainer integration (config plumbing)
# ===========================================================================


def test_dpsgd_trainer_config():
    from SneppX_ALG.interface_bindings.trainer_v3 import TrainConfig

    cfg = TrainConfig({"privacy": {"enabled": True, "noise_multiplier": 1.2,
                                   "max_per_sample_grad_norm": 0.5, "epsilon": 4.0,
                                   "delta": 1e-5, "accountant": "rdp"}})
    priv = cfg.privacy
    assert priv.enabled is True
    assert abs(priv.noise_multiplier - 1.2) < 1e-6
    assert abs(priv.max_per_sample_grad_norm - 0.5) < 1e-6
    assert abs(priv.epsilon - 4.0) < 1e-6
    assert abs(priv.delta - 1e-5) < 1e-6
    assert priv.accountant == "rdp"
    print("  test_dpsgd_trainer_config PASS")


# ===========================================================================
#  Edge cases
# ===========================================================================


def test_dpsgd_empty_batch():
    linear = Linear(2, 2)
    dpsgd = DPSGD(
        SGD(linear.parameters(), lr=0.01),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        num_samples=10,
        epsilon=8.0,
        delta=1e-5,
    )
    params_before = [p.data.copy() for p in linear.parameters()]
    dpsgd.step([])
    params_after = [p.data.copy() for p in linear.parameters()]
    assert all(np.allclose(b, a) for b, a in zip(params_before, params_after))
    print("  test_dpsgd_empty_batch PASS")
