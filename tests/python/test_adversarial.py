"""Tests for adversarial.py — FGSM/PGD attacks and adversarial training."""

import numpy as np
import pytest

from SneppX_ALG.interface_bindings import Tensor, Module, AdamW
from SneppX_ALG.interface_bindings.adversarial import (
    Attack,
    FGSM,
    PGD,
    AdversarialExample,
    AdversarialTrainer,
)


class _LinearModel(Module):
    def __init__(self, nin=4, nout=3, seed=0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.w = Tensor(rng.randn(nin, nout).astype(np.float32), requires_grad=True)
        self.b = Tensor(np.zeros(nout, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        return x @ self.w + self.b


def _make_loss():
    def loss_fn(outputs, labels):
        logits = outputs if isinstance(outputs, Tensor) else Tensor(outputs)
        return logits.cross_entropy(labels)

    return loss_fn


def _batch(n=8, nin=4, nout=3, seed=1):
    rng = np.random.RandomState(seed)
    X = Tensor(rng.randn(n, nin).astype(np.float32))
    y = Tensor(rng.randint(0, nout, size=n).astype(np.int64))
    return X, y


# ===========================================================================
#  FGSM
# ===========================================================================


def test_fgsm_changes_input():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    atk = FGSM(epsilon=0.3)
    adv = atk.generate(m, X, y, loss_fn)
    assert adv.shape == X.shape
    assert not np.allclose(adv.data, X.data)
    assert adv.requires_grad is False


def test_fgsm_respects_epsilon_ball():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    eps = 0.1
    atk = FGSM(epsilon=eps)
    adv = atk.generate(m, X, y, loss_fn)
    diff = np.abs(adv.data - X.data)
    assert diff.max() <= eps + 1e-5


def test_fgsm_clamps_to_bounds():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    atk = FGSM(epsilon=2.0, clip_min=0.0, clip_max=1.0)
    adv = atk.generate(m, X, y, loss_fn)
    assert adv.data.min() >= -1e-6
    assert adv.data.max() <= 1.0 + 1e-6


def test_fgsm_targeted_flips_sign():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    clean = FGSM(epsilon=0.1).generate(m, X, y, loss_fn)
    targeted = FGSM(epsilon=0.1, targeted=True).generate(m, X, y, loss_fn)
    # Targeted perturbation is opposite direction from untargeted.
    assert not np.allclose(clean.data, targeted.data, atol=1e-4)


# ===========================================================================
#  PGD
# ===========================================================================


def test_pgd_changes_input():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    atk = PGD(epsilon=0.3, alpha=0.05, steps=10)
    adv = atk.generate(m, X, y, loss_fn)
    assert adv.shape == X.shape
    assert not np.allclose(adv.data, X.data)
    assert adv.requires_grad is False


def test_pgd_respects_epsilon_ball():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    eps = 0.2
    atk = PGD(epsilon=eps, alpha=0.05, steps=10)
    adv = atk.generate(m, X, y, loss_fn)
    diff = np.abs(adv.data - X.data)
    assert diff.max() <= eps + 1e-5


def test_pgd_zero_steps_raises():
    with pytest.raises(ValueError):
        PGD(epsilon=0.3, alpha=0.05, steps=0)


def test_pgd_nonpositive_alpha_raises():
    with pytest.raises(ValueError):
        PGD(epsilon=0.3, alpha=0.0, steps=5)


def test_pgd_more_perturbation_than_fgsm():
    m = _LinearModel()
    loss_fn = _make_loss()
    X, y = _batch()
    eps = 0.1
    fgsm_adv = FGSM(epsilon=eps).generate(m, X, y, loss_fn)
    pgd_adv = PGD(epsilon=eps, alpha=0.02, steps=20).generate(m, X, y, loss_fn)
    # PGD (iterative) generally pushes further within the ball than one FGSM step.
    fgsm_loss = float(m(fgsm_adv).cross_entropy(y).data.item())
    pgd_loss = float(m(pgd_adv).cross_entropy(y).data.item())
    assert pgd_loss >= fgsm_loss - 1e-4


# ===========================================================================
#  AdversarialTrainer
# ===========================================================================


def test_trainer_step_runs():
    m = _LinearModel()
    loss_fn = _make_loss()
    opt = AdamW(m.parameters(), lr=0.01)
    atk = PGD(epsilon=0.1, alpha=0.02, steps=4)
    tr = AdversarialTrainer(m, opt, atk, loss_fn, attack_ratio=1.0)
    X, y = _batch()
    metrics = tr.train_step((X, y))
    assert "loss" in metrics
    assert metrics["adversarial"] is True
    assert metrics["step"] == 0


def test_trainer_attack_ratio_zero_uses_clean():
    m = _LinearModel()
    loss_fn = _make_loss()
    opt = AdamW(m.parameters(), lr=0.01)
    atk = PGD(epsilon=0.1, alpha=0.02, steps=4)
    tr = AdversarialTrainer(m, opt, atk, loss_fn, attack_ratio=0.0)
    X, y = _batch()
    metrics = tr.train_step((X, y))
    assert metrics["adversarial"] is False


def test_trainer_fit_reduces_loss():
    m = _LinearModel(seed=2)
    loss_fn = _make_loss()
    opt = AdamW(m.parameters(), lr=0.05)
    atk = FGSM(epsilon=0.05)
    tr = AdversarialTrainer(m, opt, atk, loss_fn, attack_ratio=1.0)
    X, y = _batch(seed=2)
    losses = [tr.train_step((X, y))["loss"] for _ in range(8)]
    assert losses[-1] < losses[0]


def test_trainer_robustness_eval():
    m = _LinearModel()
    loss_fn = _make_loss()
    opt = AdamW(m.parameters(), lr=0.01)
    atk = PGD(epsilon=0.1, alpha=0.02, steps=4)
    tr = AdversarialTrainer(m, opt, atk, loss_fn, attack_ratio=1.0)
    X, y = _batch()
    res = tr.evaluate_robustness([(X, y)])
    assert 0.0 <= res["clean_acc"] <= 1.0
    assert 0.0 <= res["robust_acc"] <= 1.0
    assert res["total"] == 8


def test_adversarial_example_container():
    X, y = _batch()
    adv = FGSM(epsilon=0.1).generate(_LinearModel(), X, y, _make_loss())
    ex = AdversarialExample(X, adv, epsilon=0.1)
    c, a = ex.numpy()
    assert c.shape == a.shape
    assert ex.epsilon == 0.1


def test_attack_negative_epsilon_raises():
    with pytest.raises(ValueError):
        FGSM(epsilon=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
