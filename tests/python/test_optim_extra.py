"""Tests for sneppx_serve.optim_extra (sneppx-alg interface_bindings.optim_extra)."""

import numpy as np

from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings.optim_extra import (
    Lion,
    LAMB,
    LARS,
    AdaFactor,
    RAdam,
    Sophia,
    Adan,
    ScheduleFreeAdamW,
    Adam8bit,
    SGD8bit,
)


def _make_param():
    p = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    p.grad = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    return p


def _run_step(opt_cls, **kw):
    p = _make_param()
    before = p.data.copy()
    opt = opt_cls([p], **kw)
    opt.step()
    after = p.data
    assert not np.allclose(before, after), "%s did not update params" % opt_cls.__name__
    opt.zero_grad()
    assert p.grad is None
    return True


def test_lion():
    assert _run_step(Lion, lr=0.01)


def test_lamb():
    assert _run_step(LAMB, lr=0.01)


def test_lars():
    assert _run_step(LARS, lr=0.01)


def test_adafactor():
    assert _run_step(AdaFactor, lr=0.01)


def test_radam():
    assert _run_step(RAdam, lr=0.01)


def test_sophia():
    assert _run_step(Sophia, lr=0.01)


def test_adan():
    assert _run_step(Adan, lr=0.01)


def test_schedule_free_adamw():
    p = _make_param()
    before = p.data.copy()
    opt = ScheduleFreeAdamW([p], lr=0.01)
    for _ in range(10):
        p.grad = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
        opt.step()
        opt.zero_grad()
    after = p.data
    assert not np.allclose(before, after), "ScheduleFreeAdamW did not update params"


def test_adam8bit():
    assert _run_step(Adam8bit, lr=0.01)


def test_sgd8bit():
    assert _run_step(SGD8bit, lr=0.01)


def test_multiple_steps():
    p = _make_param()
    opt = Lion([p], lr=0.01)
    for _ in range(3):
        p.grad = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
        opt.step()
        opt.zero_grad()
    assert not np.allclose(p.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
