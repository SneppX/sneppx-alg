"""Tests for sneppx_serve.optim_advanced (sneppx-alg interface_bindings.optim_advanced)."""

import numpy as np

from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings.optim_advanced import (
    SM3,
    Demeter,
    CaProp,
    SOAP,
    DistributedAdam,
    OrthoAdam,
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


def test_sm3():
    assert _run_step(SM3, lr=0.01)


def test_demeter():
    assert _run_step(Demeter, lr=0.01)


def test_caprop():
    assert _run_step(CaProp, lr=0.01)


def test_soap():
    assert _run_step(SOAP, lr=0.01)


def test_distributed_adam():
    assert _run_step(DistributedAdam, lr=0.01)


def test_orthoadam():
    assert _run_step(OrthoAdam, lr=0.01)


def test_multiple_steps():
    p = _make_param()
    opt = SM3([p], lr=0.01)
    for _ in range(3):
        p.grad = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
        opt.step()
        opt.zero_grad()
    # params should have moved
    assert not np.allclose(p.data, np.array([[1.0, 2.0], [3.0, 4.0]]))
