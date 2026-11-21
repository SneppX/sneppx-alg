"""Tests for the C-engine autodiff tape (`_Tape` / `_Variable` bindings).

Covers issue #14 follow-up work: after the ownership fixes (recorded
variables and their tensors belong to the tape; python wrappers are
non-owning views), the tape must compute correct forward values, exact
analytic backward gradients, and exit the interpreter cleanly (no heap
corruption / double free at teardown).
"""

import subprocess
import sys
import os

import numpy as np
import pytest

from SneppX_ALG import _neural_engine_bridge as ext

T = ext._Tensor
V = ext._Variable

TAPE_SCRIPT = r"""
import numpy as np
from SneppX_ALG import _neural_engine_bridge as ext
T, V = ext._Tensor, ext._Variable

def var(arr, requires_grad=1):
    t = T.create(np.array(arr.shape), ext.SNEPPXDtype.FLOAT32)
    t.set_data(np.ascontiguousarray(arr).ravel().astype(np.float32))
    return V.create(t, requires_grad)

tape = ext._Tape.create()
a = var(np.arange(6, dtype=np.float32).reshape(2, 3))
tape.record(a)
w = var(np.ones((3, 2), dtype=np.float32))
tape.record(w)
y = tape.matmul(a, w)
l = tape.mul(y, y)
tape.backward(l)
ga = np.asarray(a.grad.data()).ravel()
assert np.allclose(ga, [12, 12, 12, 48, 48, 48]), ga
gw = np.asarray(w.grad.data()).ravel()
assert np.allclose(gw, [72, 72, 102, 102, 132, 132]), gw
print("tape-ok")
"""


def _var(arr, requires_grad=1):
    t = T.create(np.array(arr.shape), ext.SNEPPXDtype.FLOAT32)
    t.set_data(np.ascontiguousarray(arr).ravel().astype(np.float32))
    return V.create(t, requires_grad)


def _tape_with(a_arr, *more):
    tape = ext._Tape.create()
    a = _var(a_arr)
    tape.record(a)
    others = []
    for m in more:
        v = _var(m)
        tape.record(v)
        others.append(v)
    return tape, a, others


def test_tape_add_mul_forward_parity():
    tape, a, (b,) = _tape_with(np.ones((2, 3), np.float32) * 3, np.ones((2, 3), np.float32) * 2)
    r = tape.add(a, b)
    assert r.data.shape == (2, 3)
    assert np.allclose(np.asarray(r.data.data()), 5.0)
    m = tape.mul(r, r)
    assert m.data.shape == (2, 3)
    assert np.allclose(np.asarray(m.data.data()), 25.0)


def test_tape_sub_div_forward_parity():
    tape, a, (b,) = _tape_with(np.ones((2, 3), np.float32) * 6, np.ones((2, 3), np.float32) * 2)
    s = tape.sub(a, b)
    assert np.allclose(np.asarray(s.data.data()), 4.0)
    d = tape.div(a, b)
    assert np.allclose(np.asarray(d.data.data()), 3.0)


def test_tape_matmul_forward_parity():
    tape, a, (w,) = _tape_with(
        np.arange(6, dtype=np.float32).reshape(2, 3), np.ones((3, 2), np.float32)
    )
    y = tape.matmul(a, w)
    assert y.data.shape == (2, 2)
    ref = np.arange(6, dtype=np.float32).reshape(2, 3) @ np.ones((3, 2), np.float32)
    assert np.allclose(np.asarray(y.data.data()), ref)


@pytest.mark.parametrize(
    "op,np_fn",
    [
        ("relu", lambda x: np.maximum(x, 0)),
        ("sigmoid", lambda x: 1 / (1 + np.exp(-x))),
        ("tanh", np.tanh),
        ("exp", np.exp),
    ],
)
def test_tape_unary_forward_parity(op, np_fn):
    a_arr = np.array([[-1.0, 0.5, 2.0], [0.0, -0.5, 1.5]], dtype=np.float32)
    tape = ext._Tape.create()
    a = _var(a_arr)
    tape.record(a)
    r = getattr(tape, op)(a)
    assert r.data.shape == (2, 3)
    assert np.allclose(np.asarray(r.data.data()), np_fn(a_arr), atol=1e-6)


def test_tape_softmax_forward_parity():
    a_arr = np.array([[-1.0, 0.5, 2.0], [0.0, -0.5, 1.5]], dtype=np.float32)
    tape = ext._Tape.create()
    a = _var(a_arr)
    tape.record(a)
    sm = tape.softmax(a, 1)
    ref = np.exp(a_arr) / np.exp(a_arr).sum(axis=1, keepdims=True)
    assert np.allclose(np.asarray(sm.data.data()), ref, atol=1e-6)


def test_tape_matmul_mul_backward_exact():
    """dL/da and dL/dw for L = (a @ w)^2 must match the analytic values."""
    tape, a, (w,) = _tape_with(
        np.arange(6, dtype=np.float32).reshape(2, 3), np.ones((3, 2), np.float32)
    )
    y = tape.matmul(a, w)
    l = tape.mul(y, y)
    tape.backward(l)
    assert a.grad is not None and w.grad is not None
    ga = np.asarray(a.grad.data()).ravel()
    gw = np.asarray(w.grad.data()).ravel()
    assert np.allclose(ga, [12, 12, 12, 48, 48, 48]), ga
    assert np.allclose(gw, [72, 72, 102, 102, 132, 132]), gw


def test_tape_mse_loss_backward_exact():
    tape, p, (tg,) = _tape_with(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )
    loss = tape.mse_loss(p, tg)
    assert np.allclose(np.asarray(loss.data.data()).ravel(), 7.5)
    tape.backward(loss)
    gp = np.asarray(p.grad.data()).ravel()
    assert np.allclose(gp, [0.5, 1.0, 1.5, 2.0]), gp


def test_tape_sum_backward_keeps_dim():
    a_arr = np.array([[-1.0, 0.5, 2.0], [0.0, -0.5, 1.5]], dtype=np.float32)
    tape = ext._Tape.create()
    a = _var(a_arr)
    tape.record(a)
    z = tape.sum(tape.mul(a, a), 1)
    assert z.data.shape == (2, 1)
    assert np.allclose(np.asarray(z.data.data()).ravel(), [5.25, 2.5])
    tape.backward(z)
    # d/d a[i,j] sum(a^2,1)[i] = 2 a[i,j]
    assert np.allclose(np.asarray(a.grad.data()), 2 * a_arr, atol=1e-6)


def test_tape_double_record_is_safe():
    """Recording the same variable twice must not corrupt teardown."""
    tape, a, (b,) = _tape_with(np.ones((2, 3), np.float32) * 3, np.ones((2, 3), np.float32))
    tape.record(a)  # duplicate on purpose
    r = tape.add(a, b)
    assert np.allclose(np.asarray(r.data.data()), 4.0)


def test_tape_teardown_clean_exit_subprocess():
    """Full tape flow in a subprocess must exit 0.

    Regression guard for issue #14: the tape owns every recorded variable,
    and an ownership mismatch between the C tape and the python wrappers
    used to trigger a double free (exit status 0xC0000374) at interpreter
    shutdown even though all values printed correctly.
    """
    env = dict(os.environ)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.path.join(repo, "bindings", "python")
    proc = subprocess.run(
        [sys.executable, "-c", TAPE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "tape-ok" in proc.stdout
