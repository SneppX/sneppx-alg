import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad, Context


def _engine_hessian(build, p):
    p.grad = None
    L = build(p)
    g = grad(L, [p], create_graph=True)[0]
    flat = p.numel
    H = np.zeros((flat, flat), dtype=np.float64)
    for j in range(flat):
        p.grad = None
        gj = g[np.unravel_index(j, g.shape)]
        gj.backward()
        H[j, :] = p.grad.data.reshape(-1).astype(np.float64)
    return H


def _analytic_grad_w(x_np, w_np, stride, padding):
    ctx = Context()
    out = ops.Conv1d.forward(ctx, Tensor(x_np), Tensor(w_np), stride, padding)
    g_out = Tensor(out.data, dtype=out.dtype)  # dL/dout = out for 0.5*sum(out^2)
    grads = ops.Conv1d.backward(ctx, g_out)
    return grads[1].data


def _analytic_grad_x(x_np, w_np, stride, padding):
    ctx = Context()
    out = ops.Conv1d.forward(ctx, Tensor(x_np), Tensor(w_np), stride, padding)
    g_out = Tensor(out.data, dtype=out.dtype)
    grads = ops.Conv1d.backward(ctx, g_out)
    return grads[0].data


def _fd_hessian(target, grad_fn, h=1e-4):
    flat = target.size
    H = np.zeros((flat, flat), dtype=np.float64)
    base = target.ravel().astype(np.float64).copy()
    for j in range(flat):
        bp = base.copy(); bp[j] += h
        bn = base.copy(); bn[j] -= h
        gp = grad_fn(bp.reshape(target.shape)).ravel().astype(np.float64)
        gn = grad_fn(bn.reshape(target.shape)).ravel().astype(np.float64)
        H[:, j] = (gp - gn) / (2 * h)
    return H


def _run(stride, padding, seed):
    np.random.seed(seed)
    N, C_in, C_out, L, K = 2, 3, 4, 9, 3
    x_np = np.random.randn(N, C_in, L).astype(np.float64) * 0.7
    w_np = np.random.randn(C_out, C_in, K).astype(np.float64) * 0.7
    x = Tensor(x_np.astype("float32"))
    x.requires_grad_(True)
    w = Tensor(w_np.astype("float32"))
    w.requires_grad_(True)

    def build(xin, wkern):
        out = ops.Conv1d.apply(xin, wkern, stride, padding)
        sq = out ** 2
        return sq.sum() * 0.5

    # Kernel Hessian (grad_w is linear in w -> FD essentially exact)
    Hw_eng = _engine_hessian(lambda ww: build(x, ww), w)
    Hw_ref = _fd_hessian(w_np, lambda ww: _analytic_grad_w(x_np, ww, stride, padding))
    assert np.allclose(Hw_eng, Hw_ref, atol=2e-2), (
        f"Conv1d(H={stride},pad={padding}) kernel Hessian max|err|="
        f"{np.max(np.abs(Hw_eng - Hw_ref)):.3e}"
    )

    # Input Hessian (grad_x is linear in x -> FD essentially exact)
    Hx_eng = _engine_hessian(lambda xx: build(xx, w), x)
    Hx_ref = _fd_hessian(x_np, lambda xx: _analytic_grad_x(xx, w_np, stride, padding))
    assert np.allclose(Hx_eng, Hx_ref, atol=2e-2), (
        f"Conv1d(H={stride},pad={padding}) input Hessian max|err|="
        f"{np.max(np.abs(Hx_eng - Hx_ref)):.3e}"
    )


def test_conv1d_double_backward_stride1_pad0():
    _run(stride=1, padding=0, seed=51)


def test_conv1d_double_backward_stride2_pad1():
    _run(stride=2, padding=1, seed=53)


if __name__ == "__main__":
    test_conv1d_double_backward_stride1_pad0()
    test_conv1d_double_backward_stride2_pad1()
    print("ALL CONV1D HIGHER-ORDER AD TESTS PASSED")
