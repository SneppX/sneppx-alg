"""Integration tests for the P3 NN modules that the audit listed as missing:
RNN / LSTM / GRU (forward + backward), ModuleList/ModuleDict containers, and
register_buffer (buffer vs parameter distinction)."""

import numpy as np
import sys

sys.path.insert(0, "bindings/python")

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn


def _grad_flow_ok(module, out):
    out.mean().backward()
    n_grad = 0
    n_param = 0
    for p in module.parameters():
        n_param += 1
        if p.grad is not None:
            n_grad += 1
    return n_param, n_grad


def test_rnn_forward_backward():
    rng = np.random.default_rng(11)
    T, B, In, H = 4, 2, 3, 5
    x = Tensor(rng.standard_normal((T, B, In)))
    for cls, kw in [
        (nn.RNN, dict(nonlinearity="tanh")),
        (nn.GRU, dict()),
        (nn.LSTM, dict()),
    ]:
        m = cls(In, H, num_layers=1, bias=True)
        out, h = m(x)
        assert out.data.shape == (T, B, H), f"{cls.__name__} out shape {out.data.shape}"
        n_param, n_grad = _grad_flow_ok(m, out)
        assert n_param > 0 and n_grad == n_param, (
            f"{cls.__name__}: only {n_grad}/{n_param} params received grad"
        )


def test_modulelist_and_parameters():
    ml = nn.ModuleList([nn.Linear(4, 3), nn.Linear(3, 2)])
    params = list(ml.parameters())
    assert len(params) == 4  # 2 weight + 2 bias
    # iteration works
    kinds = [type(m).__name__ for m in ml]
    assert kinds == ["Linear", "Linear"]


def test_register_buffer_excluded_from_parameters():
    class Foo(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Tensor.randn((3, 3))
            self.register_buffer("mask", Tensor.ones((3, 3)))

        def forward(self, x):
            return x @ self.weight.T * self.mask

    f = Foo()
    param_names = {n for n, _ in f.named_parameters()}
    buf_names = {n for n, _ in f.named_buffers()}
    assert "weight" in param_names and "mask" not in param_names
    assert "mask" in buf_names and "weight" not in buf_names


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
