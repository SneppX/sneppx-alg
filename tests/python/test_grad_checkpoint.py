import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn
from SneppX_ALG.interface_bindings import grad_checkpoint as cp


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


def _copy_weights(src, dst):
    for (_, p1), (_, p2) in zip(src.named_parameters(), dst.named_parameters()):
        p2.data = p1.data.copy()


def test_checkpoint_matches_plain():
    np.random.seed(0)
    x = Tensor(np.random.randn(5, 8).astype("float32"))

    net1 = Net()
    out1 = net1(x)
    (out1 * out1).sum().backward()

    net2 = Net()
    _copy_weights(net1, net2)
    out2 = cp.checkpoint(lambda: net2(x))
    (out2 * out2).sum().backward()

    for (n1, p1), (_, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert np.allclose(p1.grad.data, p2.grad.data, atol=1e-5), n1


def test_checkpoint_sequential_matches_plain():
    np.random.seed(1)
    x = Tensor(np.random.randn(5, 8).astype("float32"))

    net1 = Net()
    out1 = net1(x)
    (out1 * out1).sum().backward()

    net3 = Net()
    _copy_weights(net1, net3)
    fns = [lambda z: net3.l1(z), lambda z: net3.relu(z), lambda z: net3.l2(z)]
    out3 = cp.checkpoint_sequential(fns, x, segments=1)
    (out3 * out3).sum().backward()

    for (n1, p1), (_, p3) in zip(net1.named_parameters(), net3.named_parameters()):
        assert p1.grad is not None and p3.grad is not None
        assert np.allclose(p1.grad.data, p3.grad.data, atol=1e-5), n1


def test_no_grad_disables_graph():
    from SneppX_ALG.interface_bindings.autograd import no_grad, is_grad_enabled

    assert is_grad_enabled() is True
    x = Tensor(np.random.randn(3, 4).astype("float32"), requires_grad=True)
    with no_grad():
        assert is_grad_enabled() is False
        y = x * 2
    assert is_grad_enabled() is True
    assert y.grad_fn is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL CHECKPOINT TESTS PASSED")
