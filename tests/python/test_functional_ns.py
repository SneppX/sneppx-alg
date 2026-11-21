"""Tests for the torch.nn.functional-compatible namespace (F.*)."""
import numpy as np
import pytest

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import F, nn
from SneppX_ALG.interface_bindings.nn import functional as F_mod


@pytest.fixture
def x():
    return Tensor(np.random.randn(2, 3, 4).astype("float32"))


def test_namespace_surface():
    assert callable(F.relu)
    assert callable(F.softmax)
    assert callable(F.linear)
    assert callable(F.cross_entropy)
    assert F_mod is F
    assert nn.F is F
    assert hasattr(F, "relu") and hasattr(F, "pad") and hasattr(F, "one_hot")


def test_activations(x):
    assert F.relu(x).shape == x.shape
    assert np.allclose(F.relu(x).data, np.maximum(x.data, 0))
    assert F.elu(x).shape == x.shape
    assert F.selu(x).shape == x.shape
    assert F.gelu(x).shape == x.shape
    assert F.silu(x).shape == x.shape
    assert F.mish(x).shape == x.shape
    assert F.softplus(x).shape == x.shape
    assert F.softsign(x).shape == x.shape
    assert F.hardswish(x).shape == x.shape
    assert F.hardtanh(x).shape == x.shape
    assert F.hardsigmoid(x).shape == x.shape
    assert F.hardshrink(x).shape == x.shape
    assert F.softshrink(x).shape == x.shape
    assert F.tanhshrink(x).shape == x.shape
    assert F.sigmoid(x).shape == x.shape
    assert F.tanh(x).shape == x.shape
    assert F.glu(x, dim=-1).shape == (2, 3, 2)


def test_softmax_logsoftmax(x):
    sm = F.softmax(x, dim=1)
    assert np.allclose(sm.data.sum(axis=1), 1.0, atol=1e-5)
    lsm = F.log_softmax(x, dim=1)
    assert np.allclose(np.exp(lsm.data).sum(axis=1), 1.0, atol=1e-4)


def test_linear():
    x = Tensor(np.random.randn(4, 8).astype("float32"))
    w = Tensor(np.random.randn(5, 8).astype("float32"))
    b = Tensor(np.random.randn(5).astype("float32"))
    with_b = F.linear(x, w, b)
    no_b = F.linear(x, w)
    assert with_b.shape == (4, 5)
    expected = x.data @ w.data.T + b.data
    assert np.allclose(with_b.data, expected, atol=1e-4)


def test_conv2d_roundtrip():
    x = Tensor(np.random.randn(1, 2, 6, 6).astype("float32"))
    w = Tensor(np.random.randn(4, 2, 3, 3).astype("float32"))
    out = F.conv2d(x, w, None, stride=1, padding=1)
    assert out.shape == (1, 4, 6, 6)


def test_conv3d():
    x = Tensor(np.random.randn(1, 2, 4, 5, 6).astype("float32"))
    w = Tensor(np.random.randn(3, 2, 2, 3, 3).astype("float32"))
    out = F.conv3d(x, w, None, stride=1, padding=1)
    # kD=2,pad=1 -> D_out=D+1=5; kH=3,pad=1 -> H_out=H; kW=3,pad=1 -> W_out=W
    assert out.shape == (1, 3, 5, 5, 6)


def test_pool():
    x = Tensor(np.random.randn(1, 2, 8, 8).astype("float32"))
    mp = F.max_pool2d(x, 2)
    assert mp.shape == (1, 2, 4, 4)
    ap = F.avg_pool2d(x, 2)
    assert ap.shape == (1, 2, 4, 4)
    aap = F.adaptive_avg_pool2d(x, (4, 4))
    assert aap.shape == (1, 2, 4, 4)


def test_pool1d():
    x = Tensor(np.random.randn(2, 3, 10).astype("float32"))
    mp = F.max_pool1d(x, 2)
    assert mp.shape == (2, 3, 5)
    ap = F.avg_pool1d(x, 2)
    assert ap.shape == (2, 3, 5)


def test_norm(x):
    w = Tensor(np.ones((x.shape[-1],), dtype="float32"))
    b = Tensor(np.zeros((x.shape[-1],), dtype="float32"))
    ln = F.layer_norm(x, x.shape[-1:], w, b)
    assert ln.shape == x.shape
    # group_norm needs a 4D (N,C,H,W) layout
    x4 = Tensor(np.random.randn(2, 4, 3, 3).astype("float32"))
    w4 = Tensor(np.ones((4,), dtype="float32"))
    b4 = Tensor(np.zeros((4,), dtype="float32"))
    gn = F.group_norm(x4, 2, w4, b4)
    assert gn.shape == x4.shape


def test_losses():
    pred = Tensor(np.random.randn(5).astype("float32"))
    tgt = Tensor(np.random.randn(5).astype("float32"))
    assert np.isfinite(F.mse_loss(pred, tgt).data).all()
    assert np.isfinite(F.l1_loss(pred, tgt).data).all()
    assert np.isfinite(F.smooth_l1_loss(pred, tgt).data).all()
    assert np.isfinite(F.huber_loss(pred, tgt).data).all()
    # cross entropy
    logits = Tensor(np.random.randn(4, 5).astype("float32"))
    labels = Tensor(np.random.randint(0, 5, (4,)).astype("float32"))
    assert np.isfinite(F.cross_entropy(logits, labels).data).all()
    assert np.isfinite(F.nll_loss(logits, labels).data).all()
    # binary CE
    b = Tensor((np.random.rand(4, 5) > 0.5).astype("float32"))
    assert np.isfinite(F.binary_cross_entropy(b, b).data).all()
    assert np.isfinite(F.binary_cross_entropy_with_logits(logits, b).data).all()


def test_pad():
    x4 = Tensor(np.random.randn(2, 3, 4, 4).astype("float32"))
    p = F.pad(x4, (1, 1, 2, 2))
    # torch pad tuple last-dim-first: (W±1, H±2) -> (2,3,8,6)
    assert p.shape == (2, 3, 8, 6)
    x3 = Tensor(np.random.randn(2, 3, 4).astype("float32"))
    p2 = F.constant_pad_nd(x3, 2)
    assert p2.shape == (2, 3, 8)
    pr = F.reflection_pad2d(x4, (1, 1, 1, 1))
    assert pr.shape == (2, 3, 6, 6)


def test_flatten_unflatten(x):
    f = F.flatten(x, start_dim=1)
    assert f.shape == (2, 12)
    u = F.unflatten(f, 1, (3, 4))
    assert u.shape == x.shape


def test_unfold_fold():
    x4 = Tensor(np.random.randn(2, 3, 4, 4).astype("float32"))
    u = F.unfold(x4, (2, 2), stride=1)
    assert u.shape == (2, 12, 9)
    f = F.fold(u, (x4.shape[2], x4.shape[3]), (2, 2), stride=1)
    assert f.shape == x4.shape
    assert np.allclose(f.data, x4.data, atol=1e-5)


def test_embedding():
    w = Tensor(np.random.randn(10, 4).astype("float32"))
    idx = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int64))
    out = F.embedding(idx, w)
    assert out.shape == (2, 2, 4)
    assert np.allclose(out.data[0, 0], w.data[1])


def test_embedding_bag():
    w = Tensor(np.random.randn(10, 3).astype("float32"))
    idx = Tensor(np.array([0, 1, 2, 0], dtype=np.int64))
    off = Tensor(np.array([0, 2, 4], dtype=np.int64))
    out = F.embedding_bag(idx, w, off, mode="mean")
    assert out.shape == (2, 3)


def test_one_hot():
    t = Tensor(np.array([0, 1, 2], dtype=np.int64))
    oh = F.one_hot(t, num_classes=3)
    assert oh.shape == (3, 3)
    assert np.allclose(oh.data, np.eye(3))


def test_interpolate():
    x = Tensor(np.random.randn(1, 2, 4, 4).astype("float32"))
    up = F.interpolate(x, scale_factor=2, mode="nearest")
    assert up.shape == (1, 2, 8, 8)
    upb = F.interpolate(x, size=(6, 6), mode="bilinear")
    assert upb.shape == (1, 2, 6, 6)


def test_pixel_channel_shuffle():
    x = Tensor(np.random.randn(1, 4, 6, 6).astype("float32"))
    assert F.channel_shuffle(x, 2).shape == x.shape
    ps = F.pixel_shuffle(Tensor(np.random.randn(1, 8, 4, 4).astype("float32")), 2)
    assert ps.shape == (1, 2, 8, 8)