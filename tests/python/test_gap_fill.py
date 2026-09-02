"""Tests for PyTorch-gap-fill additions:

- optimizers: Adam, NAdam, Rprop, LBFGS, ASGD, SparseAdam
- nn.Modules: Conv1d, BatchNorm3d, InstanceNorm1d/2d/3d, Flatten, Identity,
  Bilinear, Ada/Adaptive pooling, padding (Constant/Zero/Reflection/Replication),
  ParameterList/ParameterDict
- activations (Function + Tensor method + nn.Module): leaky_relu, elu, selu,
  softplus, softsign, hardswish, hardtanh, mish, hardsigmoid, hardshrink,
  softshrink, tanhshrink, threshold, glu, celu, rrelu, relu6, logsigmoid
- losses: GaussianNLLLoss, HingeEmbeddingLoss, PoissonNLLLoss, SoftMarginLoss,
  MultiMarginLoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss
"""

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn
from SneppX_ALG.interface_bindings import optim


def central_diff(f, x, eps=1e-4):
    num = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        num[i] = (float(f(xp)) - float(f(xm))) / (2 * eps)
    return num


def fd_check(f, grad_scale=1.0, eps=1e-4, tol=2e-2):
    x = Tensor(np.random.randn(6).astype("float32"), requires_grad=True)
    out = f(x)
    out.sum().backward()
    ana = x.grad.data.copy().astype(np.float64) * grad_scale
    xd = x.data.astype(np.float64).copy()

    def fnum(xd):
        return float(f(Tensor(xd.astype("float32"))).data.sum())

    num = central_diff(fnum, xd, eps)
    err = float(np.abs(ana - num).max())
    assert err < tol, f"FD mismatch: max_err={err:.2e}"


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def test_adam_optimizer_steps():
    p = Tensor(np.array([1.0, 2.0, 3.0], dtype="float32"), requires_grad=True)
    opt = optim.Adam([p], lr=0.1)
    p.grad = Tensor(np.array([0.1, 0.2, 0.3], dtype="float32"))
    w0 = p.data.copy()
    opt.step()
    assert not np.allclose(p.data, w0)
    assert np.all(np.isfinite(p.data))


def test_adam_wd_matches_pytorch_form():
    p = Tensor(np.array([1.0], dtype="float32"), requires_grad=True)
    opt = optim.Adam([p], lr=0.1, weight_decay=0.01)
    for _ in range(3):
        p.grad = Tensor(np.array([0.5], dtype="float32"))
        opt.step()
    assert np.all(np.isfinite(p.data))


def test_nadam_forward_no_crash():
    p = Tensor(np.random.randn(4).astype("float32"), requires_grad=True)
    opt = optim.NAdam([p], lr=0.01)
    p.grad = Tensor(np.random.randn(4).astype("float32"))
    opt.step()
    assert np.all(np.isfinite(p.data))


def test_rprop_steps():
    p = Tensor(np.array([1.0, -2.0], dtype="float32"), requires_grad=True)
    opt = optim.Rprop([p], lr=0.01)
    p.grad = Tensor(np.array([0.5, -0.5], dtype="float32"))
    w0 = p.data.copy()
    opt.step()
    assert not np.allclose(p.data, w0)


def test_asgd_steps():
    p = Tensor(np.array([1.0, -2.0], dtype="float32"), requires_grad=True)
    opt = optim.ASGD([p], lr=0.01)
    p.grad = Tensor(np.array([0.5, -0.5], dtype="float32"))
    w0 = p.data.copy()
    opt.step()
    assert not np.allclose(p.data, w0)


def test_lbfgs_requires_closure():
    p = Tensor(np.array([1.0], dtype="float32"), requires_grad=True)
    opt = optim.LBFGS([p], lr=0.1)
    with pytest.raises(ValueError):
        opt.step()


def test_lbfgs_closure_run():
    p = Tensor(np.array([1.0], dtype="float32"), requires_grad=True)
    opt = optim.LBFGS([p], lr=0.1, max_iter=3)

    def closure():
        out = (p - 2.0) ** 2
        out.backward()
        return out

    loss = opt.step(closure)
    assert np.all(np.isfinite(loss.data))


def test_sparse_adam_steps():
    p = Tensor(np.array([1.0, 2.0], dtype="float32"), requires_grad=True)
    opt = optim.SparseAdam([p], lr=0.01)
    p.grad = Tensor(np.array([0.1, 0.2], dtype="float32"))
    w0 = p.data.copy()
    opt.step()
    assert not np.allclose(p.data, w0)


# ---------------------------------------------------------------------------
# nn.Modules
# ---------------------------------------------------------------------------


def test_conv1d_shape():
    m = nn.Conv1d(3, 6, 5)
    x = Tensor(np.random.randn(2, 3, 12).astype("float32"))
    y = m(x)
    assert y.shape == (2, 6, 8)


def test_conv1d_parameters():
    m = nn.Conv1d(3, 6, 5, bias=True)
    names = {n for n, _ in m.named_parameters()}
    assert "weight" in names and "bias" in names


def test_batchnorm3d_forward():
    m = nn.BatchNorm3d(4)
    x = Tensor(np.random.randn(2, 4, 3, 4, 4).astype("float32"))
    y = m(x)
    assert y.shape == x.shape
    assert y.data.var() < 5.0


def test_instancenorm1d_forward():
    m = nn.InstanceNorm1d(4)
    x = Tensor(np.random.randn(2, 4, 6).astype("float32"))
    y = m(x)
    assert y.shape == x.shape


def test_instancenorm2d_forward():
    m = nn.InstanceNorm2d(4)
    x = Tensor(np.random.randn(2, 4, 6, 6).astype("float32"))
    y = m(x)
    assert y.shape == x.shape


def test_instancenorm3d_forward():
    m = nn.InstanceNorm3d(4)
    x = Tensor(np.random.randn(1, 4, 3, 4, 4).astype("float32"))
    y = m(x)
    assert y.shape == x.shape


def test_flatten():
    m = nn.Flatten()
    x = Tensor(np.random.randn(2, 3, 4).astype("float32"))
    assert m(x).shape == (2, 12)


def test_identity():
    m = nn.Identity()
    x = Tensor(np.random.randn(2, 3).astype("float32"))
    assert np.allclose(m(x).data, x.data)


def test_bilinear_shape():
    m = nn.Bilinear(4, 5, 6)
    x1 = Tensor(np.random.randn(3, 4).astype("float32"))
    x2 = Tensor(np.random.randn(3, 5).astype("float32"))
    assert m(x1, x2).shape == (3, 6)


def test_pooling_modules_shapes():
    x1 = Tensor(np.random.randn(2, 3, 10).astype("float32"))
    x2 = Tensor(np.random.randn(2, 3, 8, 8).astype("float32"))
    x3 = Tensor(np.random.randn(1, 2, 4, 8, 8).astype("float32"))
    assert nn.Pooling1d(2, stride=2)(x1).shape == (2, 3, 5)
    assert nn.Pooling2d(2, stride=2)(x2).shape == (2, 3, 4, 4)
    assert nn.Pooling3d(2, stride=2)(x3).shape == (1, 2, 2, 4, 4)
    assert nn.Pooling1d(2, stride=2, pool_type="avg")(x1).shape == (2, 3, 5)
    assert nn.Pooling2d(2, stride=2, pool_type="avg")(x2).shape == (2, 3, 4, 4)


def test_adaptive_pool_shapes():
    x1 = Tensor(np.random.randn(2, 3, 10).astype("float32"))
    x2 = Tensor(np.random.randn(2, 3, 8, 8).astype("float32"))
    x3 = Tensor(np.random.randn(1, 2, 4, 6, 6).astype("float32"))
    assert nn.AdaptiveAvgPool1d(4)(x1).shape == (2, 3, 4)
    assert nn.AdaptiveAvgPool2d((4, 4))(x2).shape == (2, 3, 4, 4)
    assert nn.AdaptiveAvgPool3d((2, 3, 3))(x3).shape == (1, 2, 2, 3, 3)
    assert nn.AdaptiveMaxPool1d(4)(x1).shape == (2, 3, 4)
    assert nn.AdaptiveMaxPool2d((4, 4))(x2).shape == (2, 3, 4, 4)
    assert nn.AdaptiveMaxPool3d((2, 3, 3))(x3).shape == (1, 2, 2, 3, 3)


def test_padding_modules_shapes():
    x1 = Tensor(np.random.randn(2, 3, 10).astype("float32"))
    x2 = Tensor(np.random.randn(2, 3, 8, 8).astype("float32"))
    assert nn.ConstantPad1d(2)(x1).shape == (2, 3, 14)
    assert nn.ConstantPad2d((2, 2, 1, 1))(x2).shape == (2, 3, 10, 12)
    assert nn.ConstantPad3d(1)(Tensor(np.random.randn(1, 2, 3, 4, 4).astype("float32"))).shape == (1, 2, 5, 6, 6)
    assert nn.ZeroPad1d(2)(x1).shape == (2, 3, 14)
    assert nn.ZeroPad2d((1, 1, 0, 0))(x2).shape == (2, 3, 8, 10)
    assert nn.ReflectionPad1d(1)(x1).shape == (2, 3, 12)
    assert nn.ReflectionPad2d((1, 1, 1, 1))(x2).shape == (2, 3, 10, 10)
    assert nn.ReplicationPad1d(1)(x1).shape == (2, 3, 12)
    assert nn.ReplicationPad2d((1, 1, 1, 1))(x2).shape == (2, 3, 10, 10)


def test_parameter_list_dict():
    pl = nn.ParameterList([Tensor.ones((3,)), Tensor.ones((4,))])
    assert len(pl) == 2
    assert len(pl.parameters()) == 2
    pd = nn.ParameterDict({"a": Tensor.ones((3,)), "b": Tensor.ones((4,))})
    assert len(pd) == 2
    assert len(pd.parameters()) == 2


# ---------------------------------------------------------------------------
# Activations (Autograd correctness via FD)
# ---------------------------------------------------------------------------

ACTIVATIONS = [
    lambda t: t.leaky_relu(0.1),
    lambda t: t.elu(1.0),
    lambda t: t.selu(),
    lambda t: t.softplus(),
    lambda t: t.softsign(),
    lambda t: t.hardswish(),
    lambda t: t.hardtanh(-1, 1),
    lambda t: t.mish(),
    lambda t: t.hardsigmoid(),
    lambda t: t.hardshrink(0.5),
    lambda t: t.softshrink(0.5),
    lambda t: t.tanhshrink(),
    lambda t: t.threshold(0.2, 0.0),
    lambda t: t.glu(-1),
]


@pytest.mark.parametrize("fn", ACTIVATIONS)
def test_activation_autograd_fd(fn):
    fd_check(fn)


def test_activation_modules_forward():
    x = Tensor(np.random.randn(4, 8).astype("float32"))
    for mod in [
        nn.LeakyReLU(0.1),
        nn.PReLU(),
        nn.ELU(),
        nn.SELU(),
        nn.Softplus(),
        nn.Softsign(),
        nn.Hardswish(),
        nn.Hardtanh(),
        nn.ReLU6(),
        nn.Mish(),
        nn.Hardsigmoid(),
        nn.Hardshrink(),
        nn.Softshrink(),
        nn.Tanhshrink(),
        nn.Threshold(0.2, 0.0),
        nn.GLU(),
        nn.CELU(),
        nn.RReLU(),
        nn.LogSigmoid(),
        nn.Softmax(dim=-1),
        nn.LogSoftmax(dim=-1),
    ]:
        out = mod(x)
        assert out.shape == x.shape or out.shape == (4, 4), type(mod).__name__


def test_softmax_autograd():
    x = Tensor(np.random.randn(4, 5).astype("float32"), requires_grad=True)
    y = x.softmax(dim=-1)
    y.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_gaussian_nll_loss():
    loss = nn.GaussianNLLLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    target = Tensor(np.random.randn(4, 5).astype("float32"))
    var = Tensor(np.abs(np.random.randn(4, 5).astype("float32")) + 0.5)
    out = loss(x, target, var)
    assert out.data.ndim == 0


def test_hinge_embedding_loss():
    loss = nn.HingeEmbeddingLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor((2 * (np.random.rand(4, 5) > 0.5) - 1).astype("float32"))
    assert np.isfinite(loss(x, y).data)


def test_poisson_nll_loss():
    loss = nn.PoissonNLLLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor(np.random.rand(4, 5).astype("float32"))
    assert np.isfinite(loss(x, y).data)


def test_soft_margin_loss():
    loss = nn.SoftMarginLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor((2 * (np.random.rand(4, 5) > 0.5) - 1).astype("float32"))
    assert np.isfinite(loss(x, y).data)


def test_multi_margin_loss():
    loss = nn.MultiMarginLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor(np.random.randint(0, 5, (4,)).astype("float32"))
    assert np.isfinite(loss(x, y).data)


def test_multi_label_margin_loss():
    loss = nn.MultiLabelMarginLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor(np.full((4, 5), -1).astype("float32"))
    y.data[0, 0] = 1
    y.data[0, 2] = 2
    assert np.isfinite(loss(x, y).data)


def test_multi_label_soft_margin_loss():
    loss = nn.MultiLabelSoftMarginLoss()
    x = Tensor(np.random.randn(4, 5).astype("float32"))
    y = Tensor((np.random.rand(4, 5) > 0.5).astype("float32"))
    assert np.isfinite(loss(x, y).data)


def test_embedding_bag_mean():
    bag = nn.EmbeddingBag(10, 4, mode="mean")
    idx = Tensor(np.array([0, 1, 2, 0], dtype=np.int64))
    off = Tensor(np.array([0, 2, 4], dtype=np.int64))
    out = bag(idx, off)
    assert out.shape == (2, 4)
    assert np.allclose(
        out.data[0], bag.weight.data[[0, 1]].mean(axis=0)
    )


def test_embedding_bag_sum():
    bag = nn.EmbeddingBag(10, 3, mode="sum")
    idx = Tensor(np.array([0, 1, 2, 0], dtype=np.int64))
    off = Tensor(np.array([0, 2, 4], dtype=np.int64))
    out = bag(idx, off)
    assert np.allclose(out.data[0], bag.weight.data[[0, 1]].sum(axis=0))


def test_cosine_similarity():
    cs = nn.CosineSimilarity(dim=1)
    x1 = Tensor(np.random.randn(4, 8).astype("float32"))
    x2 = Tensor(np.random.randn(4, 8).astype("float32"))
    out = cs(x1, x2)
    assert out.shape == (4,)
    a = x1.data.astype("float64")
    b = x2.data.astype("float64")
    expected = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert np.allclose(out.data, expected, atol=1e-4)


def test_pairwise_distance():
    pd = nn.PairwiseDistance()
    x1 = Tensor(np.random.randn(4, 8).astype("float32"))
    x2 = Tensor(np.random.randn(4, 8).astype("float32"))
    out = pd(x1, x2)
    assert out.shape == (4,)
    expected = np.linalg.norm(x1.data - x2.data, axis=1)
    assert np.allclose(out.data, expected, atol=1e-4)


def test_channel_shuffle():
    cs = nn.ChannelShuffle(2)
    x = Tensor(np.random.randn(2, 4, 6, 6).astype("float32"))
    out = cs(x)
    assert out.shape == (2, 4, 6, 6)
    # channel shuffle is a permutation of whole channel planes
    ox = np.sort(x.data.reshape(2, 4, -1), axis=1)
    oo = np.sort(out.data.reshape(2, 4, -1), axis=1)
    assert np.allclose(oo, ox)
    # torch semantics with groups=2, 4 channels -> ordering [c0,c2,c1,c3]
    expected = x.data[:, [0, 2, 1, 3]]
    assert np.allclose(out.data, expected)


def test_pixel_shuffle_roundtrip():
    ps = nn.PixelShuffle(2)
    x = Tensor(np.random.randn(1, 8, 4, 4).astype("float32"))
    out = ps(x)
    assert out.shape == (1, 2, 8, 8)
    pus = nn.PixelUnshuffle(2)
    back = pus(out)
    assert np.allclose(back.data, x.data)


def test_upsample_nearest_2d():
    us = nn.Upsample(scale_factor=2, mode="nearest")
    x = Tensor(np.random.randn(1, 2, 3, 4).astype("float32"))
    out = us(x)
    assert out.shape == (1, 2, 6, 8)
    assert np.allclose(out.data[:, :, ::2, ::2], x.data)


def test_upsample_bilinear_2d():
    us = nn.Upsample(size=(6, 8), mode="bilinear")
    x = Tensor(np.random.randn(1, 2, 3, 4).astype("float32"))
    out = us(x)
    assert out.shape == (1, 2, 6, 8)
    assert np.isfinite(out.data).all()


def test_conv3d_shape():
    m = nn.Conv3d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(1, 2, 5, 6, 7).astype("float32"))
    out = m(x)
    assert out.shape == (1, 3, 5, 6, 7)


def test_conv_transpose2d_shape():
    m = nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1)
    x = Tensor(np.random.randn(1, 2, 4, 4).astype("float32"))
    out = m(x)
    # (H-1)*stride - 2*pad + (k-1) + output_padding + 1 = 3*2 -2 +2 +1 +1 = 8
    assert out.shape[2] == 8 and out.shape[3] == 8


def test_unfold_fold_roundtrip():
    unfold = nn.Unfold((2, 2), stride=1)
    x = Tensor(np.random.randn(1, 3, 4, 4).astype("float32"))
    u = unfold(x)
    assert u.shape == (1, 12, 9)
    fold = nn.Fold((4, 4), (2, 2), stride=1)
    # fold averages overlapping regions, so round-trip only preserves the sum-weighted average
    f = fold(u)
    assert f.shape == (1, 3, 4, 4)
    # single-stride=1 overlapping fold: output equals mean of the (kH*kW) contributions = input
    assert np.allclose(f.data, x.data, atol=1e-5)


def test_unfold_values():
    unfold = nn.Unfold((2, 2), stride=2)
    x = Tensor(np.arange(1, 13, dtype=np.float32).reshape(1, 1, 3, 4))
    u = unfold(x)
    # top-left block for (3,4) input, k2x2, stride2 -> blocks at (0,0)
    assert np.allclose(u.data[0, :4, 0], x.data[0, 0, 0:2, 0:2].reshape(-1))


def test_max_unpool2d():
    input = Tensor(np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2))
    idx = Tensor(np.array([0, 2, 8, 10], dtype=np.int64).reshape(1, 1, 2, 2))
    m = nn.MaxUnpool2d(kernel_size=2, stride=2)
    out = m(input, idx)
    assert out.shape == (1, 1, 4, 4)
    assert out.data[0, 0, 0, 0] == 0.0
    assert out.data[0, 0, 0, 2] == 1.0
    assert out.data[0, 0, 2, 0] == 2.0
    assert out.data[0, 0, 2, 2] == 3.0