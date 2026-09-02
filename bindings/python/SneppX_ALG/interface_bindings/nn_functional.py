"""torch.nn.functional-compatible namespace (``F.*``).

Wraps the differentiable ops defined on ``Tensor`` / in ``advanced_ops`` /
``autograd_ops`` so users can call them in the familiar functional style, e.g.::

    import SneppX_ALG as sx
    from SneppX_ALG.nn import functional as F

    y = F.relu(x)
    out = F.linear(x, w, b)
    loss = F.cross_entropy(logits, targets)

Additive only — re-exports existing op implementations where they exist and adds
thin wrappers for the missing common entries.
"""

from typing import Optional, Tuple, Union
import numpy as np

from .tensor import Tensor
from . import advanced_ops
from . import autograd_ops

__all__ = [
    # activations
    "relu", "leaky_relu", "elu", "selu", "gelu", "silu", "sigmoid", "tanh",
    "softmax", "log_softmax", "softplus", "softsign", "mish", "hardswish",
    "hardtanh", "hardsigmoid", "hardshrink", "softshrink", "tanhshrink",
    "threshold", "glu", "prelu",
    # linear / conv / pool / norm
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
    "conv_transpose2d", "conv_transpose3d", "embedding", "embedding_bag",
    "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d",
    "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool2d",
    "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d",
    "adaptive_max_pool3d", "layer_norm", "batch_norm", "group_norm",
    "instance_norm", "rms_norm", "dropout",
    # padding
    "pad", "constant_pad_nd", "reflection_pad1d", "reflection_pad2d",
    "replication_pad1d", "replication_pad2d", "zero_pad2d",
    # shape
    "flatten", "unflatten", "unfold", "fold",
    # losses
    "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "cross_entropy",
    "nll_loss", "kl_div", "binary_cross_entropy", "binary_cross_entropy_with_logits",
    "margin_ranking_loss", "triplet_margin_loss", "cosine_embedding_loss",
    "ctc_loss", "soft_margin_loss", "poisson_nll_loss",
    # misc
    "interpolate", "pixel_shuffle", "pixel_unshuffle", "channel_shuffle",
    "one_hot", "pad_sequence",
]


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    return input.relu()


def relu_(input: Tensor) -> Tensor:
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    return input.leaky_relu(negative_slope)


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    return input.elu(alpha)


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    return input.selu()


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    return input.gelu()


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    return input.silu()


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    return input.mish()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    if dim is None:
        dim = -1
    return input.softmax(dim)


def log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    if dim is None:
        dim = -1
    return input.log_softmax(dim)


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    return input.softplus(beta, threshold)


def softsign(input: Tensor) -> Tensor:
    return input.softsign()


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    return input.hardswish()


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    return input.hardtanh(min_val, max_val)


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    return input.hardsigmoid()


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    return input.hardshrink(lambd)


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    return input.softshrink(lambd)


def tanhshrink(input: Tensor) -> Tensor:
    return input.tanhshrink()


def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    return input.threshold(threshold, value)


def glu(input: Tensor, dim: int = -1) -> Tensor:
    return input.glu(dim)


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    w = weight.data
    zeros = np.where(input.data >= 0, input.data, input.data * w).astype(input.dtype_name)
    return Tensor(zeros, dtype=input.dtype_name, device=input.device)


# ---------------------------------------------------------------------------
# Linear / conv / pool / norm / embedding
# ---------------------------------------------------------------------------
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return advanced_ops.linear(input, weight, bias)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return advanced_ops.conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    return advanced_ops.conv2d(input, weight, bias, stride, padding, dilation, groups)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    return advanced_ops.conv3d(input, weight, bias, stride, padding, dilation, groups)


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    raise NotImplementedError(
        "conv_transpose1d: use nn.ConvTranspose2d (only 2d transposed conv is implemented); "
        "1d/3d transposed conv not available in this sim"
    )


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)
    return advanced_ops.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    raise NotImplementedError(
        "conv_transpose3d is not implemented; use conv_transpose2d"
    )


def embedding(input: Tensor, weight: Tensor, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False) -> Tensor:
    idx = input.data.astype(np.int64)
    w = weight.data
    out = Tensor.from_numpy(w[idx])
    if max_norm is not None:
        norms = np.linalg.norm(out.data, axis=-1, keepdims=True)
        out = Tensor.from_numpy(out.data * np.minimum(1.0, max_norm / np.maximum(norms, 1e-12)))
    return out


def embedding_bag(input: Tensor, weight: Tensor, offsets: Optional[Tensor] = None,
                  mode: str = "mean", include_last_offset: bool = False,
                  sparse: bool = False) -> Tensor:
    idx = input.data.astype(np.int64).reshape(-1)
    w = weight.data
    emb = w[idx]
    if offsets is None:
        if mode == "sum":
            agg = emb.sum(axis=0, keepdims=True)
        elif mode == "max":
            agg = emb.max(axis=0, keepdims=True)
        else:
            agg = emb.mean(axis=0, keepdims=True)
        return Tensor.from_numpy(agg)
    off = offsets.data.astype(np.int64)
    off = np.clip(off, 0, idx.size)
    n = off.size - 1
    bags = []
    for i in range(n):
        seg = emb[off[i]:off[i + 1]]
        if seg.shape[0] == 0:
            bags.append(np.zeros(emb.shape[1], dtype=np.float32))
        elif mode == "sum":
            bags.append(seg.sum(axis=0))
        elif mode == "max":
            bags.append(seg.max(axis=0))
        else:
            bags.append(seg.mean(axis=0))
    return Tensor.from_numpy(np.stack(bags))


def max_pool1d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
               return_indices=False):
    x = input.data
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    k, = kernel_size
    s, = stride
    p, = padding
    L = x.shape[-1]
    L_out = (L + 2 * p - k) // s + 1
    if p:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode="constant", constant_values=-np.inf)
    out = np.zeros(x.shape[:-1] + (L_out,), dtype=np.float32)
    idx = np.zeros_like(out, dtype=np.int64)
    for i in range(L_out):
        win = x[..., i * s:i * s + k]
        m = win.max(axis=-1)
        out[..., i] = m
        if return_indices:
            idx[..., i] = i * s + win.argmax(axis=-1)
    if return_indices:
        return Tensor.from_numpy(out), Tensor.from_numpy(idx)
    return Tensor.from_numpy(out)


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
               return_indices=False):
    return advanced_ops.max_pool2d(
        input, kernel_size, stride=stride, padding=padding, dilation=dilation,
        return_indices=return_indices,
    )


def max_pool3d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
               return_indices=False):
    x = input.data
    k = _triple(kernel_size)
    s = _triple(stride) if stride is not None else k
    p = _triple(padding)
    kd, kh, kw = k
    sd, sh, sw = s
    pd, ph, pw = p
    if p[0] or p[1] or p[2]:
        x = np.pad(x, [(0, 0)] * (x.ndim - 3) + [(pd, pd), (ph, ph), (pw, pw)],
                   mode="constant", constant_values=-np.inf)
    D, H, W = x.shape[-3:]
    od = (D + 2 * pd - kd) // sd + 1
    oh = (H + 2 * ph - kh) // sh + 1
    ow = (W + 2 * pw - kw) // sw + 1
    out = np.zeros(x.shape[:-3] + (od, oh, ow), dtype=np.float32)
    idx = np.zeros_like(out, dtype=np.int64)
    for i in range(od):
        for j in range(oh):
            for kk in range(ow):
                win = x[..., i * sd:i * sd + kd, j * sh:j * sh + kh, kk * sw:kk * sw + kw]
                flat = win.reshape(win.shape[:-3] + (-1,))
                m = flat.max(axis=-1)
                out[..., i, j, kk] = m
                if return_indices:
                    idx[..., i, j, kk] = flat.argmax(axis=-1)
    if return_indices:
        return Tensor.from_numpy(out), Tensor.from_numpy(idx)
    return Tensor.from_numpy(out)


def avg_pool1d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    k = _single(kernel_size)[0]
    s = (_single(stride)[0]) if stride is not None else k
    p = _single(padding)[0]
    L = x.shape[-1]
    L_out = (L + 2 * p - k) // s + 1
    if p:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode="constant")
    out = np.zeros(x.shape[:-1] + (L_out,), dtype=np.float32)
    for i in range(L_out):
        out[..., i] = x[..., i * s:i * s + k].mean(axis=-1)
    return Tensor.from_numpy(out)


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None) -> Tensor:
    return advanced_ops.avg_pool2d(input, kernel_size, stride=stride, padding=padding)


def avg_pool3d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None) -> Tensor:
    return _pool_nd_mean(input, 3, kernel_size, stride, padding)


def adaptive_avg_pool1d(input: Tensor, output_size) -> Tensor:
    return _adaptive_pool(input, output_size, mode="avg", ndim=1)


def adaptive_avg_pool2d(input: Tensor, output_size) -> Tensor:
    return advanced_ops.adaptive_avg_pool2d(input, output_size)


def adaptive_avg_pool3d(input: Tensor, output_size) -> Tensor:
    return _adaptive_pool(input, output_size, mode="avg", ndim=3)


def adaptive_max_pool1d(input: Tensor, output_size, return_indices=False):
    return _adaptive_pool(input, output_size, mode="max", ndim=1)


def adaptive_max_pool2d(input: Tensor, output_size, return_indices=False) -> Tensor:
    return advanced_ops.adaptive_max_pool2d(input, output_size)


def adaptive_max_pool3d(input: Tensor, output_size, return_indices=False) -> Tensor:
    return _adaptive_pool(input, output_size, mode="max", ndim=3)


def layer_norm(input: Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor:
    return input.layer_norm(weight, bias, eps)


def rms_norm(input: Tensor, normalized_shape, weight=None, eps=1e-6) -> Tensor:
    return input.layer_norm(weight, None, eps)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False,
               momentum=0.1, eps=1e-5):
    return input.batch_norm(weight, bias, running_mean, running_var, eps)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    return input.group_norm(weight, bias, num_groups, eps)


def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    shape = x.shape
    n, c = shape[:2]
    sp = shape[2:] or (1,)
    xr = x.reshape(n, c, -1)
    mean = xr.mean(axis=2, keepdims=True)
    var = xr.var(axis=2, keepdims=True)
    xn = (xr - mean) / np.sqrt(var + eps)
    if weight is not None:
        xn = xn * np.asarray(weight.data).reshape(1, c, 1)
    if bias is not None:
        xn = xn + np.asarray(bias.data).reshape(1, c, 1)
    return Tensor.from_numpy(xn.reshape(shape).astype(np.float32))


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if not training or p == 0:
        return input
    return input.dropout(p)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------
def pad(input: Tensor, pad: Tuple[int, ...], mode: str = "constant", value: Optional[float] = None) -> Tensor:
    from .nn import _ConstantPadNd
    m = _pad_mode_to_module(mode, len(pad) // 2)
    if mode == "constant" and value is not None:
        cls = _pad_mode_to_module("constant", len(pad) // 2)
        inst = cls(pad)
        inst.value = value
        return inst(input)
    inst = _pad_mode_to_module(mode, len(pad) // 2)(pad)
    return inst(input)


def _pad_mode_to_module(mode: str, ndim: int):
    from .nn import (ConstantPad1d, ConstantPad2d, ConstantPad3d,
                     ReflectionPad1d, ReflectionPad2d, ReflectionPad3d,
                     ReplicationPad1d, ReplicationPad2d, ReplicationPad3d)
    const = {1: ConstantPad1d, 2: ConstantPad2d, 3: ConstantPad3d}
    refl = {1: ReflectionPad1d, 2: ReflectionPad2d, 3: ReflectionPad3d}
    repl = {1: ReplicationPad1d, 2: ReplicationPad2d, 3: ReplicationPad3d}
    if mode == "constant":
        return const[ndim]
    if mode in ("reflect", "reflection"):
        return refl[ndim]
    if mode in ("replicate", "edge", "replication"):
        return repl[ndim]
    raise ValueError(f"pad mode {mode!r} not supported")


def constant_pad_nd(input: Tensor, padding, value: float = 0.0) -> Tensor:
    return pad(input, tuple(padding) if not isinstance(padding, int) else (padding,) * 2, mode="constant", value=value)


def reflection_pad1d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("reflect", 1)(padding)(input)


def reflection_pad2d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("reflect", 2)(padding)(input)


def reflection_pad3d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("reflect", 3)(padding)(input)


def replication_pad1d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("replicate", 1)(padding)(input)


def replication_pad2d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("replicate", 2)(padding)(input)


def replication_pad3d(input: Tensor, padding) -> Tensor:
    return _pad_mode_to_module("replicate", 3)(padding)(input)


def zero_pad2d(input: Tensor, padding) -> Tensor:
    return constant_pad_nd(input, padding, 0.0)


# ---------------------------------------------------------------------------
# Shape transforms
# ---------------------------------------------------------------------------
def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    shape = input.shape
    ndim = len(shape)
    if end_dim < 0:
        end_dim = ndim + end_dim
    if start_dim < 0:
        start_dim = ndim + start_dim
    if start_dim == end_dim:
        return input
    pre = int(np.prod(shape[:start_dim])) if start_dim > 0 else 1
    mid = int(np.prod(shape[start_dim:end_dim + 1]))
    post = int(np.prod(shape[end_dim + 1:])) if end_dim + 1 < ndim else 1
    if post != 1:
        return input.reshape(pre, mid, post)
    return input.reshape(pre, mid)


def unflatten(input: Tensor, dim: int, sizes) -> Tensor:
    sizes = tuple(sizes)
    shape = list(input.shape)
    dim = dim if dim >= 0 else input.ndim + dim
    shape[dim:dim + 1] = list(sizes)
    return input.reshape(*shape)


def unfold(input: Tensor, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    from .nn import Unfold
    return Unfold(kernel_size, dilation, padding, stride)(input)


def fold(input: Tensor, output_size, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    from .nn import Fold
    return Fold(output_size, kernel_size, dilation, padding, stride)(input)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return input.mse_loss(target)


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return input.mae_loss(target)


def smooth_l1_loss(input: Tensor, target: Tensor, reduction: str = "mean", beta: float = 1.0) -> Tensor:
    return input.smooth_l1_loss(target, beta)


def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0) -> Tensor:
    return input.huber_loss(target, delta)


def cross_entropy(input: Tensor, target: Tensor, weight=None, ignore_index=-100,
                  reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    return input.cross_entropy(target)


def nll_loss(input: Tensor, target: Tensor, weight=None, ignore_index=-100,
             reduction: str = "mean") -> Tensor:
    return input.nll_loss(target)


def kl_div(input: Tensor, target: Tensor, reduction: str = "mean", log_target: bool = False) -> Tensor:
    return input.kl_div(target)


def binary_cross_entropy(input: Tensor, target: Tensor, weight=None,
                         reduction: str = "mean") -> Tensor:
    return input.bce_loss(target)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor, weight=None,
                                     pos_weight=None, reduction: str = "mean") -> Tensor:
    return input.bce_with_logits_loss(target)


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor,
                        margin: float = 0.0, reduction: str = "mean") -> Tensor:
    return input.margin_ranking_loss(input2, target, margin)


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor,
                        margin: float = 1.0, p: float = 2.0, eps: float = 1e-6,
                        swap: bool = False, reduction: str = "mean") -> Tensor:
    return anchor.triplet_margin_loss(positive, negative, margin)


def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor,
                          margin: float = 0.0, reduction: str = "mean") -> Tensor:
    return input1.cosine_embedding_loss(input2, target, margin)


def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths, target_lengths,
             blank: int = 0, reduction: str = "mean", zero_infinity: bool = False) -> Tensor:
    return log_probs.ctc_loss(targets, input_lengths, target_lengths, blank, reduction)


def soft_margin_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    a = input.data.astype(np.float64)
    t = target.data.astype(np.float64)
    losses = np.log1p(np.exp(-a * t))
    val = _reduce(losses, reduction)
    return Tensor(val, dtype=input.dtype_name, device=input.device)


def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = True,
                     full: bool = False, eps: float = 1e-8, reduction: str = "mean") -> Tensor:
    a = input.data.astype(np.float64)
    t = target.data.astype(np.float64)
    if log_input:
        losses = np.exp(a) - t * a
    else:
        losses = a - t * np.log(a + eps)
    val = _reduce(losses, reduction)
    return Tensor(val, dtype=input.dtype_name, device=input.device)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def interpolate(input: Tensor, size=None, scale_factor=None, mode: str = "nearest",
                align_corners=None, recompute_scale_factor=None) -> Tensor:
    from .nn import Upsample
    return Upsample(size=size, scale_factor=scale_factor, mode=mode)(input)


def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    from .nn import PixelShuffle
    return PixelShuffle(upscale_factor)(input)


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    from .nn import PixelUnshuffle
    return PixelUnshuffle(downscale_factor)(input)


def channel_shuffle(input: Tensor, groups: int) -> Tensor:
    from .nn import ChannelShuffle
    return ChannelShuffle(groups)(input)


def one_hot(tensor: Tensor, num_classes: int = -1) -> Tensor:
    idx = tensor.data.astype(np.int64).reshape(-1)
    n = idx.size
    if num_classes < 0:
        num_classes = int(idx.max() + 1)
    out = np.zeros((n, num_classes), dtype=np.float32)
    out[np.arange(n), idx] = 1.0
    return Tensor.from_numpy(out.reshape(tensor.shape + (num_classes,)))


def pad_sequence(sequences, batch_first: bool = False, padding_value: float = 0.0) -> Tensor:
    if isinstance(sequences, Tensor):
        return sequences
    max_len = max(int(s.shape[0]) for s in sequences)
    out = np.full((len(sequences), max_len) + tuple(sequences[0].shape[1:]), padding_value,
                  dtype=sequences[0].data.dtype)
    for i, s in enumerate(sequences):
        n = int(s.shape[0])
        out[i, :n] = np.asarray(s.data)
    outT = out
    if batch_first:
        outT = out
    else:
        outT = out.transpose(1, 0, *range(2, out.ndim))
    return Tensor.from_numpy(outT)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _single(x):
    return (x,) if isinstance(x, int) else tuple(x)


def _triple(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


def _reduce(x, reduction: str):
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    return x.mean()


def _pool_nd_mean(input, ndim, kernel_size, stride, padding):
    x = input.data
    k = kernel_size
    k = (k,) * ndim if isinstance(k, int) else tuple(k)
    s = (k,) if stride is None else ((stride,) * ndim if isinstance(stride, int) else tuple(stride))
    p = ((padding,) * ndim if isinstance(padding, int) else tuple(padding))
    pieces = x.shape[:-ndim]
    if p[0]:
        padw = [(0, 0)] * (len(x.shape) - ndim) + [(p[i], p[i]) for i in range(ndim)]
        x = np.pad(x, padw, mode="constant")
    out_sh = pieces + tuple((x.shape[-ndim + i] - k[i]) // s[i] + 1 for i in range(ndim))
    out = np.zeros(out_sh, dtype=np.float32)
    iters = [range(o) for o in out_sh[-ndim:]]
    for oi in np.ndindex(*[range(o) for o in out_sh[-ndim:]]):
        slc = pieces + tuple(
            slice(oi[i] * s[i], oi[i] * s[i] + k[i]) for i in range(ndim)
        )
        out[(...,) + oi] = x[slc].mean()
    return Tensor.from_numpy(out)


def _adaptive_pool(input, output_size, mode="avg", ndim=1):
    x = np.asarray(input.data, dtype=np.float64)
    od = output_size
    od = (od,) * ndim if isinstance(od, int) else tuple(od)
    pieces = x.shape[:-ndim]
    inp_sp = x.shape[-ndim:]
    out = np.zeros(pieces + tuple(od), dtype=np.float32)
    for target in np.ndindex(*od):
        lo = []
        hi = []
        for d, (o, s) in enumerate(zip(od, inp_sp)):
            lo.append(int(np.floor(target[d] * s / o)))
            hi.append(int(np.ceil((target[d] + 1) * s / o)))
        slc = pieces + tuple(slice(lo[i], max(hi[i], lo[i] + 1)) for i in range(ndim))
        win = x[slc]
        out[(...,) + target] = win.max() if mode == "max" else win.mean()
    return Tensor.from_numpy(out)