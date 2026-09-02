"""Advanced tensor operations — convolution, pooling, RNN, transformer blocks."""

from typing import Optional, Tuple, List, Union, Callable
import numpy as np

from .tensor import Tensor, Dtype


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """2D convolution with optional bias.

    Args:
        input: [N, C_in, H, W] or [N, H, W, C_in]
        weight: [C_out, C_in // groups, kH, kW]
        bias: [C_out] optional
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
        dilation: (dil_h, dil_w)
        groups: number of groups

    Returns:
        output: [N, C_out, H_out, W_out]
    """
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)

    N, C_in, H, W = x.shape
    C_out, C_in_g, kH, kW = w.shape

    assert C_in == C_in_g * groups, "Input channels must match weight channels * groups"

    # Calculate output dimensions
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    stride_h, stride_w = stride

    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    # Pad input
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # im2col
    col = np.zeros((N, C_in * kH * kW, H_out * W_out), dtype=np.float64)
    for i in range(H_out):
        h_start = i * stride_h
        for j in range(W_out):
            w_start = j * stride_w
            col[:, :, i * W_out + j] = x[
                :,
                :,
                h_start : h_start + kH * dil_h : dil_h,
                w_start : w_start + kW * dil_w : dil_w,
            ].reshape(N, -1)

    # Weight matrix
    w_mat = w.reshape(C_out, -1)

    # Convolution via GEMM
    out = w_mat @ col  # [C_out, N * H_out * W_out]
    out = out.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)

    if bias is not None:
        b = np.asarray(bias.data, dtype=np.float64)
        out += b.reshape(1, -1, 1, 1)

    return Tensor.from_numpy(out.astype(np.float32), dtype="float32")


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """1D convolution."""
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)

    N, C_in, L = x.shape
    C_out, C_in_g, kL = w.shape
    assert C_in == C_in_g * groups

    L_out = (L + 2 * padding - dilation * (kL - 1) - 1) // stride + 1

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    col = np.zeros((N, C_in * kL, L_out), dtype=np.float64)
    for i in range(L_out):
        l_start = i * stride
        col[:, :, i] = x[:, :, l_start : l_start + kL * dilation : dilation].reshape(
            N, -1
        )

    w_mat = w.reshape(C_out, -1)
    out = w_mat @ col
    out = out.reshape(C_out, N, L_out).transpose(1, 0, 2)

    if bias is not None:
        b = np.asarray(bias.data, dtype=np.float64)
        out += b.reshape(1, -1, 1)

    return Tensor.from_numpy(out.astype(np.float32), dtype="float32")


def max_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """2D max pooling."""
    x = np.asarray(input.data, dtype=np.float64)
    N, C, H, W = x.shape

    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size

    if stride is None:
        stride_h = kH
        stride_w = kW
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    if pad_h > 0 or pad_w > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=-np.inf,
        )

    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)

    for i in range(H_out):
        h_start = i * stride_h
        for j in range(W_out):
            w_start = j * stride_w
            window = x[
                :,
                :,
                h_start : h_start + kH * dil_h : dil_h,
                w_start : w_start + kW * dil_w : dil_w,
            ]
            max_vals = np.max(window, axis=(2, 3))
            out[:, :, i, j] = max_vals
            # Find indices (simplified)
            idx = np.argmax(window.reshape(N, C, -1), axis=-1)
            indices[:, :, i, j, 0] = idx // kW
            indices[:, :, i, j, 1] = idx % kW

    if return_indices:
        return Tensor.from_numpy(out.astype(np.float32)), Tensor.from_numpy(indices)
    return Tensor.from_numpy(out.astype(np.float32))


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    count_include_pad: bool = True,
) -> Tensor:
    """2D average pooling."""
    x = np.asarray(input.data, dtype=np.float64)
    N, C, H, W = x.shape

    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size

    if stride is None:
        stride_h = kH
        stride_w = kW
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        h_start = i * stride_h
        for j in range(W_out):
            w_start = j * stride_w
            window = x[:, :, h_start : h_start + kH, w_start : w_start + kW]
            if count_include_pad:
                out[:, :, i, j] = np.mean(window, axis=(2, 3))
            else:
                # Only count non-padded elements
                valid = window != 0  # simplified
                out[:, :, i, j] = np.sum(window, axis=(2, 3)) / np.maximum(
                    1, np.sum(valid, axis=(2, 3))
                )

    return Tensor.from_numpy(out.astype(np.float32))


def adaptive_avg_pool2d(input: Tensor, output_size: Tuple[int, int]) -> Tensor:
    """Adaptive average pooling to fixed output size."""
    x = np.asarray(input.data, dtype=np.float64)
    N, C, H, W = x.shape
    H_out, W_out = output_size

    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        h_start = i * H // H_out
        h_end = (i + 1) * H // H_out
        for j in range(W_out):
            w_start = j * W // W_out
            w_end = (j + 1) * W // W_out
            out[:, :, i, j] = np.mean(
                x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)
            )

    return Tensor.from_numpy(out.astype(np.float32))


def adaptive_max_pool2d(input: Tensor, output_size: Tuple[int, int]) -> Tensor:
    """Adaptive max pooling to fixed output size."""
    x = np.asarray(input.data, dtype=np.float64)
    N, C, H, W = x.shape
    H_out, W_out = output_size

    out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        h_start = i * H // H_out
        h_end = (i + 1) * H // H_out
        for j in range(W_out):
            w_start = j * W // W_out
            w_end = (j + 1) * W // W_out
            out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

    return Tensor.from_numpy(out.astype(np.float32))


def rnn_cell(
    input: Tensor,
    hx: Optional[Tensor],
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
    nonlinearity: str = "tanh",
) -> Tensor:
    """Simple RNN cell: h = activation(x @ W_ih + b_ih + h_prev @ W_hh + b_hh)"""
    x = np.asarray(input.data, dtype=np.float64)
    W_ih = np.asarray(weight_ih.data, dtype=np.float64)
    W_hh = np.asarray(weight_hh.data, dtype=np.float64)

    if hx is not None:
        h_prev = np.asarray(hx.data, dtype=np.float64)
    else:
        h_prev = np.zeros((x.shape[0], W_hh.shape[0]), dtype=np.float64)

    b_ih = np.asarray(bias_ih.data, dtype=np.float64) if bias_ih is not None else 0
    b_hh = np.asarray(bias_hh.data, dtype=np.float64) if bias_hh is not None else 0

    h = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh

    if nonlinearity == "tanh":
        h = np.tanh(h)
    elif nonlinearity == "relu":
        h = np.maximum(h, 0)

    return Tensor.from_numpy(h.astype(np.float32))


def lstm_cell(
    input: Tensor,
    hx: Optional[Tensor],
    cx: Optional[Tensor],
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """LSTM cell with all gates."""
    x = np.asarray(input.data, dtype=np.float64)
    W_ih = np.asarray(weight_ih.data, dtype=np.float64)
    W_hh = np.asarray(weight_hh.data, dtype=np.float64)

    h_prev = (
        np.asarray(hx.data, dtype=np.float64)
        if hx is not None
        else np.zeros((x.shape[0], W_hh.shape[0] // 4), dtype=np.float64)
    )
    c_prev = (
        np.asarray(cx.data, dtype=np.float64)
        if cx is not None
        else np.zeros_like(h_prev)
    )

    b_ih = np.asarray(bias_ih.data, dtype=np.float64) if bias_ih is not None else 0
    b_hh = np.asarray(bias_hh.data, dtype=np.float64) if bias_hh is not None else 0

    gates = x @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh
    h_size = gates.shape[1] // 4
    i = sigmoid(gates[:, :h_size])
    f = sigmoid(gates[:, h_size : 2 * h_size])
    g = np.tanh(gates[:, 2 * h_size : 3 * h_size])
    o = sigmoid(gates[:, 3 * h_size :])

    c = f * c_prev + i * g
    h = o * np.tanh(c)

    return Tensor.from_numpy(h.astype(np.float32)), Tensor.from_numpy(
        c.astype(np.float32)
    )


def gru_cell(
    input: Tensor,
    hx: Optional[Tensor],
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
) -> Tensor:
    """GRU cell."""
    x = np.asarray(input.data, dtype=np.float64)
    W_ih = np.asarray(weight_ih.data, dtype=np.float64)
    W_hh = np.asarray(weight_hh.data, dtype=np.float64)

    h_prev = (
        np.asarray(hx.data, dtype=np.float64)
        if hx is not None
        else np.zeros((x.shape[0], W_hh.shape[0] // 3), dtype=np.float64)
    )

    b_ih = np.asarray(bias_ih.data, dtype=np.float64) if bias_ih is not None else 0
    b_hh = np.asarray(bias_hh.data, dtype=np.float64) if bias_hh is not None else 0

    gates_ih = x @ W_ih.T + b_ih
    gates_hh = h_prev @ W_hh.T + b_hh
    h_size = gates_ih.shape[1] // 3

    r = sigmoid(gates_ih[:, :h_size] + gates_hh[:, :h_size])
    z = sigmoid(gates_ih[:, h_size : 2 * h_size] + gates_hh[:, h_size : 2 * h_size])
    n = np.tanh(gates_ih[:, 2 * h_size :] + r * gates_hh[:, 2 * h_size :])

    h = (1 - z) * n + z * h_prev
    return Tensor.from_numpy(h.astype(np.float32))


def multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tensor:
    """Multi-head attention."""
    _to_arr = lambda t: np.asarray(t.data, dtype=np.float64) if hasattr(t, "data") else np.asarray(t, dtype=np.float64)
    q = _to_arr(query)
    k = _to_arr(key)
    v = _to_arr(value)

    N, L_q, D = q.shape
    _, L_k, _ = k.shape
    _, L_v, _ = v.shape

    head_dim = D // num_heads
    assert D % num_heads == 0

    q = q.reshape(N, L_q, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(N, L_k, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(N, L_v, num_heads, head_dim).transpose(0, 2, 1, 3)

    scale = head_dim**-0.5
    attn = q @ k.transpose(0, 1, 3, 2) * scale

    if mask is not None:
        m = _to_arr(mask)
        attn = attn + m.reshape(N, 1, 1, -1) * -1e9

    attn = softmax(attn, axis=-1)

    if training and dropout_p > 0:
        mask = np.random.rand(*attn.shape) > dropout_p
        attn = attn * mask / (1 - dropout_p)

    out = attn @ v  # [N, num_heads, L_q, head_dim]
    out = out.transpose(0, 2, 1, 3).reshape(N, L_q, D)

    return Tensor.from_numpy(out.astype(np.float32))


def transformer_block(
    x: Tensor,
    attn_weight: Tensor,
    attn_bias: Optional[Tensor],
    ff1_weight: Tensor,
    ff1_bias: Optional[Tensor],
    ff2_weight: Tensor,
    ff2_bias: Optional[Tensor],
    ln1_weight: Tensor,
    ln1_bias: Optional[Tensor],
    ln2_weight: Tensor,
    ln2_bias: Optional[Tensor],
    num_heads: int = 8,
    dropout: float = 0.1,
    training: bool = True,
) -> Tensor:
    """Transformer encoder block: attention + FFN with residual connections."""
    # Self-attention
    residual = x
    x_ln = layernorm(x, ln1_weight, ln1_bias)
    attn_out = multi_head_attention(
        x_ln, x_ln, x_ln, num_heads, dropout_p=dropout, training=training
    )
    x = residual + attn_out

    # FFN
    residual = x
    x_ln = layernorm(x, ln2_weight, ln2_bias)
    ff1 = linear(x_ln, ff1_weight, ff1_bias)
    x_act = gelu(ff1)
    ff2 = linear(x_act, ff2_weight, ff2_bias)
    x = residual + ff2

    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def topk(x: Tensor, k: int, dim: int = -1):
    """Return the top-k values and indices along ``dim`` (descending)."""
    data = np.asarray(x.data, dtype=np.float32)
    axis = dim if dim >= 0 else data.ndim + dim
    idx = np.argpartition(-data, k - 1, axis=axis)[..., :k]
    vals = np.take_along_axis(data, idx, axis=axis)
    order = np.argsort(-vals, axis=axis)
    idx = np.take_along_axis(idx, order, axis=axis)
    vals = np.take_along_axis(vals, order, axis=axis)
    return Tensor.from_numpy(vals.astype(np.float32)), Tensor.from_numpy(
        idx.astype(np.int64)
    )


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Negative log-likelihood of targets given logits (mean-reduced)."""
    l = np.asarray(logits.data, dtype=np.float32)
    t = np.asarray(targets.data if hasattr(targets, "data") else targets)
    orig_shape = t.shape
    l = l.reshape(-1, l.shape[-1])
    t = t.reshape(-1).astype(np.int64)
    lmax = l.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(l - lmax).sum(axis=-1, keepdims=True)) + lmax
    logp = l - logsumexp
    loss = -logp[np.arange(l.shape[0]), t]
    return Tensor.from_numpy(loss.reshape(orig_shape).astype(np.float32))


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log softmax with numerical stability."""
    x = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
    return x - log_sum_exp


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """KL divergence D_KL(P || Q) = sum P * log(P/Q)."""
    return np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=axis)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply rotary position embedding."""
    # x: [..., seq_len, head_dim]
    # cos, sin: [..., seq_len, head_dim]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def apply_rotary_pos_emb(
    q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> tuple:
    """Apply rotary position embedding to q and k."""
    q_embed = rope(q, cos, sin)
    k_embed = rope(k, cos, sin)
    return q_embed, k_embed


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layernorm(
    input: Tensor, weight: Tensor, bias: Optional[Tensor], eps: float = 1e-5
) -> Tensor:
    """Layer normalization."""
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)
    b = np.asarray(bias.data, dtype=np.float64) if bias is not None else 0

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = x_norm * w + b

    return Tensor.from_numpy(out.astype(np.float32))


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Linear transformation: y = x @ W^T + b"""
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)

    out = x @ w.T
    if bias is not None:
        b = np.asarray(bias.data, dtype=np.float64)
        out = out + b

    return Tensor.from_numpy(out.astype(np.float32))


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout."""
    if not training or p == 0:
        return input
    x = np.asarray(input.data, dtype=np.float64)
    mask = np.random.rand(*x.shape) > p
    out = x * mask / (1 - p)
    return Tensor.from_numpy(out.astype(np.float32))


def embedding(
    input: Tensor, weight: Tensor, padding_idx: Optional[int] = None
) -> Tensor:
    """Embedding lookup."""
    indices = np.asarray(input.data, dtype=np.int64)
    w = np.asarray(weight.data, dtype=np.float64)
    out = w[indices]
    return Tensor.from_numpy(out.astype(np.float32))


def batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalization."""
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)
    b = np.asarray(bias.data, dtype=np.float64)

    if training:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        if running_mean is not None:
            running_mean.data = (1 - momentum) * np.asarray(
                running_mean.data
            ) + momentum * mean.squeeze()
            running_var.data = (1 - momentum) * np.asarray(
                running_var.data
            ) + momentum * var.squeeze()
    else:
        mean = np.asarray(running_mean.data).reshape(1, -1, 1, 1) if running_mean else 0
        var = np.asarray(running_var.data).reshape(1, -1, 1, 1) if running_var else 1

    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    out = x_norm * w.reshape(1, -1, 1, 1) + b.reshape(1, -1, 1, 1)

    return Tensor.from_numpy(out.astype(np.float32))


def group_norm(
    input: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    """Group normalization."""
    x = np.asarray(input.data, dtype=np.float64)
    N, C, H, W = x.shape
    G = num_groups
    assert C % G == 0

    x = x.reshape(N, G, C // G, H, W)
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    x = x.reshape(N, C, H, W)

    w = np.asarray(weight.data, dtype=np.float64).reshape(1, C, 1, 1)
    b = np.asarray(bias.data, dtype=np.float64).reshape(1, C, 1, 1)
    out = x * w + b

    return Tensor.from_numpy(out.astype(np.float32))


def instance_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Instance normalization."""
    return group_norm(input, input.shape[1], weight, bias, eps)


def layer_norm(
    input: Tensor,
    normalized_shape_or_weight,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalization over the last dimension.

    Supports both calling conventions used across the codebase:
        layer_norm(x, weight, bias)                 # last-dim normalize
        layer_norm(x, normalized_shape, weight, bias)
    """
    if isinstance(normalized_shape_or_weight, (tuple, list)):
        w, b = weight, bias
    else:
        w, b = normalized_shape_or_weight, weight

    x = np.asarray(input.data, dtype=np.float64)
    ww = np.asarray(w.data, dtype=np.float64) if w is not None else 1.0
    bb = np.asarray(b.data, dtype=np.float64) if b is not None else 0.0

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = x_norm * ww + bb

    return Tensor.from_numpy(out.astype(np.float32))


def rmsnorm(input: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMS normalization (Root Mean Square Layer Normalization)."""
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)

    # Compute RMS
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    out = x_norm * w

    return Tensor.from_numpy(out.astype(np.float32))


# =========================================================================
# Tensor Operations Helpers
# =========================================================================


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along a dimension."""
    arrays = [np.asarray(t.data, dtype=np.float64) for t in tensors]
    out = np.concatenate(arrays, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along a new dimension."""
    arrays = [np.asarray(t.data, dtype=np.float64) for t in tensors]
    out = np.stack(arrays, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def split(
    tensor: Tensor, split_size: Union[int, List[int]], dim: int = 0
) -> List[Tensor]:
    """Split tensor into chunks."""
    x = np.asarray(tensor.data, dtype=np.float64)
    if isinstance(split_size, int):
        chunks = np.split(x, x.shape[dim] // split_size, axis=dim)
    else:
        # cumulative sum of split sizes
        split_points = np.cumsum(split_size)[:-1]
        chunks = np.split(x, split_points, axis=dim)
    return [Tensor.from_numpy(c.astype(np.float32)) for c in chunks]


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    """Split tensor into a specific number of chunks."""
    return split(tensor, tensor.shape[dim] // chunks, dim)


def repeat(tensor: Tensor, repeats: Union[int, List[int]]) -> Tensor:
    """Repeat tensor elements."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.repeat(x, repeats)
    return Tensor.from_numpy(out.astype(np.float32))


def tile(tensor: Tensor, dims: Tuple[int, ...]) -> Tensor:
    """Tile (repeat) tensor along specified dimensions."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.tile(x, dims)
    return Tensor.from_numpy(out.astype(np.float32))


def expand(tensor: Tensor, size: Tuple[int, ...]) -> Tensor:
    """Expand tensor to larger size (no copy)."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.broadcast_to(x, size)
    return Tensor.from_numpy(out.astype(np.float32))


def reshape(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape tensor."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = x.reshape(shape)
    return Tensor.from_numpy(out.astype(np.float32))


def view(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """View tensor with new shape (must be contiguous)."""
    return reshape(tensor, shape)


def permute(tensor: Tensor, dims: Tuple[int, ...]) -> Tensor:
    """Permute dimensions."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.transpose(x, dims)
    return Tensor.from_numpy(out.astype(np.float32))


def transpose(tensor: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swap two dimensions."""
    dims = list(range(tensor.ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return permute(tensor, tuple(dims))


def squeeze(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1."""
    x = np.asarray(tensor.data, dtype=np.float64)
    if dim is None:
        out = np.squeeze(x)
    else:
        out = np.squeeze(x, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.expand_dims(x, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def flatten(tensor: Tensor, start_dim: int = 1, end_dim: int = -1) -> Tensor:
    """Flatten a tensor from start_dim to end_dim."""
    x = np.asarray(tensor.data, dtype=np.float64)
    if end_dim < 0:
        end_dim = x.ndim + end_dim + 1
    shape = x.shape
    new_shape = (
        shape[:start_dim]
        + (np.prod(shape[start_dim : end_dim + 1]),)
        + shape[end_dim + 1 :]
    )
    out = x.reshape(new_shape)
    return Tensor.from_numpy(out.astype(np.float32))


def pad(
    input: Tensor, pad: Tuple[int, ...], mode: str = "constant", value: float = 0.0
) -> Tensor:
    """Pad tensor."""
    x = np.asarray(input.data, dtype=np.float64)
    # pad format: (pad_left, pad_right, pad_top, pad_bottom, ...) for each dim reversed
    pad_width = []
    for i in range(len(pad) // 2):
        pad_width.insert(0, (pad[2 * i], pad[2 * i + 1]))
    out = np.pad(x, pad_width, mode=mode, constant_values=value)
    return Tensor.from_numpy(out.astype(np.float32))


def flip(tensor: Tensor, dims: List[int]) -> Tensor:
    """Reverse tensor along specified dimensions."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.flip(x, axis=dims)
    return Tensor.from_numpy(out.astype(np.float32))


def rot90(tensor: Tensor, k: int = 1, dims: Tuple[int, int] = (0, 1)) -> Tensor:
    """Rotate tensor 90 degrees."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.rot90(x, k=k, axes=dims)
    return Tensor.from_numpy(out.astype(np.float32))


def roll(
    tensor: Tensor,
    shifts: Union[int, List[int]],
    dims: Optional[Union[int, List[int]]] = None,
) -> Tensor:
    """Roll tensor elements."""
    x = np.asarray(tensor.data, dtype=np.float64)
    out = np.roll(x, shifts, axis=dims)
    return Tensor.from_numpy(out.astype(np.float32))


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values along an axis."""
    x = np.asarray(input.data, dtype=np.float64)
    idx = np.asarray(index.data, dtype=np.int64)
    out = np.take_along_axis(x, idx, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def scatter(
    input: Tensor, dim: int, index: Tensor, src: Union[Tensor, float]
) -> Tensor:
    """Scatter values from src into input at indices."""
    x = np.asarray(input.data, dtype=np.float64).copy()
    idx = np.asarray(index.data, dtype=np.int64)
    if isinstance(src, Tensor):
        src_arr = np.asarray(src.data, dtype=np.float64)
    else:
        src_arr = float(src)
    np.put_along_axis(x, idx, src_arr, axis=dim)
    return Tensor.from_numpy(x.astype(np.float32))


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Scatter add."""
    x = np.asarray(input.data, dtype=np.float64).copy()
    idx = np.asarray(index.data, dtype=np.int64)
    src_arr = np.asarray(src.data, dtype=np.float64)
    np.add.at(x, idx, src_arr)
    return Tensor.from_numpy(x.astype(np.float32))


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Select indices along a dimension."""
    x = np.asarray(input.data, dtype=np.float64)
    idx = np.asarray(index.data, dtype=np.int64)
    out = np.take_along_axis(x, idx.reshape([-1] + [1] * (x.ndim - 1)), axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    """Select elements where mask is True."""
    x = np.asarray(input.data, dtype=np.float64)
    m = np.asarray(mask.data, dtype=bool)
    out = x[m]
    return Tensor.from_numpy(out.astype(np.float32))


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Select from x or y based on condition."""
    c = np.asarray(condition.data, dtype=bool)
    x_arr = np.asarray(x.data, dtype=np.float64)
    y_arr = np.asarray(y.data, dtype=np.float64)
    out = np.where(c, x_arr, y_arr)
    return Tensor.from_numpy(out.astype(np.float32))


def clamp(
    input: Tensor, min: Optional[float] = None, max: Optional[float] = None
) -> Tensor:
    """Clamp tensor values."""
    x = np.asarray(input.data, dtype=np.float64)
    out = np.clip(x, min, max)
    return Tensor.from_numpy(out.astype(np.float32))


def clamp_(
    input: Tensor, min: Optional[float] = None, max: Optional[float] = None
) -> Tensor:
    """In-place clamp."""
    return clamp(input, min, max)


# =========================================================================
# Reduction Operations
# =========================================================================


def sum(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = np.sum(x, axis=dim, keepdims=keepdim)
    return Tensor.from_numpy(out.astype(np.float32))


def mean(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = np.mean(x, axis=dim, keepdims=keepdim)
    return Tensor.from_numpy(out.astype(np.float32))


def var(
    input: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    ddof = 1 if unbiased else 0
    out = np.var(x, axis=dim, keepdims=keepdim, ddof=ddof)
    return Tensor.from_numpy(out.astype(np.float32))


def std(
    input: Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    ddof = 1 if unbiased else 0
    out = np.std(x, axis=dim, keepdims=keepdim, ddof=ddof)
    return Tensor.from_numpy(out.astype(np.float32))


def max(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    x = np.asarray(input.data, dtype=np.float64)
    if dim is None:
        out = np.max(x)
        return Tensor.from_numpy(np.array(out, dtype=np.float32))
    values = np.max(x, axis=dim, keepdims=keepdim)
    indices = np.argmax(x, axis=dim)
    if not keepdim:
        indices = np.expand_dims(indices, axis=dim)
    return Tensor.from_numpy(values.astype(np.float32)), Tensor.from_numpy(
        indices.astype(np.int64)
    )


def min(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    x = np.asarray(input.data, dtype=np.float64)
    if dim is None:
        out = np.min(x)
        return Tensor.from_numpy(np.array(out, dtype=np.float32))
    values = np.min(x, axis=dim, keepdims=keepdim)
    indices = np.argmin(x, axis=dim)
    if not keepdim:
        indices = np.expand_dims(indices, axis=dim)
    return Tensor.from_numpy(values.astype(np.float32)), Tensor.from_numpy(
        indices.astype(np.int64)
    )


def prod(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = np.prod(x, axis=dim, keepdims=keepdim)
    return Tensor.from_numpy(out.astype(np.float32))


def cumsum(input: Tensor, dim: int) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = np.cumsum(x, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def cumprod(input: Tensor, dim: int) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = np.cumprod(x, axis=dim)
    return Tensor.from_numpy(out.astype(np.float32))


def norm(
    input: Tensor, p: float = 2, dim: Optional[int] = None, keepdim: bool = False
) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    if p == 1:
        out = np.sum(np.abs(x), axis=dim, keepdims=keepdim)
    elif p == 2:
        out = np.sqrt(np.sum(x**2, axis=dim, keepdims=keepdim))
    elif p == np.inf:
        out = np.max(np.abs(x), axis=dim, keepdims=keepdim)
    elif p == -np.inf:
        out = np.min(np.abs(x), axis=dim, keepdims=keepdim)
    else:
        out = np.sum(np.abs(x) ** p, axis=dim, keepdims=keepdim) ** (1 / p)
    return Tensor.from_numpy(out.astype(np.float32))


# =========================================================================
# Pointwise Operations
# =========================================================================


def add(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: x + y)


def sub(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: x - y)


def mul(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: x * y)


def div(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: x / y)


def pow(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: x**y)


def eq(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x == y).astype(np.float32))


def ne(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x != y).astype(np.float32))


def lt(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x < y).astype(np.float32))


def le(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x <= y).astype(np.float32))


def gt(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x > y).astype(np.float32))


def ge(a: Tensor, b: Union[Tensor, float]) -> Tensor:
    return binary_op(a, b, lambda x, y: (x >= y).astype(np.float32))


def binary_op(a: Tensor, b: Union[Tensor, float], op) -> Tensor:
    a_arr = np.asarray(a.data, dtype=np.float64)
    if isinstance(b, Tensor):
        b_arr = np.asarray(b.data, dtype=np.float64)
    else:
        b_arr = np.array(b, dtype=np.float64)
    out = op(a_arr, b_arr)
    return Tensor.from_numpy(out.astype(np.float32))


def abs(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.abs(x).astype(np.float32))


def neg(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy((-x).astype(np.float32))


def sign(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.sign(x).astype(np.float32))


def sqrt(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.sqrt(x).astype(np.float32))


def rsqrt(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy((1 / np.sqrt(x)).astype(np.float32))


def exp(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.exp(x).astype(np.float32))


def log(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.log(x).astype(np.float32))


def log1p(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.log1p(x).astype(np.float32))


def sin(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.sin(x).astype(np.float32))


def cos(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.cos(x).astype(np.float32))


def tan(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.tan(x).astype(np.float32))


def asin(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.arcsin(x).astype(np.float32))


def acos(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.arccos(x).astype(np.float32))


def atan(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.arctan(x).astype(np.float32))


def sinh(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.sinh(x).astype(np.float32))


def cosh(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.cosh(x).astype(np.float32))


def tanh(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.tanh(x).astype(np.float32))


def relu(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.maximum(x, 0).astype(np.float32))


def gelu(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    out = x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return Tensor.from_numpy(out.astype(np.float32))


def silu(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy((x / (1 + np.exp(-x))).astype(np.float32))


def relu6(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.minimum(np.maximum(x, 0), 6).astype(np.float32))


def hardswish(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(
        (x * np.minimum(np.maximum(x + 3, 0), 6) / 6).astype(np.float32)
    )


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(np.where(x >= 0, x, x * negative_slope).astype(np.float32))


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy(
        np.where(x >= 0, x, alpha * (np.exp(x) - 1)).astype(np.float32)
    )


def selu(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return Tensor.from_numpy(
        scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1)).astype(np.float32)
    )


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    if beta != 1.0:
        x = x * beta
    out = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    if beta != 1.0:
        out = out / beta
    return Tensor.from_numpy(out.astype(np.float32))


def softsign(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy((x / (1 + np.abs(x))).astype(np.float32))


def mish(input: Tensor) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    return Tensor.from_numpy((x * np.tanh(np.log1p(np.exp(x)))).astype(np.float32))


def topk(input: Tensor, k: int, dim: int = -1):
    """Return the k largest elements and their indices along ``dim``."""
    x = np.asarray(input.data, dtype=np.float64)
    axis = dim if dim >= 0 else x.ndim + dim
    partitioned = np.argpartition(-x, k - 1, axis=axis)
    index = [slice(None)] * x.ndim
    index[axis] = slice(0, k)
    idx = partitioned[tuple(index)]
    values = np.take_along_axis(x, idx, axis=axis)
    return (
        Tensor.from_numpy(values.astype(np.float32)),
        Tensor.from_numpy(idx.astype(np.int64)),
    )
def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> Tensor:
    """3D convolution (NCDHW layout) via im2col GEMM.

    input: [N, C_in, D, H, W]
    weight: [C_out, C_in // groups, kD, kH, kW]
    """
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)
    N, C_in, D, H, W = x.shape
    C_out, C_in_g, kD, kH, kW = w.shape
    assert C_in == C_in_g * groups
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    str_d, str_h, str_w = stride
    D_out = (D + 2 * pad_d - dil_d * (kD - 1) - 1) // str_d + 1
    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // str_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // str_w + 1
    if pad_d or pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    col = np.zeros((N, C_in * kD * kH * kW, D_out * H_out * W_out), dtype=np.float64)
    o = 0
    for i in range(D_out):
        d_start = i * str_d
        for j in range(H_out):
            h_start = j * str_h
            for k in range(W_out):
                w_start = k * str_w
                col[:, :, o] = x[
                    :,
                    :,
                    d_start : d_start + kD * dil_d : dil_d,
                    h_start : h_start + kH * dil_h : dil_h,
                    w_start : w_start + kW * dil_w : dil_w,
                ].reshape(N, -1)
                o += 1
    w_mat = w.reshape(C_out, -1)
    out = w_mat @ col
    out = out.reshape(C_out, N, D_out, H_out, W_out).transpose(1, 0, 2, 3, 4)
    if bias is not None:
        out += np.asarray(bias.data, dtype=np.float64).reshape(1, -1, 1, 1, 1)
    return Tensor.from_numpy(out.astype(np.float32))


def _conv_transpose2d_impl(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    groups: int,
) -> Tensor:
    x = np.asarray(input.data, dtype=np.float64)
    w = np.asarray(weight.data, dtype=np.float64)
    N, C_in, H, W = x.shape
    C_in_g, C_out, kH, kW = w.shape
    str_h, str_w = stride
    pad_h, pad_w = padding
    op_h, op_w = output_padding
    H_out = (H - 1) * str_h - 2 * pad_h + kH + op_h
    W_out = (W - 1) * str_w - 2 * pad_w + kW + op_w
    out = np.zeros((N, C_out * groups, H_out, W_out), dtype=np.float64)
    g = groups
    ng = C_in // g
    oc_g = C_out  # output channels per group
    for ci in range(C_in):
        gi = ci // ng
        for co in range(C_out):
            co_global = gi * oc_g + co
            wv = w[ci % ng, co]
            for i in range(H):
                for j in range(W):
                    th = i * str_h - pad_h
                    tw = j * str_w - pad_w
                    for kh in range(kH):
                        ah = th + kh
                        if ah < 0 or ah >= H_out:
                            continue
                        for kw in range(kW):
                            aw = tw + kw
                            if aw < 0 or aw >= W_out:
                                continue
                            out[:, co_global, ah, aw] += x[:, ci, i, j] * wv[kh, kw]
    if bias is not None:
        out += np.asarray(bias.data, dtype=np.float64).reshape(1, -1, 1, 1)
    return Tensor.from_numpy(out.astype(np.float32))


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_padding: Tuple[int, int] = (0, 0),
    groups: int = 1,
) -> Tensor:
    return _conv_transpose2d_impl(input, weight, bias, stride, padding, output_padding, groups)
