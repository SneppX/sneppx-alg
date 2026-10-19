"""Neural Network Module — layers, activations, containers with CUDA kernel support."""

from typing import Callable, List, Optional, Tuple, Union
from collections import namedtuple
from .tensor import Tensor, Dtype
from .autograd import Function
import numpy as np
import math

# Try to import CUDA kernels
try:
    from .cuda_kernels import (
        gemm_kernel,
        layernorm_kernel,
        rmsnorm_kernel,
        flash_attention_v2_kernel,
        dropout_kernel,
        gelu_kernel,
        silu_kernel,
        relu_kernel,
        _HAS_CUDA_BACKEND,
    )

    _HAS_CUDA_KERNELS = True
except ImportError:
    _HAS_CUDA_KERNELS = False
    _HAS_CUDA_BACKEND = False


def _is_cuda_tensor(x: Tensor) -> bool:
    """Check if tensor is on CUDA device and CUDA kernels are available."""
    return _HAS_CUDA_KERNELS and x.device.startswith("cuda")


def _state_value_to_array(v):
    """Normalize a state_dict value (ndarray, _TensorData or Tensor) to ndarray."""
    if isinstance(v, Tensor):
        return np.asarray(v.data)
    return np.asarray(v)


IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._persistent_buffers = {}
        self._training = True
        self._name = self.__class__.__name__
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._backward_hooks = {}
        self._hook_count = 0

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        if args:
            self._run_forward_pre_hooks(args[0])
        output = self.forward(*args, **kwargs)
        if args:
            self._run_forward_hooks(args[0], output)
        return output

    def parameters(self):
        params = []
        for p in self._parameters.values():
            if isinstance(p, Tensor):
                params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        named = []
        for name, p in self._parameters.items():
            if isinstance(p, Tensor):
                named.append((f"{prefix}.{name}" if prefix else name, p))
        for name, m in self._modules.items():
            for n, p in m.named_parameters(f"{prefix}.{name}" if prefix else name):
                named.append((n, p))
        return named

    def register_buffer(self, name: str, tensor, persistent: bool = True) -> None:
        object.__setattr__(self, name, tensor)
        self._buffers[name] = tensor
        self._persistent_buffers[name] = persistent

    def get_buffer(self, name: str):
        for n, b in self.named_buffers():
            if n == name:
                return b
        raise KeyError(f"buffer with name {name!r} does not exist")

    def get_parameter(self, name: str):
        for n, p in self.named_parameters():
            if n == name:
                return p
        raise KeyError(f"parameter with name {name!r} does not exist")

    def named_buffers(self, prefix="", remove_duplicate: bool = True, persistent=None):
        named = []
        for name, b in self._buffers.items():
            if not isinstance(b, Tensor):
                continue
            is_persistent = self._persistent_buffers.get(name, True)
            if persistent is not None and is_persistent != persistent:
                continue
            named.append((f"{prefix}.{name}" if prefix else name, b))
        for name, m in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            for n, b in m.named_buffers(child_prefix, remove_duplicate, persistent):
                named.append((n, b))
        return named

    def buffers(self):
        return [b for _, b in self.named_buffers()]

    def register_forward_pre_hook(self, hook) -> int:
        self._hook_count += 1
        self._forward_pre_hooks[self._hook_count] = hook
        return self._hook_count

    def register_forward_hook(self, hook) -> int:
        self._hook_count += 1
        self._forward_hooks[self._hook_count] = hook
        return self._hook_count

    def register_full_backward_hook(self, hook) -> int:
        self._hook_count += 1
        self._backward_hooks[self._hook_count] = hook
        return self._hook_count

    def _run_forward_pre_hooks(self, x):
        for hook in self._forward_pre_hooks.values():
            hook(self, x)

    def _run_forward_hooks(self, x, output):
        for hook in self._forward_hooks.values():
            hook(self, x, output)

    def state_dict(self) -> dict:
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.data.copy()
        for name, b in self.named_buffers(persistent=True):
            sd[name] = b.data.copy()
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        params = {n: p for n, p in self.named_parameters()}
        buffers = {n: b for n, b in self.named_buffers()}
        expected = set(params) | set(buffers)
        missing = list(expected - set(state_dict))
        unexpected = list(set(state_dict) - expected)
        if not strict and unexpected:
            unexpected = []
        for name, arr in state_dict.items():
            target = params.get(name, buffers.get(name))
            if target is None:
                continue
            arr = _state_value_to_array(arr)
            if tuple(arr.shape) != tuple(target.shape):
                err = (
                    f"size mismatch for {name}: copying a param with shape "
                    f"{tuple(arr.shape)} from checkpoint, the shape in current model "
                    f"is {tuple(target.shape)}."
                )
                if strict:
                    raise RuntimeError(err)
                continue
            target.data = arr
        if strict and (missing or unexpected):
            err = f"Missing key(s) in state_dict: {missing}. Unexpected key(s) in state_dict: {unexpected}."
            raise RuntimeError(err)
        return IncompatibleKeys(missing_keys=missing, unexpected_keys=unexpected)

    def to(self, device: str):
        for p in self.parameters():
            if p.device != device:
                p.data = p.data.astype(p.dtype_name)  # Ensure correct dtype
                p._data = p._data  # Keep the same data buffer
                p._data._is_cuda = device.startswith("cuda")
                p.device = device
        for b in self.buffers():
            if b.device != device:
                b._data._is_cuda = device.startswith("cuda")
                b.device = device
        return self

    def train(self):
        self._training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self._training = False
        for m in self._modules.values():
            m.eval()

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            value.requires_grad_(True)
            self._parameters[name] = value
        super().__setattr__(name, value)


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn(
            (out_features, in_features), dtype=dtype
        ) / math.sqrt(in_features)
        self.bias = Tensor.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            # Use CUDA GEMM kernel
            # Reshape for batched GEMM if needed
            if x.ndim > 2:
                # Flatten batch dimensions
                batch_shape = x.shape[:-1]
                batch_size = np.prod(batch_shape)
                x_2d = x.reshape((batch_size, x.shape[-1]))
                out_data = np.empty((batch_size, self.out_features), dtype=np.float32)
                gemm_kernel(
                    x_2d.data,
                    self.weight.data.T,
                    out_data,
                    batch_size,
                    self.out_features,
                    x_2d.shape[1],
                )
                out_data = out_data.reshape(batch_shape + (self.out_features,))
            else:
                out_data = np.empty((x.shape[0], self.out_features), dtype=np.float32)
                gemm_kernel(
                    x.data,
                    self.weight.data.T,
                    out_data,
                    x.shape[0],
                    self.out_features,
                    x.shape[1],
                )
            out = Tensor(out_data, dtype=x.dtype_name, device=x.device)
            if self.bias is not None:
                # Ensure bias is on same device
                bias = (
                    self.bias.to(x.device)
                    if self.bias.device != x.device
                    else self.bias
                )
                out = out + bias
            return out
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype="float32"):
        super().__init__()
        self.weight = Tensor.randn((num_embeddings, embedding_dim), dtype=dtype) * 0.01

    def forward(self, indices: Tensor) -> Tensor:
        from .autograd_ops import EmbeddingFn

        return EmbeddingFn.apply(self.weight, indices)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return x
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            mask = dropout_kernel(x.data, out_data, self.p, True)
            return Tensor.from_numpy(out_data, dtype=x.dtype_name)
        mask = np.random.binomial(1, 1.0 - self.p, x.shape).astype(np.float32)
        mask /= 1.0 - self.p
        return x * Tensor.from_numpy(mask, dtype=x.dtype_name)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor.ones(normalized_shape, dtype=dtype)
            self.bias = Tensor.zeros(normalized_shape, dtype=dtype)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            w = (
                self.weight.data
                if self.weight is not None
                else np.ones(self.normalized_shape)
            )
            b = (
                self.bias.data
                if self.bias is not None
                else np.zeros(self.normalized_shape)
            )
            layernorm_kernel(
                x.data,
                w,
                b,
                out_data,
                self.normalized_shape,
                self.eps,
            )
            return Tensor(out_data, dtype=x.dtype_name, device=x.device)
        if self.elementwise_affine and len(self.normalized_shape) == 1:
            from .autograd_ops import LayerNorm as _LayerNormFn

            return _LayerNormFn.apply(x, self.weight, self.bias, self.eps)
        arr = x.data
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = arr.mean(axis=axis, keepdims=True)
        var = arr.var(axis=axis, keepdims=True)
        out = (arr - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.data + self.bias.data
        return Tensor(out, dtype=x.dtype_name, device=x.device)


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        kH, kW = kernel_size
        scale = math.sqrt(1.0 / (in_channels * kH * kW))
        self.weight = (
            Tensor.randn((out_channels, in_channels // groups, kH, kW), dtype=dtype)
            * scale
        )
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import Conv2d as _Conv2dFn

        return _Conv2dFn.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv1d(Module):
    """1D convolution layer (NCL layout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(stride, int):
            stride = (stride,)
        if isinstance(padding, int):
            padding = (padding,)
        if isinstance(dilation, int):
            dilation = (dilation,)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        kL = kernel_size[0] if kernel_size else 1
        scale = math.sqrt(1.0 / (in_channels * kL))
        self.weight = (
            Tensor.randn((out_channels, in_channels // groups, kL), dtype=dtype) * scale
        )
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if (
            self.groups == 1
            and (
                self.dilation[0]
                if hasattr(self.dilation, "__getitem__")
                else self.dilation
            )
            == 1
        ):
            from .autograd_ops import Conv1d as _Conv1dFn

            stride_ = (
                self.stride[0] if hasattr(self.stride, "__getitem__") else self.stride
            )
            padding_ = (
                self.padding[0]
                if hasattr(self.padding, "__getitem__")
                else self.padding
            )
            y = _Conv1dFn.apply(x, self.weight, stride_, padding_)
            if self.bias is not None:
                return y + self.bias.reshape(1, self.out_channels, 1)
            return y
        from .advanced_ops import conv1d as _conv1d

        return _conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride[0] if self.stride else 1,
            padding=self.padding[0] if self.padding else 0,
            dilation=self.dilation[0] if self.dilation else 1,
            groups=self.groups,
        )


class Conv3d(Module):
    """3D convolution layer (NCDHW layout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        kD, kH, kW = kernel_size
        scale = math.sqrt(1.0 / (in_channels * kD * kH * kW))
        self.weight = (
            Tensor.randn((out_channels, in_channels // groups, kD, kH, kW), dtype=dtype)
            * scale
        )
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import Conv3d as _Conv3dFn

        return _Conv3dFn.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _ConvTransposeNd(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        dtype="float32",
    ):
        super().__init__()
        ndim = self.conv_ndim
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        if isinstance(stride, int):
            stride = (stride,) * ndim
        if isinstance(padding, int):
            padding = (padding,) * ndim
        if isinstance(output_padding, int):
            output_padding = (output_padding,) * ndim
        if isinstance(dilation, int):
            dilation = (dilation,) * ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        k = 1
        for kk in kernel_size:
            k *= kk
        scale = math.sqrt(1.0 / (in_channels * k))
        # transposed weight layout: [C_in//groups, C_out, k...]
        self.weight = (
            Tensor.randn(
                (in_channels // groups, out_channels) + tuple(kernel_size), dtype=dtype
            )
            * scale
        )
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if self.conv_ndim == 2:
            if self.groups == 1 and all(d == 1 for d in self.dilation):
                from .autograd_ops import ConvTranspose2d as _CT2dFn

                return _CT2dFn.apply(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.groups,
                )
            from .advanced_ops import conv_transpose2d as _ct2d

            return _ct2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
            )
        if self.groups == 1 and all(d == 1 for d in self.dilation):
            from .autograd_ops import ConvTransposeNd as _CTNdFn

            return _CTNdFn.apply(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
            )
        from .advanced_ops import _conv_transpose_nd_impl

        return _conv_transpose_nd_impl(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class ConvTranspose1d(_ConvTransposeNd):
    conv_ndim = 1


class ConvTranspose2d(_ConvTransposeNd):
    conv_ndim = 2


class ConvTranspose3d(_ConvTransposeNd):
    conv_ndim = 3


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones((dim,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            rmsnorm_kernel(x.data, self.weight.data, out_data, (self.dim,), self.eps)
            return Tensor(out_data, dtype=x.dtype_name, device=x.device)
        from .autograd_ops import RMSNorm as _RMSNormFn

        return _RMSNormFn.apply(x, self.weight, self.eps)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh_act()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = modules[0]
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x

    def __iter__(self):
        return iter(self._modules.values())


class MultiheadAttention(Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, dtype="float32"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.o_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.dropout = Dropout(dropout)

    def forward(
        self, q: Tensor, k: Tensor = None, v: Tensor = None, mask: Tensor = None
    ) -> Tensor:
        if k is None:
            k = q
        if v is None:
            v = q
        B, T, C = q.shape
        H = self.num_heads
        D = self.head_dim

        # Check if we can use flash attention (CUDA tensors)
        use_flash = _is_cuda_tensor(q) and _is_cuda_tensor(k) and _is_cuda_tensor(v)

        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        if use_flash:
            # Flash Attention: (B, H, T, D)
            q_proj = q_proj.reshape(B, T, H, D).permute(0, 2, 1, 3)
            k_proj = k_proj.reshape(B, -1, H, D).permute(0, 2, 1, 3)
            v_proj = v_proj.reshape(B, -1, H, D).permute(0, 2, 1, 3)

            out = np.empty_like(q_proj.data)
            flash_attention_v2_kernel(
                q_proj.data,
                k_proj.data,
                v_proj.data,
                out,
                B,
                H,
                T,
                D,
                causal=(mask is not None),
            )
            out_t = Tensor(out, dtype=q.dtype_name, device=q.device)
            out_t = out_t.permute(0, 2, 1, 3).reshape(B, T, C)
            return self.o_proj(out_t)
        else:
            q_np = q_proj.data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k_np = k_proj.data.reshape(B, -1, H, D).transpose(0, 2, 1, 3)
            v_np = v_proj.data.reshape(B, -1, H, D).transpose(0, 2, 1, 3)
            scores = (q_np @ k_np.transpose(0, 1, 3, 2)) / math.sqrt(D)
            if mask is not None:
                scores = scores + mask.data
            attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = attn / attn.sum(axis=-1, keepdims=True)
            out_np = attn @ v_np
            out_np = out_np.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.o_proj(Tensor(out_np, dtype=q.dtype_name, device=q.device))


class TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        dtype="float32",
    ):
        super().__init__()
        self.attention = MultiheadAttention(dim, num_heads, dropout, dtype)
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.ffn = Sequential(
            Linear(dim, ffn_dim, dtype=dtype),
            GELU(),
            Linear(ffn_dim, dim, dtype=dtype),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        dtype="float32",
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, dim, dtype)
        self.pos_embedding = Embedding(max_seq_len, dim, dtype)
        self.blocks = Sequential(
            *[
                TransformerBlock(dim, num_heads, ffn_dim, dropout, dtype)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(dim, dtype=dtype)
        self.lm_head = Linear(dim, vocab_size, dtype=dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        B, T = tokens.shape
        tok_emb = self.token_embedding(tokens)
        pos = Tensor.arange(0, T, dtype="int64").unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)


class ModuleList(Module):
    """Holds submodules in a list; recurses parameter/buffer/hook traversal."""

    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module):
        self._list.append(module)
        return module

    def __getitem__(self, i):
        return self._list[i]

    def __setitem__(self, i, module):
        self._list[i] = module

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def parameters(self):
        params = []
        for m in self._list:
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        named = []
        for i, m in enumerate(self._list):
            p = f"{prefix}.{i}" if prefix else str(i)
            if hasattr(m, "named_parameters"):
                named.extend(m.named_parameters(p))
            elif isinstance(m, Tensor):
                named.append((p, m))
        return named

    def named_buffers(self, prefix="", remove_duplicate: bool = True, persistent=None):
        named = []
        for i, m in enumerate(self._list):
            item_prefix = f"{prefix}.{i}" if prefix else str(i)
            if hasattr(m, "named_buffers"):
                for n, b in m.named_buffers(item_prefix, remove_duplicate, persistent):
                    named.append((n, b))
        return named

    def state_dict(self) -> dict:
        sd = {}
        for i, m in enumerate(self._list):
            if hasattr(m, "state_dict"):
                for k, v in m.state_dict().items():
                    sd[f"{i}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict):
        for i, m in enumerate(self._list):
            if hasattr(m, "load_state_dict"):
                sub = {
                    k[len(f"{i}.") :]: v
                    for k, v in state_dict.items()
                    if k.startswith(f"{i}.")
                }
                m.load_state_dict(sub)

    def to(self, device: str):
        for m in self._list:
            if hasattr(m, "to"):
                m.to(device)
        return self

    def train(self):
        self._training = True
        for m in self._list:
            m.train()

    def eval(self):
        self._training = False
        for m in self._list:
            m.eval()

    def forward(self, x: Tensor) -> Tensor:
        for m in self._list:
            x = m(x)
        return x


class ModuleDict(Module):
    """Holds submodules in a dict; recurses parameter/buffer/hook traversal."""

    def __init__(self, modules=None):
        super().__init__()
        self._dict = {}
        if modules is not None:
            self._dict.update(modules)

    def __getitem__(self, k):
        return self._dict[k]

    def __setitem__(self, k, v):
        self._dict[k] = v

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def parameters(self):
        params = []
        for m in self._dict.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        named = []
        for k, m in self._dict.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if hasattr(m, "named_parameters"):
                named.extend(m.named_parameters(p))
            elif isinstance(m, Tensor):
                named.append((p, m))
        return named

    def named_buffers(self, prefix="", remove_duplicate: bool = True, persistent=None):
        named = []
        for k, m in self._dict.items():
            item_prefix = f"{prefix}.{k}" if prefix else str(k)
            if hasattr(m, "named_buffers"):
                for n, b in m.named_buffers(item_prefix, remove_duplicate, persistent):
                    named.append((n, b))
        return named

    def state_dict(self) -> dict:
        sd = {}
        for k, m in self._dict.items():
            if hasattr(m, "state_dict"):
                for kk, v in m.state_dict().items():
                    sd[f"{k}.{kk}"] = v
        return sd

    def load_state_dict(self, state_dict: dict):
        for k, m in self._dict.items():
            if hasattr(m, "load_state_dict"):
                sub = {
                    kk[len(f"{k}.") :]: v
                    for kk, v in state_dict.items()
                    if kk.startswith(f"{k}.")
                }
                m.load_state_dict(sub)

    def to(self, device: str):
        for m in self._dict.values():
            if hasattr(m, "to"):
                m.to(device)
        return self

    def train(self):
        self._training = True
        for m in self._dict.values():
            m.train()

    def eval(self):
        self._training = False
        for m in self._dict.values():
            m.eval()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "ModuleDict has no default forward; index modules explicitly."
        )


# ===========================================================================
#  Recurrent layers: RNN / GRU / LSTM (pure-NumPy, autograd-aware)
# ===========================================================================


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Tensor.randn((hidden_size, input_size)) * 0.1
        self.weight_hh = Tensor.randn((hidden_size, hidden_size)) * 0.1
        if bias:
            self.bias_ih = Tensor.zeros((hidden_size,))
            self.bias_hh = Tensor.zeros((hidden_size,))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        lin = x @ self.weight_ih.T
        if self.bias_ih is not None:
            lin = lin + self.bias_ih
        lin = lin + (h @ self.weight_hh.T)
        if self.bias_hh is not None:
            lin = lin + self.bias_hh
        if self.nonlinearity == "relu":
            return lin.relu()
        return lin.tanh()


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Tensor.randn((4 * hidden_size, input_size)) * 0.1
        self.weight_hh = Tensor.randn((4 * hidden_size, hidden_size)) * 0.1
        if bias:
            self.bias_ih = Tensor.zeros((4 * hidden_size,))
            self.bias_hh = Tensor.zeros((4 * hidden_size,))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, hc) -> tuple:
        h, c = hc
        lin = x @ self.weight_ih.T
        if self.bias_ih is not None:
            lin = lin + self.bias_ih
        lin = lin + (h @ self.weight_hh.T)
        if self.bias_hh is not None:
            lin = lin + self.bias_hh
        i, f, g, o = (
            lin[:, : self.hidden_size],
            lin[:, self.hidden_size : 2 * self.hidden_size],
            lin[:, 2 * self.hidden_size : 3 * self.hidden_size],
            lin[:, 3 * self.hidden_size :],
        )
        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Tensor.randn((3 * hidden_size, input_size)) * 0.1
        self.weight_hh = Tensor.randn((3 * hidden_size, hidden_size)) * 0.1
        if bias:
            self.bias_ih = Tensor.zeros((3 * hidden_size,))
            self.bias_hh = Tensor.zeros((3 * hidden_size,))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        lin_ih = x @ self.weight_ih.T
        if self.bias_ih is not None:
            lin_ih = lin_ih + self.bias_ih
        lin_hh = h @ self.weight_hh.T
        if self.bias_hh is not None:
            lin_hh = lin_hh + self.bias_hh
        r = (lin_ih[:, : self.hidden_size] + lin_hh[:, : self.hidden_size]).sigmoid()
        z = (
            lin_ih[:, self.hidden_size : 2 * self.hidden_size]
            + lin_hh[:, self.hidden_size : 2 * self.hidden_size]
        ).sigmoid()
        n = (
            lin_ih[:, 2 * self.hidden_size :] + r * lin_hh[:, 2 * self.hidden_size :]
        ).tanh()
        return (1 - z) * n + z * h


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [
                RNNCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    bias,
                    nonlinearity,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, h0=None):
        if self.batch_first:
            x = x.transpose(1, 0) if x.ndim == 3 else x
        T = x.shape[0]
        batch = x.shape[1]
        if h0 is None:
            h = [
                Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)
            ]
        else:
            h = [h0[i] for i in range(self.num_layers)]
        outputs = []
        for t in range(T):
            xt = x[t]
            for l in range(self.num_layers):
                h[l] = self.layers[l](xt, h[l])
                xt = h[l]
            outputs.append(h[-1])
        out = Tensor.stack(outputs)  # (T, batch, hidden)
        if self.batch_first:
            out = out.transpose(1, 0)
        h_n = Tensor.stack(h)  # (num_layers, batch, hidden)
        return out, h_n


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, hc0=None):
        if self.batch_first:
            x = x.transpose(1, 0)
        T = x.shape[0]
        batch = x.shape[1]
        if hc0 is None:
            h = [
                Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)
            ]
            c = [
                Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)
            ]
        else:
            h = [hc0[0][i] for i in range(self.num_layers)]
            c = [hc0[1][i] for i in range(self.num_layers)]
        outputs = []
        for t in range(T):
            xt = x[t]
            for l in range(self.num_layers):
                h[l], c[l] = self.layers[l](xt, (h[l], c[l]))
                xt = h[l]
            outputs.append(h[-1])
        out = Tensor.stack(outputs)
        if self.batch_first:
            out = out.transpose(1, 0)
        h_n = Tensor.stack(h)
        c_n = Tensor.stack(c)
        return out, (h_n, c_n)


class GRU(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [
                GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, h0=None):
        if self.batch_first:
            x = x.transpose(1, 0)
        T = x.shape[0]
        batch = x.shape[1]
        if h0 is None:
            h = [
                Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)
            ]
        else:
            h = [h0[i] for i in range(self.num_layers)]
        outputs = []
        for t in range(T):
            xt = x[t]
            for l in range(self.num_layers):
                h[l] = self.layers[l](xt, h[l])
                xt = h[l]
            outputs.append(h[-1])
        out = Tensor.stack(outputs)
        if self.batch_first:
            out = out.transpose(1, 0)
        h_n = Tensor.stack(h)
        return out, h_n


# ===========================================================================
#  Normalization layers: BatchNorm (1d/2d) and GroupNorm
# ===========================================================================


class BatchNormFn(Function):
    @staticmethod
    def forward(
        ctx, x, weight, bias, running_mean, running_var, training, momentum, eps
    ):
        xd = x.data
        C = xd.shape[1]
        keep = (1, C) + (1,) * (xd.ndim - 2)
        axes = (0,) + tuple(range(2, xd.ndim))
        if training:
            mean = xd.mean(axis=axes, keepdims=True)
            var = ((xd - mean) ** 2).mean(axis=axes, keepdims=True)
            rm = running_mean.data * (1 - momentum) + momentum * mean.reshape(C)
            rv = running_var.data * (1 - momentum) + momentum * var.reshape(C)
            running_mean.data = rm.copy()
            running_var.data = rv.copy()
            use_mean = mean
            use_var = var
        else:
            use_mean = running_mean.data.reshape(keep)
            use_var = running_var.data.reshape(keep)
        inv = 1.0 / np.sqrt(use_var + eps)
        xhat = (xd - use_mean) * inv
        y = xhat * weight.data.reshape(keep) + bias.data.reshape(keep)
        N = int(np.prod([xd.shape[a] for a in axes]))
        ctx.save_attr(
            inv=inv,
            xhat=xhat,
            w=weight.data.reshape(keep),
            axes=axes,
            N=N,
            training=training,
        )
        return Tensor(y, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.data
        inv = ctx.get_attr("inv")
        xhat = ctx.get_attr("xhat")
        w = ctx.get_attr("w")
        axes = ctx.get_attr("axes")
        N = ctx.get_attr("N")
        training = ctx.get_attr("training")
        grad_w = (g * xhat).sum(axis=axes)
        grad_b = g.sum(axis=axes)
        if training:
            grad_xhat = g * w
            dx = (
                N * grad_xhat
                - grad_xhat.sum(axis=axes, keepdims=True)
                - xhat * (grad_xhat * xhat).sum(axis=axes, keepdims=True)
            )
            dx = dx / N * inv
        else:
            dx = g * w * inv
        return [
            Tensor(dx, dtype=grad_output.dtype),
            Tensor(grad_w, dtype=grad_output.dtype),
            Tensor(grad_b, dtype=grad_output.dtype),
            None,
            None,
        ]


class GroupNormFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, num_groups, eps):
        xd = x.data
        N, C = xd.shape[0], xd.shape[1]
        G = num_groups
        Cg = C // G
        spatial = xd.shape[2:]
        xg = xd.reshape(N, G, Cg, *spatial)
        reduce_axes = tuple(range(2, xg.ndim))  # (Cg, *spatial) within each (n, g)
        mean = xg.mean(axis=reduce_axes, keepdims=True)
        var = xg.var(axis=reduce_axes, keepdims=True)
        xhat = (xg - mean) / np.sqrt(var + eps)
        xhat = xhat.reshape(xd.shape)
        keep = (1, C) + (1,) * (xd.ndim - 2)
        y = xhat * weight.data.reshape(keep) + bias.data.reshape(keep)
        Nred = int(np.prod(xg.shape[2:]))
        inv_ch = np.repeat(np.sqrt(var + eps).reshape(N, G, 1, 1), Cg, axis=1).reshape(
            N, C, 1, 1
        )  # (N, C, 1, 1)
        ctx.save_attr(
            inv=inv_ch, xhat=xhat, w=weight.data.reshape(keep), N=Nred, num_groups=G
        )
        return Tensor(y, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.data
        inv = ctx.get_attr("inv")
        xhat = ctx.get_attr("xhat")
        w = ctx.get_attr("w")
        N = ctx.get_attr("N")
        num_groups = ctx.get_attr("num_groups")
        C = g.shape[1]
        Cg = C // num_groups
        spatial = g.shape[2:]
        gg = g.reshape(-1, num_groups, Cg, *spatial)
        xh = xhat.reshape(-1, num_groups, Cg, *spatial)
        inv_g = inv.reshape(g.shape[0], num_groups, Cg, 1, 1)[:, :, 0].reshape(
            g.shape[0], num_groups, 1, 1, 1
        )
        rax = tuple(range(2, gg.ndim))  # (Cg, *spatial) within each (n, g)
        sum_g = gg.sum(axis=rax, keepdims=True)
        sum_gx = (gg * xh).sum(axis=rax, keepdims=True)
        dxg = (N * gg - sum_g - xh * sum_gx) / N / inv_g
        dx = dxg.reshape(g.shape)
        sax = tuple(range(3, gg.ndim))  # spatial dims
        grad_w = (gg * xh).sum(axis=(0,) + sax).reshape(C)
        grad_b = gg.sum(axis=(0,) + sax).reshape(C)
        return [
            Tensor(dx, dtype=grad_output.dtype),
            Tensor(grad_w, dtype=grad_output.dtype),
            Tensor(grad_b, dtype=grad_output.dtype),
            None,
        ]


class _BatchNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor.ones((num_features,))
            self.weight.requires_grad_(True)
            self.bias = Tensor.zeros((num_features,))
            self.bias.requires_grad_(True)
        if track_running_stats:
            self.register_buffer("running_mean", Tensor.zeros((num_features,)))
            self.register_buffer("running_var", Tensor.ones((num_features,)))

    def forward(self, x: Tensor) -> Tensor:
        training = self._training and self.track_running_stats
        if self.affine:
            w, b = self.weight, self.bias
        else:
            w = Tensor.ones((self.num_features,))
            b = Tensor.zeros((self.num_features,))
        if self.track_running_stats:
            rm, rv = self.running_mean, self.running_var
        else:
            rm = Tensor.zeros((self.num_features,))
            rv = Tensor.ones((self.num_features,))
        return BatchNormFn.apply(x, w, b, rm, rv, training, self.momentum, self.eps)


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass


class BatchNorm3d(_BatchNorm):
    pass


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Tensor.ones((num_channels,))
            self.weight.requires_grad_(True)
            self.bias = Tensor.zeros((num_channels,))
            self.bias.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        if self.affine:
            w, b = self.weight, self.bias
        else:
            w = Tensor.ones((self.num_channels,))
            b = Tensor.zeros((self.num_channels,))
        return GroupNormFn.apply(x, w, b, self.num_groups, self.eps)


class InstanceNormFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps):
        xd = x.data
        axes = tuple(range(2, xd.ndim))  # spatial dims, per (n, c)
        mean = xd.mean(axis=axes, keepdims=True)
        var = xd.var(axis=axes, keepdims=True)
        xhat = (xd - mean) / np.sqrt(var + eps)
        keep = (1, xd.shape[1]) + (1,) * (xd.ndim - 2)
        y = xhat * weight.data.reshape(keep) + bias.data.reshape(keep)
        ctx.save_attr(
            inv=1.0 / np.sqrt(var + eps),
            xhat=xhat,
            w=weight.data.reshape(keep),
            axes=axes,
        )
        return Tensor(y, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.data
        inv = ctx.get_attr("inv")
        xhat = ctx.get_attr("xhat")
        w = ctx.get_attr("w")
        axes = ctx.get_attr("axes")
        N = int(np.prod([g.shape[a] for a in axes]))
        grad_w = (g * xhat).sum(axis=axes)
        grad_b = g.sum(axis=axes)
        grad_xhat = g * w
        dx = (
            N * grad_xhat
            - grad_xhat.sum(axis=axes, keepdims=True)
            - xhat * (grad_xhat * xhat).sum(axis=axes, keepdims=True)
        )
        dx = dx / N * inv
        return [
            Tensor(dx, dtype=grad_output.dtype),
            Tensor(grad_w, dtype=grad_output.dtype),
            Tensor(grad_b, dtype=grad_output.dtype),
            None,
            None,
        ]


class _InstanceNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor.ones((num_features,))
            self.weight.requires_grad_(True)
            self.bias = Tensor.zeros((num_features,))
            self.bias.requires_grad_(True)
        if track_running_stats:
            self.register_buffer("running_mean", Tensor.zeros((num_features,)))
            self.register_buffer("running_var", Tensor.ones((num_features,)))

    def forward(self, x: Tensor) -> Tensor:
        if self.affine:
            w, b = self.weight, self.bias
        else:
            w = Tensor.ones((self.num_features,))
            b = Tensor.zeros((self.num_features,))
        if self.track_running_stats:
            rm, rv = self.running_mean, self.running_var
        else:
            rm = Tensor.zeros((self.num_features,))
            rv = Tensor.ones((self.num_features,))
        return InstanceNormFn.apply(x, w, b, rm, rv, self.eps)


class InstanceNorm1d(_InstanceNorm):
    pass


class InstanceNorm2d(_InstanceNorm):
    pass


class InstanceNorm3d(_InstanceNorm):
    pass


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        start_dim = (
            self.start_dim if self.start_dim >= 0 else len(shape) + self.start_dim
        )
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        flat = 1
        for s in shape[start_dim : end_dim + 1]:
            flat *= s
        new_shape = shape[:start_dim] + [flat] + shape[end_dim + 1 :]
        return x.reshape(new_shape)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Bilinear(Module):
    def __init__(
        self, in1_features, in2_features, out_features, bias=True, dtype="float32"
    ):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = (
            Tensor.randn((out_features, in1_features, in2_features), dtype=dtype) * 0.1
        )
        self.bias = Tensor.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        from .autograd_ops import BilinearFn

        return BilinearFn.apply(input1, input2, self.weight, self.bias)


# ===========================================================================
# Loss functions (torch.nn-compatible Module wrappers)
# ===========================================================================


class _Loss(Module):
    """Base class for loss modules (mirrors torch.nn.modules.loss._Loss)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction


def _reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Apply a reduction to a differentiable loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class MSELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.mse_loss(target)


class L1Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.mae_loss(target)


class MAELoss(L1Loss):
    pass


class CrossEntropyLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.cross_entropy(target)


class NLLLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.nll_loss(target)


class KLDivLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.kl_div(target)


class BCELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.bce_loss(target)


class BCEWithLogitsLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.bce_with_logits_loss(target)


class SmoothL1Loss(_Loss):
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.smooth_l1_loss(target, self.beta)


class HuberLoss(_Loss):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.huber_loss(target, self.delta)


class FocalLoss(_Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return input.focal_loss(target, self.gamma, self.alpha)


class TripletMarginLoss(_Loss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return anchor.triplet_margin_loss(positive, negative, self.margin)


class ContrastiveLoss(_Loss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, other: Tensor, y: Tensor) -> Tensor:
        return input.contrastive_loss(other, y, self.margin)


class MarginRankingLoss(_Loss):
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, other: Tensor, y: Tensor) -> Tensor:
        return input.margin_ranking_loss(other, y, self.margin)


class CTCLoss(_Loss):
    def __init__(self, blank: int = 0, reduction: str = "mean"):
        super().__init__(reduction)
        self.blank = blank

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Optional[List[int]] = None,
        target_lengths: Optional[List[int]] = None,
    ) -> Tensor:
        return log_probs.ctc_loss(
            targets, self.blank, input_lengths, target_lengths, self.reduction
        )


class GaussianNLLLoss(_Loss):
    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        eps = self.eps
        if self.full:
            loss = 0.5 * (
                (var * (2 * np.pi) + eps).log() + (input - target) ** 2 / (var + eps)
            )
        else:
            loss = 0.5 * (
                (var + eps).log() + (input - target) ** 2 / (var + eps)
            ) + 0.5 * math.log(2 * np.pi)
        return _reduce_loss(loss, self.reduction)


class HingeEmbeddingLoss(_Loss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # y in {-1, +1}: mask selects x for y>=0, max(0, margin - x) otherwise.
        mask = Tensor((target.data >= 0).astype(np.float32))
        loss = mask * input + (1.0 - mask) * (self.margin - input).clamp_min(0.0)
        return _reduce_loss(loss, self.reduction)


class PoissonNLLLoss(_Loss):
    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = input.exp() - target * input
        if self.full:
            y = target.data.astype(np.float64)
            const = (y * np.log(y + self.eps) - y + y * np.log(2 * np.pi) / 2).astype(
                "float32"
            )
            loss = loss + Tensor(const)
        return _reduce_loss(loss, self.reduction)


class SoftMarginLoss(_Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        z = (input * target * (-1.0)).exp()
        loss = (z + 1.0).log()
        return _reduce_loss(loss, self.reduction)


class MultiMarginLoss(_Loss):
    def __init__(
        self, p: int = 1, margin: float = 1.0, weight=None, reduction: str = "mean"
    ):
        super().__init__(reduction)
        self.p = p
        self.margin = margin
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n, c = input.shape
        y = target.data.astype(np.int64)
        onehot = np.zeros((n, c), dtype=np.float32)
        onehot[np.arange(n), y] = 1.0
        oh_t = Tensor(onehot)
        nt_t = Tensor(1.0 - onehot)
        target_scores = (oh_t * input).sum(dim=1).unsqueeze(1)  # (n, 1)
        diff = (self.margin - (target_scores - input)).clamp_min(0.0)  # (n, c)
        losses = diff ** float(self.p)
        if self.weight is not None:
            w = self.weight.data if hasattr(self.weight, "data") else self.weight
            losses = losses * Tensor(np.asarray(w, dtype=np.float32)[None, :])
        loss = (losses * nt_t).sum(dim=1)
        return _reduce_loss(loss, self.reduction)


class MultiLabelMarginLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n, c = input.shape
        tmask = (target.data >= 0).astype(np.float32)  # (n, c)
        tm = Tensor(tmask)
        x_col = input.unsqueeze(2)  # (n, c, 1) -> x[tn]
        x_row = input.unsqueeze(1)  # (n, 1, c) -> x[k]
        term = (1.0 - x_col + x_row).clamp_min(0.0)
        term = term * tm.unsqueeze(2) * (1.0 - tm).unsqueeze(1)
        loss_sum = term.sum(dim=2).sum(dim=1).sum()
        if self.reduction == "sum":
            return loss_sum
        return loss_sum / float(n * c)


class MultiLabelSoftMarginLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # per-element BCE (multi-label, many-hot targets)
        pos = input.clamp(1e-7, 1.0).log()
        neg = (1.0 - input).clamp(1e-7, 1.0).log()
        loss = -(target * pos + (1.0 - target) * neg)
        return _reduce_loss(loss, self.reduction)


class Pooling1d(Module):
    """1D pooling layer."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        pool_type: str = "max",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool_type = pool_type

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool, MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 1
        )
        fn = MaxPool if self.pool_type == "max" else AvgPool
        return fn.apply(x, k, s, p, d)


class Pooling2d(Module):
    """2D pooling layer."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        pool_type="max",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool_type = pool_type

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool, MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 2
        )
        fn = MaxPool if self.pool_type == "max" else AvgPool
        return fn.apply(x, k, s, p, d)


class Pooling3d(Module):
    """3D pooling layer."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        pool_type="max",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool_type = pool_type

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool, MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 3
        )
        fn = MaxPool if self.pool_type == "max" else AvgPool
        return fn.apply(x, k, s, p, d)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AdaptiveAvgPoolFn

        return AdaptiveAvgPoolFn.apply(x, tuple(self.output_size), True)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AdaptiveAvgPoolFn

        return AdaptiveAvgPoolFn.apply(x, tuple(self.output_size))


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AdaptiveAvgPoolFn

        return AdaptiveAvgPoolFn.apply(x, tuple(self.output_size), True)


class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, l = x.shape
        out_l = self.output_size[0]
        from .autograd_ops import AdaptiveMaxPoolFn

        return AdaptiveMaxPoolFn.apply(x, tuple(self.output_size), True)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AdaptiveMaxPoolFn

        return AdaptiveMaxPoolFn.apply(x, tuple(self.output_size), False)


class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AdaptiveMaxPoolFn

        return AdaptiveMaxPoolFn.apply(x, tuple(self.output_size), True)


def _norm_pool_param(kernel_size, stride, padding, dilation, ndim):
    """Normalize pooling hyper-parameters to per-dim tuples (additive helper)."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndim
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride,) * ndim
    if isinstance(padding, int):
        padding = (padding,) * ndim
    if isinstance(dilation, int):
        dilation = (dilation,) * ndim
    return tuple(kernel_size), tuple(stride), tuple(padding), tuple(dilation)


class MaxPool1d(Module):
    """1D max pooling (differentiable, padding + dilation)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(stride, int):
            stride = (stride,)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 1
        )
        return MaxPool.apply(x, k, s, p, d)


class MaxPool2d(Module):
    """2D max pooling (differentiable, padding + dilation)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 2
        )
        return MaxPool.apply(x, k, s, p, d)


class MaxPool3d(Module):
    """3D max pooling (differentiable, padding + dilation)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import MaxPool

        k, s, p, d = _norm_pool_param(
            self.kernel_size, self.stride, self.padding, self.dilation, 3
        )
        return MaxPool.apply(x, k, s, p, d)


class AvgPool1d(Module):
    """1D average pooling (differentiable, count_include_pad=True)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()
        self.kernel_size = (
            (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool

        k, s, p, d = _norm_pool_param(self.kernel_size, self.stride, self.padding, 1, 1)
        return AvgPool.apply(x, k, s, p, d)


class AvgPool2d(Module):
    """2D average pooling (differentiable, count_include_pad=True)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool

        k, s, p, d = _norm_pool_param(self.kernel_size, self.stride, self.padding, 1, 2)
        return AvgPool.apply(x, k, s, p, d)


class AvgPool3d(Module):
    """3D average pooling (differentiable, count_include_pad=True)."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import AvgPool

        k, s, p, d = _norm_pool_param(self.kernel_size, self.stride, self.padding, 1, 3)
        return AvgPool.apply(x, k, s, p, d)


class _ConstantPadNd(Module):
    def __init__(self, padding, value: float = 0.0):
        super().__init__()
        if isinstance(padding, int):
            padding = tuple([padding] * (2 * self.pad_ndim))
        self.padding = tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padding
        ndim = x.ndim
        npads = len(pad) // 2
        pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(npads)]
        pad_width = [(0, 0)] * (ndim - npads) + list(reversed(pairs))
        from .autograd_ops import PadNdFn

        return PadNdFn.apply(x, pad_width, "constant", float(self.value))


class ConstantPad1d(_ConstantPadNd):
    pad_ndim = 1


class ConstantPad2d(_ConstantPadNd):
    pad_ndim = 2


class ConstantPad3d(_ConstantPadNd):
    pad_ndim = 3


class ZeroPad1d(ConstantPad1d):
    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ZeroPad2d(ConstantPad2d):
    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ZeroPad3d(ConstantPad3d):
    def __init__(self, padding):
        super().__init__(padding, 0.0)


class _ReflectionPadNd(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = tuple([padding] * (2 * self.pad_ndim))
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padding
        ndim = x.ndim
        npads = len(pad) // 2
        pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(npads)]
        pad_width = [(0, 0)] * (ndim - npads) + list(reversed(pairs))
        from .autograd_ops import PadNdFn

        return PadNdFn.apply(x, pad_width, "reflect", 0.0)


class ReflectionPad1d(_ReflectionPadNd):
    pad_ndim = 1


class ReflectionPad2d(_ReflectionPadNd):
    pad_ndim = 2


class ReflectionPad3d(_ReflectionPadNd):
    pad_ndim = 3


class _ReplicationPadNd(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = tuple([padding] * (2 * self.pad_ndim))
        self.padding = tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.padding
        ndim = x.ndim
        npads = len(pad) // 2
        pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(npads)]
        pad_width = [(0, 0)] * (ndim - npads) + list(reversed(pairs))
        from .autograd_ops import PadNdFn

        return PadNdFn.apply(x, pad_width, "edge", 0.0)


class ReplicationPad1d(_ReplicationPadNd):
    pad_ndim = 1


class ReplicationPad2d(_ReplicationPadNd):
    pad_ndim = 2


class ReplicationPad3d(_ReplicationPadNd):
    pad_ndim = 3


class ParameterList(Module):
    def __init__(self, parameters=None):
        super().__init__()
        self._list = []
        if parameters is not None:
            for p in parameters:
                self.append(p)

    def append(self, parameter):
        self._list.append(parameter)
        return parameter

    def extend(self, parameters):
        for p in parameters:
            self.append(p)
        return self

    def __getitem__(self, i):
        return self._list[i]

    def __setitem__(self, i, parameter):
        self._list[i] = parameter

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def parameters(self):
        return [p for p in self._list if isinstance(p, Tensor)]

    def named_parameters(self, prefix=""):
        named = []
        for i, p in enumerate(self._list):
            if isinstance(p, Tensor):
                named.append((f"{prefix}.{i}" if prefix else str(i), p))
            elif hasattr(p, "named_parameters"):
                named.extend(p.named_parameters(f"{prefix}.{i}" if prefix else str(i)))
        return named

    def state_dict(self) -> dict:
        return {name: p.data.copy() for name, p in self.named_parameters()}


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        self._dict = {}
        if parameters is not None:
            self._dict.update(parameters)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def parameters(self):
        return [p for p in self._dict.values() if isinstance(p, Tensor)]

    def named_parameters(self, prefix=""):
        named = []
        for k, p in self._dict.items():
            if isinstance(p, Tensor):
                named.append((f"{prefix}.{k}" if prefix else str(k), p))
            elif hasattr(p, "named_parameters"):
                named.extend(p.named_parameters(f"{prefix}.{k}" if prefix else str(k)))
        return named


# ===========================================================================
#  Activation functions (nn.Module wrappers)
# ===========================================================================


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.data = np.maximum(x.data * self.negative_slope, x.data)
            return x
        return x.leaky_relu(self.negative_slope)


class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = Tensor.full((num_parameters,), init)

    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import PReluFn

        return PReluFn.apply(x, self.weight)


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return x.elu(self.alpha)


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(dim=self.dim)


class SELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.selu()


class Softplus(Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        return x.softplus(beta=self.beta, threshold=self.threshold)


class Softsign(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.softsign()


class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.hardswish()


class Hardtanh(Module):
    def __init__(
        self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.hardtanh(self.min_val, self.max_val)


class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = False):
        super().__init__(0.0, 6.0, inplace)


class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.mish()


class Hardsigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.hardsigmoid()


class Hardshrink(Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return x.hardshrink(self.lambd)


class Softshrink(Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return x.softshrink(self.lambd)


class Tanhshrink(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanhshrink()


class Threshold(Module):
    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.threshold(self.threshold, self.value)


class GLU(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.glu(self.dim)


class CELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.elu(self.alpha)


class RReLU(Module):
    def __init__(
        self,
        lower: float = 0.125,
        upper: float = 0.3333333333333333,
        inplace: bool = False,
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        if self._training:
            slope = np.random.uniform(self.lower, self.upper, xd.shape).astype(xd.dtype)
        else:
            slope = np.full(xd.shape, (self.lower + self.upper) / 2.0, dtype=xd.dtype)
        # positive side slope 1, negative side sampled slope -> y = x * c (differentiable)
        c = np.where(xd >= 0, np.float32(1.0), slope).astype(xd.dtype)
        return x * Tensor(c, dtype=x.dtype_name, device=x.device)


class LogSigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        from .autograd_ops import LogSigmoidFn

        return LogSigmoidFn.apply(x)


class CosineEmbeddingLoss(_Loss):
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, other: Tensor, y: Tensor) -> Tensor:
        return input.cosine_embedding_loss(other, y, self.margin)


class EmbeddingBag(Module):
    """Sum or mean of the embeddings of the entries indexed by `input`.

    Mirrors ``torch.nn.EmbeddingBag`` (mode="sum"/"mean"/"max").
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        include_last_offset: bool = False,
        dtype="float32",
    ):
        super().__init__()
        self.weight = Tensor.randn((num_embeddings, embedding_dim), dtype=dtype) * 0.01
        self.mode = mode
        self.include_last_offset = include_last_offset

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        from .autograd_ops import EmbeddingBagFn

        idx = input.data.astype(np.int64).reshape(-1)
        off = offsets.data.astype(np.int64) if offsets is not None else None
        psw = (
            per_sample_weights.data.astype(np.float64).reshape(-1)
            if per_sample_weights is not None
            else None
        )
        return EmbeddingBagFn.apply(self.weight, idx, off, self.mode, psw)


class CosineSimilarity(Module):
    """Computes cosine similarity between x1 and x2 along `dim`."""

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        dim = self.dim % len(x1.shape)
        num = (x1 * x2).sum(dim)
        den = ((x1 * x1).sum(dim) + self.eps).sqrt() * (
            (x2 * x2).sum(dim) + self.eps
        ).sqrt()
        return num / den.clamp_min(self.eps)


class PairwiseDistance(Module):
    """Computes the pairwise distance between x1 and x2 using `p`-norm."""

    def __init__(self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        d = x1 - x2
        if self.p == 2.0:
            out = ((d * d).sum(-1) + self.eps).sqrt()
        elif self.p == 1.0:
            out = d.abs().sum(-1)
        else:
            out = (d.abs().sum(-1) + self.eps) ** (1.0 / self.p)
        if self.keepdim:
            out = out.unsqueeze(-1)
        return out


class ChannelShuffle(Module):
    """Divides channels into `groups` groups and rearranges them."""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        n, c, *spatial = x.shape
        cgh = c // self.groups
        out = x.reshape(n, self.groups, cgh, *spatial)
        out = out.permute(0, 2, 1, *range(3, 3 + len(spatial)))
        return out.reshape(n, c, *spatial)


class PixelShuffle(Module):
    """Rearranges (C*r^2, H, W) to (C, H*r, W*r)."""

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.upscale_factor
        n, c, h, w = x.shape
        c_out = c // (r * r)
        return (
            x.reshape(n, c_out, r, r, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, c_out, h * r, w * r)
        )


class PixelUnshuffle(Module):
    """Inverse of PixelShuffle."""

    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.downscale_factor
        n, c, h, w = x.shape
        c_out = c * r * r
        return (
            x.reshape(n, c, h // r, r, w // r, r)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(n, c_out, h // r, w // r)
        )


class Upsample(Module):
    """Upsamples a given multi-channel 1D/2D/3D input (nearest or bilinear)."""

    def __init__(
        self, size=None, scale_factor=None, mode: str = "nearest", align_corners=None
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        xd = np.asarray(x.data, dtype=np.float64)
        ndim = xd.ndim - 2
        spatial = xd.shape[-ndim:]
        if self.scale_factor is not None:
            target = tuple(int(round(s * self.scale_factor)) for s in spatial)
        elif self.size is not None:
            target = tuple(
                int(s)
                for s in (
                    self.size
                    if isinstance(self.size, (tuple, list))
                    else (self.size,) * ndim
                )
            )
        else:
            target = spatial
        scales = tuple(t / s for s, t in zip(spatial, target))
        if self.mode in ("bilinear", "trilinear"):
            from .autograd_ops import UpsampleLinearNdFn

            return UpsampleLinearNdFn.apply(x, target, bool(self.align_corners))
        from .autograd_ops import UpsampleNearestFn

        return UpsampleNearestFn.apply(x, target)


def _zoom_nn(xd, target):
    """Nearest-neighbor upsampling for a batch x channel x spatial tensor."""
    out = np.asarray(xd, dtype=np.float64)
    ndim = out.ndim - 2
    spatial = out.shape[-ndim:]
    for i, (s, t) in enumerate(zip(spatial, target)):
        axis = out.ndim - ndim + i
        idx = np.repeat(np.arange(s, dtype=np.int64), int(np.ceil(t / s)))[:t]
        out = np.take(out, idx, axis=axis)
    return out


class UpsamplingNearest2d(Upsample):
    def __init__(self, size=None, scale_factor=None):
        super().__init__(size=size, scale_factor=scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    def __init__(self, size=None, scale_factor=None):
        super().__init__(size=size, scale_factor=scale_factor, mode="bilinear")


class Unfold(Module):
    """Extracts sliding local blocks from a batched tensor.

    Mirrors ``torch.nn.Unfold`` (output ``(N, C*prod(kernel_size), L)``).
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        xd = np.asarray(x.data, dtype=np.float64)
        k = self.kernel_size
        if isinstance(k, int):
            k = (k, k)
        dil = self.dilation
        if isinstance(dil, int):
            dil = (dil, dil)
        pad = self.padding
        if isinstance(pad, int):
            pad = (pad, pad)
        st = self.stride
        if isinstance(st, int):
            st = (st, st)
        N, C, H, W = xd.shape
        kH, kW = k
        dilH, dilW = dil
        padH, padW = pad
        sH, sW = st
        H_out = (H + 2 * padH - dilH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * padW - dilW * (kW - 1) - 1) // sW + 1
        from .autograd_ops import UnfoldFn

        return UnfoldFn.apply(x, (kH, kW), (dilH, dilW), (padH, padW), (sH, sW))


class Fold(Module):
    """Combines an array of sliding local blocks into a larger tensor.

    Mirrors ``torch.nn.Fold`` (input ``(N, C*prod(kernel_size), L)``).
    """

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        xd = np.asarray(x.data, dtype=np.float64)
        k = self.kernel_size
        if isinstance(k, int):
            k = (k, k)
        dil = self.dilation
        if isinstance(dil, int):
            dil = (dil, dil)
        pad = self.padding
        if isinstance(pad, int):
            pad = (pad, pad)
        st = self.stride
        if isinstance(st, int):
            st = (st, st)
        od = self.output_size
        if isinstance(od, int):
            od = (od, od)
        from .autograd_ops import FoldFn

        H, W = od
        kH, kW = k
        dilH, dilW = dil
        padH, padW = pad
        sH, sW = st
        return FoldFn.apply(x, H, W, (kH, kW), (dilH, dilW), (padH, padW), (sH, sW))


class _MaxUnpoolNd(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.pool_ndim
        self.kernel_size = kernel_size
        stride = stride or kernel_size
        if isinstance(stride, int):
            stride = (stride,) * self.pool_ndim
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding,) * self.pool_ndim
        self.padding = padding

    def forward(self, input, indices, output_size=None):
        xd = np.asarray(input.data)
        idx = indices.data
        ndim = self.pool_ndim
        N, C = xd.shape[:2]
        in_sp = xd.shape[2:]
        if output_size:
            out_sp = tuple(int(s) for s in np.atleast_1d(output_size))
        else:
            out_sp = tuple(
                (s - 1) * st + k
                for s, st, k in zip(in_sp, self.stride, self.kernel_size)
            )
        from .autograd_ops import MaxUnpoolFn

        return MaxUnpoolFn.apply(input, indices, tuple(out_sp))


class MaxUnpool1d(_MaxUnpoolNd):
    pool_ndim = 1


class MaxUnpool2d(_MaxUnpoolNd):
    pool_ndim = 2


class MaxUnpool3d(_MaxUnpoolNd):
    pool_ndim = 3


# Parameter initialization namespace (torch.nn.init-compatible).
from .nn_init import (  # noqa: E402,F401
    zeros_ as _zeros_,
    ones_ as _ones_,
    constant_ as _constant_,
    uniform_ as _uniform_,
    normal_ as _normal_,
    xavier_uniform_ as _xavier_uniform_,
    xavier_normal_ as _xavier_normal_,
    kaiming_uniform_ as _kaiming_uniform_,
    kaiming_normal_ as _kaiming_normal_,
    trunc_normal_ as _trunc_normal_,
    calculate_gain as _calculate_gain,
)

from . import nn_init as init  # noqa: E402,F401

from . import nn_functional as functional  # noqa: E402,F401
from . import nn_functional as F  # noqa: E402,F401
