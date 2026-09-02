"""Neural Network Module — layers, activations, containers with CUDA kernel support."""

from typing import Callable, List, Optional, Tuple, Union
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


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
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

    def register_buffer(self, name: str, tensor) -> None:
        object.__setattr__(self, name, tensor)
        self._buffers[name] = tensor

    def named_buffers(self):
        named = []
        for name, b in self._buffers.items():
            if isinstance(b, Tensor):
                named.append((name, b))
        for name, m in self._modules.items():
            for n, b in m.named_buffers():
                named.append((f"{name}.{n}", b))
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
        for name, b in self.named_buffers():
            sd[name] = b.data.copy()
        return sd

    def load_state_dict(self, state_dict: dict):
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = state_dict[name]
        for name, b in self.named_buffers():
            if name in state_dict:
                b.data = state_dict[name]

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
        idx = indices.data.astype(np.int64)
        w = self.weight.data
        return Tensor.from_numpy(w[idx])


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
        return Tensor.from_numpy(x.data * mask, dtype=x.dtype_name)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape, dtype=dtype)
        self.bias = Tensor.zeros(normalized_shape, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if _is_cuda_tensor(x):
            out_data = np.empty_like(x.data)
            layernorm_kernel(
                x.data,
                self.weight.data,
                self.bias.data,
                out_data,
                self.normalized_shape,
                self.eps,
            )
            return Tensor(out_data, dtype=x.dtype_name, device=x.device)
        arr = x.data
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = arr.mean(axis=axis, keepdims=True)
        var = arr.var(axis=axis, keepdims=True)
        out = (arr - mean) / np.sqrt(var + self.eps)
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
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, kH, kW), dtype=dtype
        ) * scale
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .advanced_ops import conv2d as _conv2d

        return _conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d(Module):
    """1D convolution layer."""

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
        kL = kernel_size[0]
        scale = math.sqrt(1.0 / (in_channels * kL))
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, kL), dtype=dtype
        ) * scale
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .advanced_ops import conv1d as _conv1d

        return _conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride[0] if isinstance(self.stride, (list, tuple)) else self.stride,
            padding=self.padding[0] if isinstance(self.padding, (list, tuple)) else self.padding,
            dilation=self.dilation[0] if isinstance(self.dilation, (list, tuple)) else self.dilation,
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
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, kD, kH, kW), dtype=dtype
        ) * scale
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .advanced_ops import conv3d as _conv3d

        return _conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
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
        self.weight = Tensor.randn(
            (in_channels // groups, out_channels) + tuple(kernel_size), dtype=dtype
        ) * scale
        self.bias = Tensor.zeros((out_channels,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if self.conv_ndim != 2:
            raise NotImplementedError(
                f"ConvTranspose{self.conv_ndim}d backward/forward only supported for 2d in this sim"
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
        arr = x.data
        rms = np.sqrt((arr**2).mean(axis=-1, keepdims=True) + self.eps)
        return Tensor(arr / rms * self.weight.data, dtype=x.dtype_name, device=x.device)


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

    def named_buffers(self):
        named = []
        for i, m in enumerate(self._list):
            if hasattr(m, "named_buffers"):
                for n, b in m.named_buffers():
                    named.append((f"{i}.{n}", b))
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
                sub = {k[len(f"{i}."):]: v for k, v in state_dict.items() if k.startswith(f"{i}.")}
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

    def named_buffers(self):
        named = []
        for k, m in self._dict.items():
            if hasattr(m, "named_buffers"):
                for n, b in m.named_buffers():
                    named.append((f"{k}.{n}", b))
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
                sub = {kk[len(f"{k}."):]: v for kk, v in state_dict.items() if kk.startswith(f"{k}.")}
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
        raise NotImplementedError("ModuleDict has no default forward; index modules explicitly.")


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
        i, f, g, o = lin[:, : self.hidden_size], lin[:, self.hidden_size:2 * self.hidden_size], \
            lin[:, 2 * self.hidden_size:3 * self.hidden_size], lin[:, 3 * self.hidden_size:]
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
        z = (lin_ih[:, self.hidden_size:2 * self.hidden_size] + lin_hh[:, self.hidden_size:2 * self.hidden_size]).sigmoid()
        n = (lin_ih[:, 2 * self.hidden_size:] + r * lin_hh[:, 2 * self.hidden_size:]).tanh()
        return (1 - z) * n + z * h


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh",
                 bias=True, batch_first=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity)
             for i in range(num_layers)]
        )

    def forward(self, x: Tensor, h0=None):
        if self.batch_first:
            x = x.transpose(1, 0) if x.ndim == 3 else x
        T = x.shape[0]
        batch = x.shape[1]
        if h0 is None:
            h = [Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
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
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
             for i in range(num_layers)]
        )

    def forward(self, x: Tensor, hc0=None):
        if self.batch_first:
            x = x.transpose(1, 0)
        T = x.shape[0]
        batch = x.shape[1]
        if hc0 is None:
            h = [Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
            c = [Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
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
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = ModuleList(
            [GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias)
             for i in range(num_layers)]
        )

    def forward(self, x: Tensor, h0=None):
        if self.batch_first:
            x = x.transpose(1, 0)
        T = x.shape[0]
        batch = x.shape[1]
        if h0 is None:
            h = [Tensor.zeros((batch, self.hidden_size)) for _ in range(self.num_layers)]
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
    def forward(ctx, x, weight, bias, running_mean, running_var, training, momentum, eps):
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
        ctx.save_attr(inv=inv, xhat=xhat, w=weight.data.reshape(keep), axes=axes, N=N, training=training)
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
            dx = N * grad_xhat - grad_xhat.sum(axis=axes, keepdims=True) - xhat * (grad_xhat * xhat).sum(axis=axes, keepdims=True)
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
        inv_ch = np.repeat(np.sqrt(var + eps).reshape(N, G, 1, 1), Cg, axis=1).reshape(N, C, 1, 1)  # (N, C, 1, 1)
        ctx.save_attr(inv=inv_ch, xhat=xhat, w=weight.data.reshape(keep), N=Nred, num_groups=G)
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
        inv_g = inv.reshape(g.shape[0], num_groups, Cg, 1, 1)[:, :, 0].reshape(g.shape[0], num_groups, 1, 1, 1)
        rax = tuple(range(2, gg.ndim))           # (Cg, *spatial) within each (n, g)
        sum_g = gg.sum(axis=rax, keepdims=True)
        sum_gx = (gg * xh).sum(axis=rax, keepdims=True)
        dxg = (N * gg - sum_g - xh * sum_gx) / N / inv_g
        dx = dxg.reshape(g.shape)
        sax = tuple(range(3, gg.ndim))           # spatial dims
        grad_w = (gg * xh).sum(axis=(0,) + sax).reshape(C)
        grad_b = gg.sum(axis=(0,) + sax).reshape(C)
        return [
            Tensor(dx, dtype=grad_output.dtype),
            Tensor(grad_w, dtype=grad_output.dtype),
            Tensor(grad_b, dtype=grad_output.dtype),
            None,
        ]


class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
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
        ctx.save_attr(inv=1.0 / np.sqrt(var + eps), xhat=xhat, w=weight.data.reshape(keep), axes=axes)
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
        dx = N * grad_xhat - grad_xhat.sum(axis=axes, keepdims=True) - xhat * (grad_xhat * xhat).sum(axis=axes, keepdims=True)
        dx = dx / N * inv
        return [
            Tensor(dx, dtype=grad_output.dtype),
            Tensor(grad_w, dtype=grad_output.dtype),
            Tensor(grad_b, dtype=grad_output.dtype),
            None,
            None,
        ]


class _InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
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
        start_dim = self.start_dim if self.start_dim >= 0 else len(shape) + self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        flat = 1
        for s in shape[start_dim:end_dim + 1]:
            flat *= s
        new_shape = shape[:start_dim] + [flat] + shape[end_dim + 1:]
        return x.reshape(new_shape)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Bilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Tensor.randn((out_features, in1_features, in2_features), dtype=dtype) * 0.1
        self.bias = Tensor.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        x1 = input1.data
        x2 = input2.data
        w = self.weight.data
        out = np.einsum("bi,bj,kij->bk", x1, x2, w)
        if self.bias is not None:
            out = out + self.bias.data
        return Tensor(out, dtype=input1.dtype_name, device=input1.device)


# ===========================================================================
# Loss functions (torch.nn-compatible Module wrappers)
# ===========================================================================


class _Loss(Module):
    """Base class for loss modules (mirrors torch.nn.modules.loss._Loss)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction


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
        x = input.data
        y = target.data
        v = var.data
        eps = self.eps
        loss = 0.5 * (
            np.log(2 * np.pi) + np.log(v + eps) + (x - y) ** 2 / (v + eps)
        )
        if self.full:
            loss = 0.5 * (np.log(2 * np.pi * v + eps) + (x - y) ** 2 / (v + eps))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


class HingeEmbeddingLoss(_Loss):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input.data
        y = target.data
        # y in {-1, +1}
        loss = np.where(y >= 0, x, np.maximum(0.0, self.margin - x))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


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
        x = input.data
        y = target.data
        loss = np.exp(x) - y * x
        if self.full:
            loss = loss + y * np.log(y + self.eps) - y + y * np.log(2 * np.pi) / 2
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


class SoftMarginLoss(_Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input.data
        y = target.data
        loss = np.log1p(np.exp(-y * x))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


class MultiMarginLoss(_Loss):
    def __init__(self, p: int = 1, margin: float = 1.0, weight=None, reduction: str = "mean"):
        super().__init__(reduction)
        self.p = p
        self.margin = margin
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input.data
        y = target.data.astype(np.int64)
        n = x.shape[0]
        c = x.shape[1]
        target_scores = x[np.arange(n), y]
        mask = np.arange(c) != y[:, None]
        margin_diff = self.margin - (x - target_scores[:, None])
        margin_diff = np.where(mask, margin_diff, 0)  # zero out the target class
        losses = np.maximum(0, np.power(np.maximum(margin_diff, 0), self.p))
        if self.weight is not None:
            w = self.weight.data if hasattr(self.weight, "data") else self.weight
            losses = losses * w[None, :]
        loss = losses.sum(axis=1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


class MultiLabelMarginLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input.data
        y = target.data
        n, c = x.shape
        loss_sum = 0.0
        for i in range(n):
            targets_i = [int(t) for t in y[i] if t >= 0]
            for tn in targets_i:
                for k in range(c):
                    if k not in targets_i:
                        loss_sum += float(np.maximum(0, 1 - x[i, tn] + x[i, k]))
        loss = loss_sum / float(n * c)
        if self.reduction == "sum":
            loss = np.array(loss_sum)
        return Tensor(np.array(loss, dtype=np.float32))


class MultiLabelSoftMarginLoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        x = input.data
        y = target.data
        # per-element BCE (multi-label, many-hot targets)
        loss = -(y * np.log(np.clip(x, 1e-7, 1.0)) + (1 - y) * np.log(np.clip(1 - x, 1e-7, 1.0)))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return Tensor(np.array(loss, dtype=np.float32))


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
        n, c, l = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation
        if p > 0:
            x_data = np.pad(x.data, ((0, 0), (0, 0), (p, p)), mode="constant")
            l = l + 2 * p
        else:
            x_data = x.data
        l_out = (l - d * (k - 1) - 1) // s + 1
        out = np.zeros((n, c, l_out), dtype=np.float32)
        for i in range(l_out):
            start = i * s
            end = start + d * k
            window = x_data[:, :, start:end]
            if self.pool_type == "max":
                out[:, :, i] = window.max(axis=2).astype(np.float32)
            else:
                out[:, :, i] = window.mean(axis=2).astype(np.float32)
        return Tensor(out, dtype=x.dtype, device=x.device)


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
        from .advanced_ops import max_pool2d as _max_pool2d
        from .advanced_ops import avg_pool2d as _avg_pool2d

        if self.pool_type == "max":
            return _max_pool2d(
                x, self.kernel_size, self.stride, self.padding, self.dilation
            )
        return _avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )


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
        n, c, dep, h, w = x.shape
        kd, kh, kw = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        dD, dH, dW = self.dilation
        if pD > 0 or pH > 0 or pW > 0:
            x_data = np.pad(
                x.data,
                ((0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)),
                mode="constant",
            )
            dep = dep + 2 * pD
            h = h + 2 * pH
            w = w + 2 * pW
        else:
            x_data = x.data
        out_d = (dep - dD * (kd - 1) - 1) // sD + 1
        out_h = (h - dH * (kh - 1) - 1) // sH + 1
        out_w = (w - dW * (kw - 1) - 1) // sW + 1
        out = np.zeros((n, c, out_d, out_h, out_w), dtype=np.float32)
        for i in range(out_d):
            for j in range(out_h):
                for k in range(out_w):
                    d_start = i * sD
                    d_end = d_start + dD * kd
                    h_start = j * sH
                    h_end = h_start + dH * kh
                    w_start = k * sW
                    w_end = w_start + dW * kw
                    window = x_data[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    if self.pool_type == "max":
                        out[:, :, i, j, k] = window.max(axis=(2, 3, 4))
                    else:
                        out[:, :, i, j, k] = window.mean(axis=(2, 3, 4))
        return Tensor(out, dtype=x.dtype, device=x.device)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, l = x.shape
        out_l = self.output_size[0]
        out = np.zeros((n, c, out_l), dtype=np.float32)
        for i in range(out_l):
            start = int(np.floor(i * l / out_l))
            end = int(np.ceil((i + 1) * l / out_l))
            if end > start:
                out[:, :, i] = x.data[:, :, start:end].mean(axis=2).astype(np.float32)
        return Tensor(out, dtype=x.dtype, device=x.device)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .advanced_ops import adaptive_avg_pool2d as _aap2d

        return _aap2d(x, self.output_size)


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, d, h, w = x.shape
        od, oh, ow = self.output_size
        out = np.zeros((n, c, od, oh, ow), dtype=np.float32)
        for i in range(od):
            d0 = int(np.floor(i * d / od))
            d1 = int(np.ceil((i + 1) * d / od))
            for j in range(oh):
                h0 = int(np.floor(j * h / oh))
                h1 = int(np.ceil((j + 1) * h / oh))
                for k in range(ow):
                    w0 = int(np.floor(k * w / ow))
                    w1 = int(np.ceil((k + 1) * w / ow))
                    win = x.data[:, :, d0:max(d1, d0 + 1), h0:max(h1, h0 + 1), w0:max(w1, w0 + 1)]
                    out[:, :, i, j, k] = win.mean(axis=(2, 3, 4)).astype(np.float32)
        return Tensor(out, dtype=x.dtype, device=x.device)


class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, l = x.shape
        out_l = self.output_size[0]
        out = np.zeros((n, c, out_l), dtype=np.float32)
        for i in range(out_l):
            start = int(np.floor(i * l / out_l))
            end = int(np.ceil((i + 1) * l / out_l))
            if end > start:
                out[:, :, i] = x.data[:, :, start:end].max(axis=2).astype(np.float32)
        return Tensor(out, dtype=x.dtype, device=x.device)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        from .advanced_ops import adaptive_max_pool2d as _amp2d

        return _amp2d(x, self.output_size)


class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, d, h, w = x.shape
        od, oh, ow = self.output_size
        out = np.zeros((n, c, od, oh, ow), dtype=np.float32)
        for i in range(od):
            d0 = int(np.floor(i * d / od))
            d1 = int(np.ceil((i + 1) * d / od))
            for j in range(oh):
                h0 = int(np.floor(j * h / oh))
                h1 = int(np.ceil((j + 1) * h / oh))
                for k in range(ow):
                    w0 = int(np.floor(k * w / ow))
                    w1 = int(np.ceil((k + 1) * w / ow))
                    win = x.data[:, :, d0:max(d1, d0 + 1), h0:max(h1, h0 + 1), w0:max(w1, w0 + 1)]
                    out[:, :, i, j, k] = win.max(axis=(2, 3, 4)).astype(np.float32)
        return Tensor(out, dtype=x.dtype, device=x.device)


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
        x_data = np.pad(
            x.data, pad_width, mode="constant", constant_values=self.value
        )
        return Tensor(x_data, dtype=x.dtype, device=x.device)


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
        x_data = np.pad(x.data, pad_width, mode="reflect")
        return Tensor(x_data, dtype=x.dtype, device=x.device)


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
        x_data = np.pad(x.data, pad_width, mode="edge")
        return Tensor(x_data, dtype=x.dtype, device=x.device)


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
        return Tensor(np.maximum(x.data * self.negative_slope, x.data), dtype=x.dtype_name, device=x.device)


class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = Tensor.full((num_parameters,), init)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(x.data, self.weight.data * x.data), dtype=x.dtype_name, device=x.device)


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1)), dtype=x.dtype_name, device=x.device)


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
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
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
    def __init__(self, lower: float = 0.125, upper: float = 0.3333333333333333, inplace: bool = False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self._training:
            slope = (self.lower + self.upper) / 2.0
            return x.leaky_relu(slope)
        xd = x.data
        mask = (np.random.uniform(self.lower, self.upper, xd.shape).astype(xd.dtype)) * (xd < 0).astype(xd.dtype)
        out = xd * (xd >= 0).astype(xd.dtype) + mask * xd
        return Tensor(out, dtype=x.dtype_name, device=x.device)


class LogSigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        out = -np.log1p(np.exp(-np.abs(xd))) + np.minimum(xd, 0)
        return Tensor(out, dtype=x.dtype_name, device=x.device)


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
        idx = input.data.astype(np.int64).reshape(-1)
        w = self.weight.data
        emb = w[idx]
        if per_sample_weights is not None:
            psw = per_sample_weights.data.astype(np.float64).reshape(-1)
            emb = emb * psw[:, None]
        if offsets is None:
            if self.mode == "sum":
                agg = emb.sum(axis=0, keepdims=True)
            elif self.mode == "mean":
                agg = emb.mean(axis=0, keepdims=True)
            else:  # max
                agg = emb.max(axis=0, keepdims=True)
            return Tensor(agg, dtype=self.weight.dtype, device=self.weight.device)
        off = offsets.data.astype(np.int64)
        off = np.clip(off, 0, idx.size)
        n = off.size - 1
        bags = []
        for i in range(n):
            seg = emb[off[i]:off[i + 1]]
            if seg.shape[0] == 0:
                bags.append(np.zeros(emb.shape[1], dtype=np.float32))
            elif self.mode == "sum":
                bags.append(seg.sum(axis=0))
            elif self.mode == "mean":
                bags.append(seg.mean(axis=0))
            else:
                bags.append(seg.max(axis=0))
        return Tensor(np.stack(bags), dtype=self.weight.dtype, device=self.weight.device)


class CosineSimilarity(Module):
    """Computes cosine similarity between x1 and x2 along `dim`."""

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        a = x1.data.astype(np.float64)
        b = x2.data.astype(np.float64)
        dim = self.dim % a.ndim
        num = (a * b).sum(axis=dim)
        den = (
            np.sqrt((a * a).sum(axis=dim) + self.eps)
            * np.sqrt((b * b).sum(axis=dim) + self.eps)
        )
        out = num / np.maximum(den, self.eps)
        return Tensor(out.astype(np.float32), dtype=x1.dtype, device=x1.device)


class PairwiseDistance(Module):
    """Computes the pairwise distance between x1 and x2 using `p`-norm."""

    def __init__(self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        a = x1.data.astype(np.float64)
        b = x2.data.astype(np.float64)
        d = a - b
        if self.p == 2.0:
            out = np.sqrt((d * d).sum(axis=-1) + self.eps)
        elif self.p == 1.0:
            out = np.abs(d).sum(axis=-1)
        else:
            out = np.power(np.abs(d).sum(axis=-1) + self.eps, 1.0 / self.p)
        if self.keepdim:
            out = out[..., None]
        return Tensor(out.astype(np.float32), dtype=x1.dtype, device=x1.device)


class ChannelShuffle(Module):
    """Divides channels into `groups` groups and rearranges them."""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        n, c, *spatial = xd.shape
        cgh = c // self.groups
        out = xd.reshape(n, self.groups, cgh, *spatial).transpose(0, 2, 1, *range(3, 3 + len(spatial)))
        out = out.reshape(n, c, *spatial)
        return Tensor(out, dtype=x.dtype, device=x.device)


class PixelShuffle(Module):
    """Rearranges (C*r^2, H, W) to (C, H*r, W*r)."""

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        r = self.upscale_factor
        n, c, h, w = xd.shape
        c_out = c // (r * r)
        out = xd.reshape(n, c_out, r, r, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(n, c_out, h * r, w * r)
        return Tensor(out, dtype=x.dtype, device=x.device)


class PixelUnshuffle(Module):
    """Inverse of PixelShuffle."""

    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        r = self.downscale_factor
        n, c, h, w = xd.shape
        c_out = c * r * r
        out = xd.reshape(n, c, h // r, r, w // r, r).transpose(0, 1, 3, 5, 2, 4).reshape(n, c_out, h // r, w // r)
        return Tensor(out, dtype=x.dtype, device=x.device)


class Upsample(Module):
    """Upsamples a given multi-channel 1D/2D/3D input (nearest or bilinear)."""

    def __init__(self, size=None, scale_factor=None, mode: str = "nearest", align_corners=None):
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
            target = tuple(int(s) for s in (self.size if isinstance(self.size, (tuple, list)) else (self.size,) * ndim))
        else:
            target = spatial
        scales = tuple(t / s for s, t in zip(spatial, target))
        if self.mode == "bilinear":
            from scipy.ndimage import zoom

            out = zoom(xd, (1.0, 1.0) + scales, order=1, mode="nearest")
        else:  # nearest
            out = _zoom_nn(xd, target)
        return Tensor(out.astype(xd.dtype), dtype=x.dtype, device=x.device)


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

