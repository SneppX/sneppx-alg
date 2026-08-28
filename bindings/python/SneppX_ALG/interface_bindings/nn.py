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


class CosineEmbeddingLoss(_Loss):
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, input: Tensor, other: Tensor, y: Tensor) -> Tensor:
        return input.cosine_embedding_loss(other, y, self.margin)


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

