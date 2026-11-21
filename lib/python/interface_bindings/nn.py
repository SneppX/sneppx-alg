"""Neural Network Module — layers, activations, containers."""

from typing import Callable, List, Optional, Tuple, Union
from .core import Tensor
import numpy as np
import math


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True
        self._name = self.__class__.__name__

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        params = []
        for p in self._parameters.values():
            if isinstance(p, Tensor):
                params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self):
        named = []
        for name, p in self._parameters.items():
            if isinstance(p, Tensor):
                named.append((name, p))
        for name, m in self._modules.items():
            for n, p in m.named_parameters():
                named.append((f"{name}.{n}", p))
        return named

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
            self._parameters[name] = value
        super().__setattr__(name, value)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.randn((out_features, in_features), dtype=dtype) / math.sqrt(in_features)
        self.bias = Tensor.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype="float32"):
        super().__init__()
        self.weight = Tensor.randn((num_embeddings, embedding_dim), dtype=dtype) * 0.01

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.numpy().astype(np.int64)
        w = self.weight.numpy()
        return Tensor(w[idx], dtype=self.weight.dtype)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1.0 - self.p, x.shape).astype(np.float32)
        mask /= (1.0 - self.p)
        return Tensor(x.numpy() * mask, dtype=x.dtype)


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, dtype="float32"):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape, dtype=dtype)
        self.bias = Tensor.zeros(normalized_shape, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        arr = x.numpy()
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = arr.mean(axis=axis, keepdims=True)
        var = arr.var(axis=axis, keepdims=True)
        out = (arr - mean) / np.sqrt(var + self.eps)
        out = out * self.weight.numpy() + self.bias.numpy()
        return Tensor(out, dtype=x.dtype)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones((dim,), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        arr = x.numpy()
        rms = np.sqrt((arr ** 2).mean(axis=-1, keepdims=True) + self.eps)
        return Tensor(arr / rms * self.weight.numpy(), dtype=x.dtype)


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
        return x.tanh()


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, dtype="float32"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.o_proj = Linear(embed_dim, embed_dim, dtype=dtype)
        self.dropout = Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor = None, v: Tensor = None, mask: Tensor = None) -> Tensor:
        if k is None: k = q
        if v is None: v = q
        B, T, C = q.shape
        H = self.num_heads
        D = self.head_dim
        q = self.q_proj(q).numpy().reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(k).numpy().reshape(B, -1, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(v).numpy().reshape(B, -1, H, D).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask.numpy()
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.o_proj(Tensor(out, dtype=q.dtype))


class TransformerBlock(Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0, dtype="float32"):
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
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_layers: int,
                 ffn_dim: int, max_seq_len: int = 2048, dropout: float = 0.0, dtype="float32"):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, dim, dtype)
        self.pos_embedding = Embedding(max_seq_len, dim, dtype)
        self.blocks = Sequential(*[
            TransformerBlock(dim, num_heads, ffn_dim, dropout, dtype)
            for _ in range(num_layers)
        ])
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