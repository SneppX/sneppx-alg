# Cookbook — Models & Layers

## 1. Build a feed-forward network with Sequential

**Intent:** Stack layers imperatively.

```python
from SneppX_ALG import Sequential, Linear, ReLU, GELU, Dropout, Tensor

net = Sequential(
    Linear(784, 256), ReLU(),
    Linear(256, 128), GELU(),
    Dropout(0.1),
    Linear(128, 10),
)
x = Tensor.randn((4, 784))
logits = net(x)               # (4, 10)
# CPU-safe; requires C backend to get gradients via .backward()
```

**Notes:** `Sequential` is registered as `nn.Sequential` and also re-exported
by the Keras `keras_api` shim.

## 2. Use a Transformer encoder block

**Intent:** Self-attention + residual MLP.

```python
from SneppX_ALG import TransformerBlock, Tensor

blk = TransformerBlock(dim=512, num_heads=8, ffn_dim=2048, dropout=0.1)
x = Tensor.randn((2, 16, 512))     # (batch, seq, dim)
out = blk(x)                        # (2, 16, 512)
```

**Notes:** `TransformerBlock` uses standard scaled-dot-product attention
(Flash Attention v2 when `_HAS_CUDA`). CPU-safe.

## 3. Construct a full GPT-style Transformer

**Intent:** Token + position embeddings → stacked blocks → LM head.

```python
from SneppX_ALG import Transformer, Tensor

model = Transformer(
    vocab_size=32000, dim=512, num_heads=8,
    num_layers=6, ffn_dim=2048, max_seq_len=1024, dropout=0.1,
)
ids = Tensor.arange(0, 128).unsqueeze(0)   # (1, 128)
logits = model(ids)                         # (1, 128, vocab)
```

**Notes:** `Transformer.forward` returns **logits** (not a dict). Wrap it to
add `past_key_values` for incremental decoding (see
[Generation](generation.md)). Needs C backend to `backward()`.

## 4. Multi-head attention with mask

**Intent:** Attention with an explicit mask tensor.

```python
from SneppX_ALG import MultiheadAttention, Tensor

attn = MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
q = Tensor.randn((2, 16, 512))
mask = Tensor.zeros((16, 16)).fill_(float("-inf"))  # causal-ish
out = attn(q, k=q, v=q, mask=mask)                  # (2, 16, 512)
```

**Notes:** CUDA path uses `flash_attention_v2_kernel`; CPU path uses NumPy.

## 5. RMSNorm (pre-norm LLM style)

**Intent:** Match Llama/Qwen normalization.

```python
from SneppX_ALG import RMSNorm, Tensor

norm = RMSNorm(dim=512, eps=1e-6)
x = Tensor.randn((4, 512))
y = norm(x)
```

## 6. Embedding layer for token IDs

**Intent:** Map integer token IDs to dense vectors.

```python
from SneppX_ALG import Embedding, Tensor

emb = Embedding(num_embeddings=32000, embedding_dim=512)
ids = Tensor([1, 42, 7, 2048])
vecs = emb(ids)          # (4, 512)
```

**Notes:** `forward` does `idx = indices.data.astype(np.int64)`.

## 7. Build a model from a config dict

**Intent:** Translate an HF config dict into a `nn.Transformer`.

```python
from SneppX_ALG import build_transformer_from_config

cfg = {
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 2048,
    "vocab_size": 32000,
    "max_position_embeddings": 1024,
}
model = build_transformer_from_config(cfg)   # nn.Module
```

**Notes:** `build_model_from_config(cfg)` returns a **param-count dict**
instead of a model — use `build_transformer_from_config` when you want the
module. CPU-safe.

## 8. Load a pretrained model config

**Intent:** Inspect a known family (no weights downloaded).

```python
from SneppX_ALG import from_pretrained, get_model_config, list_available_models

print(list_available_models())               # ['llama2:7B', 'llama2:13B', ...]
cfg = get_model_config("llama2", "7B")
info = from_pretrained("llama-2-7b")
print(info["family"], info["size"], info["total_params"])
```

**Notes:** `from_pretrained` returns a **dict with config + param estimate**,
not a runnable model (weights require `convert_hf_to_sneppx`). CPU-safe.

## 9. Keras-compatible functional API

**Intent:** TF/Keras `Input` → layers → `Model`.

```python
from SneppX_ALG import Input, Dense, Dropout, GELU, Model, Tensor, AdamW

inp  = Input(shape=(784,))
x    = Dense(256, activation="relu")(inp)
x    = Dropout(0.2)(x)
out  = Dense(10)(x)
model = Model(inp, out)
logits = model(Tensor.randn((4, 784)))
```

**Notes:** `keras_api.Sequential`, `Conv2D`, `LayerNorm`, `BatchNormalization`
are all available. CPU-safe.

## 10. Parameter counting & state dict

**Intent:** Inspect / serialize weights.

```python
from SneppX_ALG import Transformer

model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=512)

n = sum(p.numel for p in model.parameters())
sd = model.state_dict()          # {name: np.ndarray}
model.load_state_dict(sd)        # round-trip
```

**Notes:** `state_dict()` returns numpy arrays (CPU). CPU-safe.

## 11. Move a model to a device

**Intent:** CPU→CUDA placement.

```python
from SneppX_ALG import Transformer
model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=512)
model.to("cuda") if cuda_available else None   # see cuda_device.cuda_is_available
```

**Notes:** CUDA tensors need `SNEPPX_BUILD_CUDA=ON` + GPU. Without GPU,
`to("cuda")` simulates.

## 12. Custom Module with gradient

**Intent:** Subclass `Module`.

```python
from SneppX_ALG import Module, Linear, Tensor

class MyNet(Module):
    def __init__(self, d=64):
        super().__init__()
        self.l1 = Linear(d, d)
        self.l2 = Linear(d, 1)
    def forward(self, x):
        return self.l2(self.l1(x).relu())

net = MyNet()
y = net(Tensor.randn((4, 64)))
```

**Notes:** Override `forward`; `__call__` is `forward`. CPU-safe to run;
needs C backend for `y.backward()`.

## 13. Vision: ViT patch embedding

**Intent:** Use a ready-made vision model.

```python
from SneppX_ALG import VisionTransformer, create_vision_model, Tensor

vit = create_vision_model("vit_tiny_patch16_224")
x = Tensor.randn((1, 3, 224, 224))
out = vit(x)
```

**Notes:** Registered as `vit_tiny/small/base/large/huge` and `mae_*`.
CPU-safe (NumPy path) but slow without the C backend.

## 14. MAE autoencoder

**Intent:** Masked autoencoder forward.

```python
from SneppX_ALG import mae_base, Tensor

mae = mae_base()
x = Tensor.randn((1, 3, 224, 224))
recon, mask = mae(x)      # reconstruction + mask
```
