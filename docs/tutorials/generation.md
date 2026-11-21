# Tutorial — Text Generation

**Notebook:** [`generation.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/generation.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/generation.ipynb))

## What you'll build

A minimal causal-LM generation loop using the SNEPPX `Transformer` and the
sampling/beam-search utilities in `SneppX_ALG.interface_bindings.generation`.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np
from SneppX_ALG import Transformer, Tokenizer, Tensor
from SneppX_ALG.interface_bindings.generation import (
    generate, GenerationConfig, TextStreamer, TokenStreamer,
)
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. A tiny model

We keep vocab/size small so the NumPy path is fast enough to preview behavior.

```python
model = Transformer(
    vocab_size=200, dim=64, num_heads=4, num_layers=2,
    ffn_dim=128, max_seq_len=64, dropout=0.0,
)
model.eval()
```

## 2. Greedy search (do_sample=False)

```python
cfg = GenerationConfig(max_new_tokens=24, do_sample=False)
ids = [1, 2, 3]                       # BOS, token, token
out = generate(model, ids, generation_config=cfg)
print(out["output_ids"])              # (1, 3 + 24)
```

## 3. Sampling with top-k + top-p

```python
cfg = GenerationConfig(
    max_new_tokens=48,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.15,
)
out = generate(model, ids, generation_config=cfg)
print("final length:", out["output_ids"].shape)
```

`generate` picks the strategy automatically:

| Condition | Strategy |
|-----------|----------|
| `num_beams > 1` | beam search |
| `do_sample and temperature > 1e-6` | sampling (top-k/top-p) |
| otherwise | greedy |

## 4. Beam search

```python
cfg = GenerationConfig(
    max_new_tokens=32,
    num_beams=3,
    length_penalty=0.7,
    early_stopping=True,
)
out = generate(model, ids, generation_config=cfg)
```

## 5. Stream tokens live

```python
tok = Tokenizer(vocab_size=200)
stream = TextStreamer(tokenizer=tok, skip_prompt=True)
out = generate(model, ids, generation_config=GenerationConfig(max_new_tokens=40), streamer=stream)
# tokens print as they are produced
```

For raw IDs instead of text, swap `TextStreamer` for `TokenStreamer` and read
`stream.get_tokens()`.

## 6. Batch generation (variable-length prompts)

```python
from SneppX_ALG.interface_bindings.generation import batch_generate

prompts = [[1,2,3], [1,4,5,6,7], [1,8,9]]
cfg = GenerationConfig(max_new_tokens=16, temperature=0.7)
out = batch_generate(model, prompts, generation_config=cfg)
print(out["output_ids"].shape)       # (3, max_len + 16)
```

## Key takeaways

- `generate` expects a model whose `forward(input_ids=..., past_key_values=...,
  use_cache=True)` returns `{"logits", "past_key_values"}`; `Transformer`
  returns logits directly, so wrap it for true KV-cache decoding in
  production (see `docs/API.md`).
- CPU-safe NumPy sampling loop works without the C backend — just slower.
- `repetition_penalty`, `top_k`, `top_p`, and `stop_strings` are all
  configurable via `GenerationConfig` (or `**kwargs` override).

## Next steps

- Pair with a real `Tokenizer` (HF `tokenizers` JSON) — see
  [Data Pipeline](data_pipeline.md).
- Serve the model with `sneppx-serve` — see
  [Quantization + Serving](quantization_serving.md).
