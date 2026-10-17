# Tutorial — Data Pipeline

**Notebook:** [`data_pipeline.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/data_pipeline.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/data_pipeline.ipynb))

## What you'll build

A complete data pipeline: train a `SimpleTokenizer` on text, encode/decode,
batch with `DataLoader` + `DistributedSampler`, and serve batches to a model.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np, os
from SneppX_ALG import SimpleTokenizer, Tokenizer, Tensor
from SneppX_ALG.interface_bindings.data_loader import DataLoader, TensorDataset
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. SimpleTokenizer (no HF deps)

```python
corpus = ["hello world sneppx", "hello neural engine", "sneppx is fast",
          "neural networks rock", "sneppx sneppx hello"]

tok = SimpleTokenizer(vocab_size=50)
tok.train(corpus, min_freq=1)
ids = tok.encode("hello sneppx")
print("ids:", ids)
print("text:", tok.decode(ids))
```

## 2. Production tokenizer (HF tokenizers JSON)

```python
# If you have a tokenizer.json from HuggingFace:
if os.path.exists("tokenizer.json"):
    ptok = Tokenizer("tokenizer.json")
    print("vocab size:", ptok.vocab_size)
    print("bos/eos/pad:", ptok.bos_token_id, ptok.eos_token_id, ptok.pad_token_id)
    print(ptok.decode(ptok.encode("Hello, world!")))
else:
    print("no tokenizer.json present — using SimpleTokenizer")
```

## 3. Chat template formatting

```python
ptok = Tokenizer(vocab_size=32000)     # falls back to SimpleTokenizer
messages = [
    {"role": "system", "content": "You are SNEPPX, a helpful assistant."},
    {"role": "user",   "content": "Summarize neural programming engines."},
]
prompt = ptok.apply_chat_template(messages)
ids = ptok.encode(prompt)
print(prompt)
```

## 4. TensorDataset + DataLoader

```python
X = np.random.randint(0, 300, (200, 16)).astype(np.int64)   # token ids
Y = np.random.randint(0, 300, (200, 16)).astype(np.int64)   # labels
ds = TensorDataset(Tensor.from_numpy(X), Tensor.from_numpy(Y))
loader = DataLoader(ds, batch_size=32, shuffle=True)

for xb, yb in loader:
    print("batch:", xb.shape, yb.shape)    # (32, 16) (32, 16)
```

## 5. Streaming large corpora (mmap)

```python
from SneppX_ALG.interface_bindings.data_loader import (
    MemoryMappedTextDataset, StreamingTokenDataset,
)

# Random-access over a huge file via mmap
mmap_ds = MemoryMappedTextDataset(path="big_corpus.txt", tokenizer=tok, seq_len=512)

# Streaming webdataset-style shard reader
stream = StreamingTokenDataset(path="data/shard_*.jsonl", tokenizer=tok, seq_len=1024)
```

## 6. Distributed sharding

```python
from SneppX_ALG import DistributedSampler, get_world_size, get_rank

sampler = DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
loader  = DataLoader(ds, batch_size=32, sampler=sampler)
sampler.set_epoch(epoch)      # re-shuffle per epoch
```

## Key takeaways

- `SimpleTokenizer` (word-level) needs **no dependencies**; `Tokenizer` wraps
  HF `tokenizers` for BPE/WordPiece when available.
- Special tokens: `<pad>=0, <unk>=1, <s>=2, </s>=3`.
- `DataLoader` lives in `interface_bindings.data_loader` — **not** re-exported
  via `from SneppX_ALG import *`.
- `apply_chat_template` emits `<|system|>` / `<|user|>` / `<|assistant|>` tags.
- CPU-safe throughout (no C backend needed for the data pipeline).

## Next steps

- Tokenize + generate text — see
  [Text Generation](generation.md).
- Build a language-model dataset for distributed training — see
  [Distributed Training](distributed_training.md).
