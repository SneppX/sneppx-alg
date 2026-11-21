# Cookbook — Data & Tokenization

## 1. TensorDataset + DataLoader

**Intent:** Wrap tensors into a batched iterable.

```python
from SneppX_ALG import Tensor, TensorDataset
from SneppX_ALG.interface_bindings.data_loader import DataLoader

x = Tensor.randn((100, 8))
y = Tensor.randn((100, 1))
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)
for xb, yb in loader:        # tensors of shape (16, 8) / (16, 1)
    ...
```

**Notes:** `DataLoader` lives in `interface_bindings.data_loader` (not in the
top-level `*` re-export). The basic version shuffles with `np.random`.
CPU-safe.

## 2. Custom Dataset

**Intent:** Subclass the base `Dataset`.

```python
from SneppX_ALG import Dataset, Tensor

class CSVDataset(Dataset):
    def __init__(self, path):
        import numpy as np
        arr = np.loadtxt(path, delimiter=",")
        self.data    = Tensor(arr[:, :-1])
        self.targets = Tensor(arr[:, -1:])
    def __len__(self):   return len(self.data)
    def __getitem__(self, i): return self.data[i], self.targets[i]
```

**Notes:** Two `Dataset` classes exist (`data.Dataset` and
`data_loader.Dataset`) — both are re-exported; prefer `data_loader.Dataset`
for DataLoader interop.

## 3. Tokenize text (HuggingFace tokenizers)

**Intent:** Production BPE/WordPiece via `tokenizers`.

```python
from SneppX_ALG import SimpleTokenizer
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer

tok = Tokenizer(path="path/to/tokenizer.json")  # HF tokenizers JSON
ids = tok.encode("Hello, SNEPPX!", add_special_tokens=True)
text = tok.decode(ids, skip_special_tokens=True)
print(tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.vocab_size)
```

**Notes:** Falls back to `SimpleTokenizer` (word-level) if `tokenizers` or the
path is unavailable. `Tokenizer` is in `interface_bindings.tokenizer`.

## 4. SimpleTokenizer fallback (no HF deps)

**Intent:** Quick vocab for prototyping.

```python
from SneppX_ALG import SimpleTokenizer

tok = SimpleTokenizer(vocab_size=1000)
tok.train(["hello world sneppx", "hello neural engine"], min_freq=1)
ids = tok.encode("hello sneppx")
print(tok.decode(ids))
```

**Notes:** `SimpleTokenizer` reserves `{<pad>:0, <unk>:1, <s>:2, </s>:3}`.
CPU-safe, no dependencies.

## 5. Chat template formatting

**Intent:** Format multi-turn conversations.

```python
from SneppX_ALG.interface_bindings.tokenizer import Tokenizer

tok = Tokenizer(vocab_size=32000)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is SNEPPX?"},
]
prompt = tok.apply_chat_template(messages)
ids = tok.encode(prompt)
```

**Notes:** `apply_chat_template` uses `<|system|>` / `<|user|>` /
`<|assistant|>` tags and appends a trailing `<|assistant|>`. CPU-safe.

## 6. Streaming token dataset (large corpora)

**Intent:** Don't load the whole corpus into RAM.

```python
from SneppX_ALG.interface_bindings.data_loader import MemoryMappedTextDataset, StreamingTokenDataset

ds = MemoryMappedTextDataset(path="/data/big_corpus.txt", tokenizer=tok, seq_len=1024)
stream = StreamingTokenDataset(path="/data/stream.jsonl", tokenizer=tok, seq_len=512)
```

**Notes:** `MemoryMappedTextDataset` uses `mmap` for random access;
`StreamingTokenDataset` yields one shard at a time for webdatasets.
