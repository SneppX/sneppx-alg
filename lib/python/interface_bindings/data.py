"""Data pipeline — datasets, loaders, tokenizers, preprocessing."""

from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from .core import Tensor
import numpy as np
import json
import os


class Dataset:
    def __init__(self, data, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        if self.targets is not None:
            return x, self.targets[idx]
        return x


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        self.tensors = tensors
        self._len = len(tensors[0]) if tensors else 0

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)

    def __len__(self):
        return self._len


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        tokens = tokens[:self.max_length]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return Tensor(np.array(tokens, dtype=np.int64))


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, texts: List[str], min_freq: int = 2):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        idx = len(self.vocab)
        for word, freq in counter.most_common(self.vocab_size - idx):
            if freq >= min_freq:
                self.vocab[word] = idx
                self.inv_vocab[idx] = word
                idx += 1

    def encode(self, text: str) -> List[int]:
        tokens = [self.vocab.get(w, 1) for w in text.split()]
        return [2] + tokens + [3]

    def decode(self, tokens: List[int]) -> str:
        return " ".join(self.inv_vocab.get(t, "<unk>") for t in tokens)

    def __len__(self):
        return len(self.vocab)


class BatchCollator:
    def __init__(self, padding_value: int = 0, max_length: Optional[int] = None):
        self.padding_value = padding_value
        self.max_length = max_length

    def __call__(self, batch: List) -> Tuple[Tensor, ...]:
        if isinstance(batch[0], tuple):
            xs = [b[0] for b in batch]
            ys = [b[1] for b in batch]
            return self._collate_tensors(xs), self._collate_tensors(ys)
        return self._collate_tensors(batch)

    def _collate_tensors(self, tensors: List) -> Tensor:
        arrs = [t.numpy() if isinstance(t, Tensor) else np.array(t) for t in tensors]
        max_len = self.max_length or max(a.shape[-1] for a in arrs)
        padded = []
        for a in arrs:
            if a.ndim == 1:
                pad_width = max_len - len(a)
                if pad_width > 0:
                    a = np.pad(a, (0, pad_width), constant_values=self.padding_value)
                padded.append(a)
            else:
                padded.append(a)
        return Tensor(np.stack(padded))


class Preprocessor:
    def __init__(self):
        self.transforms = []

    def add(self, fn: Callable):
        self.transforms.append(fn)

    def __call__(self, x):
        for fn in self.transforms:
            x = fn(x)
        return x