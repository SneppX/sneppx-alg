"""Tokenizer — unified encode/decode interface with HuggingFace ``tokenizers`` support."""

import json
import os
from typing import List, Optional, Union, Dict, Any


class Tokenizer:
    """Unified tokenizer interface.

    Wraps either a HuggingFace ``tokenizers`` model or a built-in
    ``SimpleTokenizer`` fallback.
    """

    def __init__(self, path: Optional[str] = None, vocab_size: int = 32000):
        self._pad_token_id = 0
        self._unk_token_id = 1
        self._bos_token_id = 2
        self._eos_token_id = 3
        self._tokenizer = None
        self._simple = None

        if path is not None:
            self._try_load(path)
        else:
            self._simple = SimpleTokenizer(vocab_size=vocab_size)

        if self._tokenizer is None and self._simple is None:
            self._simple = SimpleTokenizer(vocab_size=vocab_size)

    def _try_load(self, path: str):
        """Try loading as HF tokenizers JSON or tokenizer.json."""
        json_path = path
        if os.path.isdir(path):
            json_path = os.path.join(path, "tokenizer.json")
        if os.path.isfile(json_path) and json_path.endswith(".json"):
            try:
                from tokenizers import Tokenizer as HFTokenizer

                self._tokenizer = HFTokenizer.from_file(json_path)
                if hasattr(self._tokenizer, "get_vocab_size"):
                    self._vocab_size = self._tokenizer.get_vocab_size()

                eos = self._tokenizer.token_to_id("<|endoftext|>")
                if eos is not None:
                    self._eos_token_id = eos
                bos = self._tokenizer.token_to_id("<s>")
                if bos is not None:
                    self._bos_token_id = bos
                pad = self._tokenizer.token_to_id("<pad>")
                if pad is not None:
                    self._pad_token_id = pad
                return
            except Exception:
                pass

        # Fallback: load as SimpleTokenizer vocab JSON
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "vocab" in data:
                self._simple = SimpleTokenizer()
                self._simple.vocab = data["vocab"]
                self._simple.inv_vocab = {v: k for k, v in data["vocab"].items()}
            elif isinstance(data, list):
                self._simple = SimpleTokenizer(vocab_size=len(data))
                self._simple.vocab = {w: i for i, w in enumerate(data)}
                self._simple.inv_vocab = {i: w for i, w in enumerate(data)}
        except Exception:
            self._simple = SimpleTokenizer()

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return len(self._simple.vocab) if self._simple else 32000

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def unk_token_id(self) -> int:
        return self._unk_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Tokenize text to token IDs."""
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(text).ids
        elif self._simple is not None:
            ids = self._simple.encode(text)
        else:
            ids = []
        if add_special_tokens:
            ids = [self._bos_token_id] + ids + [self._eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        ids = token_ids
        if skip_special_tokens:
            special = {
                self._pad_token_id,
                self._bos_token_id,
                self._eos_token_id,
                self._unk_token_id,
            }
            ids = [t for t in ids if t not in special]
        if self._tokenizer is not None:
            return self._tokenizer.decode(ids)
        elif self._simple is not None:
            return self._simple.decode(ids)
        return ""

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode_batch(
        self, batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Basic chat template (for models without a template, uses simple format)."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            else:
                parts.append(content)
        parts.append("<|assistant|>\n")
        return "".join(parts)


class SimpleTokenizer:
    """Basic word-level tokenizer for fallback use."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab: Dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size = vocab_size

    def train(self, texts: List[str], min_freq: int = 2):
        from collections import Counter

        counter = Counter()
        for text in texts:
            counter.update(text.strip().split())
        for word, freq in counter.most_common(self.vocab_size - len(self.vocab)):
            if freq >= min_freq:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inv_vocab[idx] = word

    def encode(self, text: str) -> List[int]:
        tokens = text.strip().split()
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.inv_vocab.get(i, "<unk>") for i in ids)


__all__ = ["Tokenizer", "SimpleTokenizer"]
