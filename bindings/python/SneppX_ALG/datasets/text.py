"""Text datasets — vocabularies and tokenization helpers (pure Python)."""

from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional


class Vocabulary:
    """A token <-> id mapping with optional special tokens.

    Example::

        v = Vocabulary.from_corpus(["the cat sat", "the cat ran"])
        ids = v.encode(["the", "cat"])
        v.decode(ids)  # ['the', 'cat']
    """

    def __init__(self, specials: Iterable[str] = ("<unk>", "<pad>")):
        self._tok2id: Dict[str, int] = {}
        self._id2tok: List[str] = []
        self._counts: Counter = Counter()
        self.specials = list(specials)
        for tok in self.specials:
            self._add_token(tok)

    def _add_token(self, token: str) -> int:
        if token not in self._tok2id:
            self._tok2id[token] = len(self._id2tok)
            self._id2tok.append(token)
        return self._tok2id[token]

    def add(self, token: str) -> int:
        """Add a token (idempotent) and return its id."""
        return self._add_token(token)

    def __len__(self) -> int:
        return len(self._id2tok)

    def __contains__(self, token: str) -> bool:
        return token in self._tok2id

    def __getitem__(self, token: str) -> int:
        return self.token_to_id(token)

    @property
    def pad_id(self) -> Optional[int]:
        return self._tok2id.get("<pad>")

    @property
    def unk_id(self) -> Optional[int]:
        return self._tok2id.get("<unk>")

    def token_to_id(self, token: str, default: Optional[int] = None) -> int:
        if token in self._tok2id:
            return self._tok2id[token]
        if default is not None:
            return default
        return self._tok2id.get("<unk>", 0)

    def id_to_token(self, idx: int) -> str:
        return self._id2tok[idx]

    def encode(
        self,
        tokens: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: Optional[int] = None,
    ) -> List[int]:
        ids = [self.token_to_id(t) for t in tokens]
        if add_bos:
            ids = [self._tok2id.get("<bos>", self.unk_id or 0)] + ids
        if add_eos:
            ids = ids + [self._tok2id.get("<eos>", self.unk_id or 0)]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self._id2tok[i] for i in ids]

    def most_common(self, n: int):
        return self._counts.most_common(n)

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable,
        tokenizer: Callable[[str], List[str]] = str.split,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Iterable[str] = ("<unk>", "<pad>", "<bos>", "<eos>"),
    ) -> "Vocabulary":
        """Build a vocabulary from a corpus of strings or token sequences."""
        counts: Counter = Counter()
        for doc in corpus:
            tokens = tokenizer(doc) if isinstance(doc, str) else list(doc)
            counts.update(tokens)
        vocab = cls(specials=specials)
        for token, freq in counts.most_common():
            if freq < min_freq:
                continue
            vocab._counts[token] = freq
            vocab._add_token(token)
            if max_size is not None and len(vocab) >= max_size:
                break
        return vocab


class CharVocabulary(Vocabulary):
    """Character-level vocabulary: tokenizer splits a string into characters."""

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Iterable[str] = ("<unk>", "<pad>", "<bos>", "<eos>"),
    ) -> "CharVocabulary":
        def char_tokenizer(s: str) -> List[str]:
            return list(s)

        return super().from_corpus(
            corpus,
            tokenizer=char_tokenizer,
            min_freq=min_freq,
            max_size=max_size,
            specials=specials,
        )
