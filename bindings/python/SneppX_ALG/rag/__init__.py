"""Retrieval-augmented-generation building blocks (pure Python).

Chunking, lexical (BM25) and dense (hashing embedding) retrieval, and a
hybrid retriever — no model download or network access required.
"""

from . import bm25, chunking, embeddings, retriever, text
from .bm25 import BM25
from .chunking import (
    chunk_by_sentences,
    chunk_by_words,
    chunk_recursive,
    sliding_window,
)
from .embeddings import EmbeddingIndex, cosine_similarity, hashing_embed
from .retriever import Retriever
from .text import normalize_whitespace, split_sentences, tokenize, tokenize_no_stop

__all__ = [
    "text",
    "chunking",
    "bm25",
    "embeddings",
    "retriever",
    "tokenize",
    "tokenize_no_stop",
    "split_sentences",
    "normalize_whitespace",
    "chunk_by_words",
    "chunk_by_sentences",
    "chunk_recursive",
    "sliding_window",
    "BM25",
    "hashing_embed",
    "cosine_similarity",
    "EmbeddingIndex",
    "Retriever",
]
