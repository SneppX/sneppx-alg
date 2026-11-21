"""PagedAttention — paged KV-cache allocator, block tables, prefix caching.

Implements vLLM-style paged KV cache management on the CPU/numpy backend:

- :class:`KVBlockPool` owns a fixed pool of fixed-size KV blocks.
- :class:`BlockTable` tracks the block list backing one sequence.
- :class:`PagedKVCache` allocates/frees/fork-copies block tables, stores
  keys/values at arbitrary positions, gathers dense K/V via block tables,
  and supports copy-on-write for sequence forking.
- :class:`PrefixCache` hashes token blocks for prefix reuse.
- :func:`paged_attention_forward` runs attention against gathered paged K/V.
- :func:`chunked_prefill_forward` splits long prompts into chunks with an
  incremental (online-softmax) attention state.

Typical usage::

    cache = PagedKVCache(PagedAttentionConfig(num_blocks=64))
    table = cache.allocate()
    cache.store(table, layer=0, positions=[0, 1, 2], keys=K, values=V)
    out = paged_attention_forward(q, cache, table, layer=0, seq_len=3)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PagedAttentionConfig",
    "KVBlockPool",
    "BlockTable",
    "PagedKVCache",
    "PrefixCache",
    "prefix_hash",
    "paged_attention_forward",
    "incremental_attention_forward",
    "IncrementalAttentionState",
    "chunked_prefill_forward",
]


@dataclass
class PagedAttentionConfig:
    """Configuration for a paged KV cache."""

    num_layers: int = 1
    num_kv_heads: int = 1
    head_dim: int = 64
    block_size: int = 16
    num_blocks: int = 64
    dtype: str = "float32"


def prefix_hash(tokens: Tuple[int, ...]) -> str:
    """Hash a block of tokens to a stable string key."""
    return ":".join(str(int(t)) for t in tokens)


class KVBlockPool:
    """Fixed-size pool of KV blocks.

    Each block stores ``block_size`` slots of ``[num_kv_heads, head_dim]``.
    Blocks are recycled through a free list and reference-counted so forked
    sequences can share blocks (copy-on-write).
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int, dtype: str = "float32"):
        self.block_size = int(block_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.np_dtype = np.dtype(dtype)
        self.keys = np.zeros(
            (int(num_blocks), block_size, num_kv_heads, head_dim),
            dtype=self.np_dtype,
        )
        self.values = np.zeros_like(self.keys)
        self._free = list(range(int(num_blocks)))
        self._refcounts = [0] * int(num_blocks)

    def num_free(self) -> int:
        return len(self._free)

    def num_used(self) -> int:
        return len(self._refcounts) - len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("KV block pool exhausted")
        block_id = self._free.pop()
        self._refcounts[block_id] = 1
        return block_id

    def ref(self, block_id: int) -> None:
        self._refcounts[block_id] += 1

    def unref(self, block_id: int) -> None:
        self._refcounts[block_id] -= 1
        if self._refcounts[block_id] <= 0:
            self._refcounts[block_id] = 0
            self.keys[block_id].fill(0)
            self.values[block_id].fill(0)
            self._free.append(block_id)

    def is_shared(self, block_id: int) -> bool:
        return self._refcounts[block_id] > 1

    def copy_block(self, src_id: int) -> int:
        new_id = self.allocate()
        self.keys[new_id] = self.keys[src_id]
        self.values[new_id] = self.values[src_id]
        return new_id

    def utilization(self) -> float:
        total = len(self._refcounts)
        if total == 0:
            return 0.0
        return self.num_used() / total


class BlockTable:
    """Block list backing a sequence (vLLM block_table)."""

    def __init__(self, block_ids: Optional[List[int]] = None):
        self.block_ids = list(block_ids or [])

    def __len__(self) -> int:
        return len(self.block_ids)

    def __getitem__(self, index: int) -> int:
        return self.block_ids[index]

    def append(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    def to_list(self) -> List[int]:
        return list(self.block_ids)


class PagedKVCache:
    """Manager over a :class:`KVBlockPool` with per-sequence block tables.

    Provides allocate / free / fork / store / gather operations at the token
    granularity, resolving token positions to ``(block_id, slot)`` pairs.
    """

    def __init__(self, config: Optional[PagedAttentionConfig] = None):
        self.config = config or PagedAttentionConfig()
        self.pool = KVBlockPool(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            dtype=self.config.dtype,
        )
        self._tables: Dict[int, BlockTable] = {}
        self._next_id = 0

    # ---- block-table lifecycle -------------------------------------------
    def allocate(self, num_blocks: int = 1) -> BlockTable:
        table = BlockTable([self.pool.allocate() for _ in range(num_blocks)])
        self._tables[self._next_id] = table
        self._next_id += 1
        return table

    def free(self, table: BlockTable) -> None:
        for block_id in list(table.block_ids):
            self.pool.unref(block_id)
        table.block_ids.clear()
        for tid, t in list(self._tables.items()):
            if t is table:
                del self._tables[tid]

    def fork(self, table: BlockTable) -> BlockTable:
        child = BlockTable(list(table.block_ids))
        for block_id in child.block_ids:
            self.pool.ref(block_id)
        self._tables[self._next_id] = child
        self._next_id += 1
        return child

    def _ensure_capacity(self, table: BlockTable, seq_len: int) -> None:
        needed = (int(seq_len) + self.pool.block_size - 1) // self.pool.block_size
        while len(table) < needed:
            table.append(self.pool.allocate())

    def _write_slot(self, table: BlockTable, block_index: int,
                    slot: int, key: np.ndarray, value: np.ndarray) -> None:
        block_id = table[block_index]
        if self.pool.is_shared(block_id):
            old_id = block_id
            new_id = self.pool.copy_block(old_id)
            self.pool.unref(old_id)
            table.block_ids[block_index] = new_id
            block_id = new_id
        self.pool.keys[block_id, slot] = key
        self.pool.values[block_id, slot] = value

    # ---- store / gather --------------------------------------------------
    def store(self, table: BlockTable, layer: int,
              positions: List[int], keys: np.ndarray, values: np.ndarray) -> None:
        """Store K/V at absolute token positions for a layer.

        ``keys``/``values`` may have shape ``[num_positions, num_kv_heads,
        head_dim]`` or ``[num_kv_heads, head_dim]`` for a single position.
        """
        keys = np.asarray(keys, dtype=self.pool.np_dtype)
        values = np.asarray(values, dtype=self.pool.np_dtype)
        if keys.ndim == 2:
            keys = keys[None, ...]
            values = values[None, ...]
            positions = positions[:1]
        max_pos = max(positions) if positions else 0
        self._ensure_capacity(table, max_pos + 1)
        for i, pos in enumerate(positions):
            block_index = pos // self.pool.block_size
            slot = pos % self.pool.block_size
            self._write_slot(table, block_index, slot, keys[i], values[i])

    def get_kv(self, table: BlockTable, layer: int,
               seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gather dense K/V ``[num_kv_heads, seq_len, head_dim]`` for a layer."""
        self._ensure_capacity(table, seq_len)
        nh = self.pool.num_kv_heads
        hd = self.pool.head_dim
        k = np.zeros((nh, int(seq_len), hd), dtype=self.pool.np_dtype)
        v = np.zeros_like(k)
        for pos in range(int(seq_len)):
            block_index = pos // self.pool.block_size
            slot = pos % self.pool.block_size
            block_id = table[block_index]
            k[:, pos, :] = self.pool.keys[block_id, slot]
            v[:, pos, :] = self.pool.values[block_id, slot]
        return k, v

    # ---- stats -----------------------------------------------------------
    def stats(self) -> Dict[str, object]:
        return {
            "num_blocks": len(self.pool._refcounts),
            "free_blocks": self.pool.num_free(),
            "used_blocks": self.pool.num_used(),
            "utilization": self.pool.utilization(),
            "sequences": len(self._tables),
        }


class PrefixCache:
    """Hash-based prefix block reuse (vLLM automatic prefix caching).

    ``insert`` records the block ids backing a token prefix so a later
    ``query`` can reuse them. Inserted blocks are reference-counted, so a
    reused block is shared between the cache and the new sequence.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = int(block_size)
        self._entries: Dict[str, List[int]] = {}

    @staticmethod
    def _chunks(tokens: List[int], block_size: int) -> List[Tuple[int, ...]]:
        stop = len(tokens) - (len(tokens) % block_size)
        return [
            tuple(tokens[i:i + block_size])
            for i in range(0, stop, block_size)
        ]

    def insert(self, tokens: List[int], block_ids: List[int]) -> None:
        chunks = self._chunks(list(tokens), self.block_size)
        for i, chunk in enumerate(chunks):
            key = prefix_hash(chunk)
            self._entries.setdefault(key, []).append(block_ids[i])

    def query(self, tokens: List[int]) -> Tuple[int, List[int]]:
        """Return (num_matching_blocks, block_ids) for the longest prefix hit."""
        chunks = self._chunks(list(tokens), self.block_size)
        matched: List[int] = []
        for chunk in chunks:
            key = prefix_hash(chunk)
            candidates = self._entries.get(key)
            if not candidates:
                break
            matched.append(candidates[0])
        return len(matched), matched


def paged_attention_forward(
    q: np.ndarray,
    cache: PagedKVCache,
    table: BlockTable,
    layer: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    causal: bool = True,
) -> np.ndarray:
    """Run multi-head attention against a paged KV cache.

    Args:
        q: query ``[num_heads, head_dim]``, ``[1, num_heads, head_dim]``
            (decode, one token) or ``[seq_len, num_heads, head_dim]``
            (prefill, one query per cached position).
        cache: the paged KV cache holding the key/value blocks.
        table: the requesting sequence's block table.
        layer: cache layer to read.
        seq_len: number of cached tokens to attend to.
        num_heads: number of query heads (GQA expands kv heads).
        num_kv_heads: kv heads in cache; defaults to cache config.
        causal: apply a causal mask so position ``t`` attends to ``[0, t]``.
            A single-query (decode) call has no future keys, so the mask is a
            no-op there.

    Returns:
        Output ``[num_heads, head_dim]`` for decode or
        ``[seq_len, num_heads, head_dim]`` for prefill.
    """
    q = np.asarray(q, dtype=np.float32)
    kv_heads = num_kv_heads or cache.pool.num_kv_heads
    k, v = cache.get_kv(table, layer, seq_len)
    expand = num_heads // kv_heads

    k_exp = np.repeat(k, expand, axis=0)
    v_exp = np.repeat(v, expand, axis=0)

    if q.ndim == 3 and q.shape[0] == seq_len and q.shape[1] == num_heads:
        # prefill: one query per cached position -> full causal masking.
        scores = np.einsum("thd,hkd->thk", q, k_exp)
        if causal and seq_len > 1:
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores = np.where(mask[:, None, :], -np.inf, scores)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return np.einsum("thk,hkd->thd", probs, v_exp)

    # decode: single query token; no future keys exist.
    if q.ndim == 2:
        qq = q
    elif q.ndim == 3 and q.shape[1] == 1:
        qq = q[:, 0, :]
    else:
        raise ValueError("q must be [num_heads, head_dim], "
                         "[1, num_heads, head_dim] or "
                         "[seq_len, num_heads, head_dim]")
    scores = np.einsum("hkd,hd->hk", k_exp, qq)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.einsum("hk,hkd->hd", probs, v_exp)
    return out


class IncrementalAttentionState:
    """Online-softmax attention accumulator (for chunked prefill)."""

    def __init__(self, num_heads: int, head_dim: int, dtype="float32"):
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.acc = np.zeros((num_heads, head_dim), dtype=dtype)
        self.max = np.full((num_heads, 1), -np.inf, dtype=dtype)
        self.lse = np.zeros((num_heads, 1), dtype=dtype)

    def update(self, scores: np.ndarray, values: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        new_max = np.maximum(self.max, scores.max(axis=-1, keepdims=True))
        correction = np.exp(self.max - new_max)
        exp = np.exp(scores - new_max)
        self.acc = self.acc * correction + np.einsum(
            "hk,hkd->hd", exp, values
        )
        self.lse = self.lse * correction + exp.sum(axis=-1, keepdims=True)
        self.max = new_max

    def logsumexp(self) -> np.ndarray:
        return self.max + np.log(self.lse)

    def output(self) -> np.ndarray:
        return self.acc / np.where(self.lse > 0, self.lse, 1.0)


def incremental_attention_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    state: Optional[IncrementalAttentionState] = None,
) -> Tuple[np.ndarray, IncrementalAttentionState]:
    """One chunk of online-softmax attention against dense K/V.

    Args:
        q: query ``[num_heads, head_dim]``.
        k, v: chunk keys/values ``[num_heads, chunk_len, head_dim]``.
        state: running attention state (created if ``None``).

    Returns:
        ``(output, state)`` where output is the running output for the query.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if q.ndim == 3 and q.shape[1] == 1:
        q = q[:, 0, :]
    if state is None:
        state = IncrementalAttentionState(q.shape[0], q.shape[1])
    scores = (q[:, None, :] @ k.transpose(0, 2, 1))[:, 0, :]
    state.update(scores, v)
    return state.output(), state


def chunked_prefill_forward(
    q: np.ndarray,
    keys: List[np.ndarray],
    values: List[np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Prefill-style attention where K/V arrive in chunks.

    Args:
        q: query ``[num_heads, head_dim]``.
        keys/values: iterables of chunked K/V
            ``[num_heads, chunk_len, head_dim]``.
        chunk_size: ignored (chunk boundaries come from the inputs).

    Returns:
        Final online-softmax attention output ``[num_heads, head_dim]``.
    """
    state: Optional[IncrementalAttentionState] = None
    for k, v in zip(keys, values):
        _, state = incremental_attention_forward(q, k, v, state)
    return state.output()
