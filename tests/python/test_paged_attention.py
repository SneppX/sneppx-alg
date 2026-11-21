"""Tests for paged attention / chunked prefill / prefix caching."""

import numpy as np

from SneppX_ALG.interface_bindings.paged_attention import (
    PagedAttentionConfig,
    KVBlockPool,
    BlockTable,
    PagedKVCache,
    PrefixCache,
    prefix_hash,
    paged_attention_forward,
    IncrementalAttentionState,
    incremental_attention_forward,
    chunked_prefill_forward,
)

RNG = np.random.default_rng(7)


def _kv(seq, heads, dim):
    return RNG.normal(size=(seq, heads, dim)).astype(np.float32)


def test_block_pool_alloc_free():
    pool = KVBlockPool(num_blocks=8, block_size=16, num_kv_heads=4, head_dim=32)
    assert pool.num_free() == 8
    b1 = pool.allocate()
    b2 = pool.allocate()
    assert b1 != b2
    assert pool.num_free() == 6
    pool.unref(b1)
    assert pool.num_free() == 7
    print("  test_block_pool_alloc_free PASS")


def test_block_table_basics():
    t = BlockTable([3, 7])
    assert len(t) == 2
    assert t[1] == 7
    t.append(9)
    assert t.to_list() == [3, 7, 9]
    print("  test_block_table_basics PASS")


def test_paged_kv_store_get():
    config = PagedAttentionConfig(num_layers=1, num_kv_heads=4, head_dim=16,
                                  block_size=8, num_blocks=16)
    cache = PagedKVCache(config)
    tok = [1, 2, 3, 4, 5]
    k = _kv(len(tok), 4, 16)
    v = _kv(len(tok), 4, 16)
    table = cache.allocate()
    cache.store(table, layer=0, positions=list(range(len(tok))), keys=k, values=v)
    rk, rv = cache.get_kv(table, layer=0, seq_len=len(tok))
    assert rk.shape == (4, 5, 16)
    assert np.allclose(rk, k.transpose(1, 0, 2), atol=1e-5)
    assert np.allclose(rv, v.transpose(1, 0, 2), atol=1e-5)
    stats = cache.stats()
    assert stats["used_blocks"] > 0
    assert stats["sequences"] == 1
    cache.free(table)
    assert cache.stats()["sequences"] == 0
    print("  test_paged_kv_store_get PASS")


def test_paged_kv_fork_cow():
    config = PagedAttentionConfig(num_layers=1, num_kv_heads=2, head_dim=8,
                                  block_size=4, num_blocks=32)
    cache = PagedKVCache(config)
    tok = list(range(6))
    k = _kv(6, 2, 8)
    v = _kv(6, 2, 8)
    parent = cache.allocate()
    cache.store(parent, layer=0, positions=list(range(6)), keys=k, values=v)
    child = cache.fork(parent)
    assert child is not parent
    assert child.to_list() == parent.to_list()
    # write to child at a new position must not corrupt parent (copy-on-write)
    k_new = _kv(1, 2, 8)
    cache.store(child, layer=0, positions=[7], keys=k_new, values=_kv(1, 2, 8))
    pk, _ = cache.get_kv(parent, layer=0, seq_len=6)
    assert np.allclose(pk, k.transpose(1, 0, 2), atol=1e-5)
    # child keeps the copied parent prefix but owns the new write at pos 7
    ck, _ = cache.get_kv(child, layer=0, seq_len=8)
    assert np.allclose(ck[:, :6], k.transpose(1, 0, 2), atol=1e-5)
    assert np.allclose(ck[:, 7], k_new.transpose(1, 0, 2)[:, 0], atol=1e-5)
    print("  test_paged_kv_fork_cow PASS")


def test_prefix_cache():
    pc = PrefixCache(block_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    pc.insert(tokens, [100, 200])
    n, ids = pc.query([1, 2, 3, 4, 5, 6, 7, 8])
    assert n == 2 and ids == [100, 200]
    n, ids = pc.query([1, 2, 3, 4, 5])
    assert n == 1 and ids == [100]
    n, _ = pc.query([9, 9, 9, 9])
    assert n == 0
    assert prefix_hash((1, 2)) == "1:2"
    print("  test_prefix_cache PASS")


def test_paged_attention_forward_math():
    heads, kv_heads, dim, seq = 3, 1, 8, 6
    config = PagedAttentionConfig(num_layers=1, num_kv_heads=kv_heads,
                                  head_dim=dim, block_size=4, num_blocks=16)
    cache = PagedKVCache(config)
    k = _kv(seq, kv_heads, dim)
    v = _kv(seq, kv_heads, dim)
    table = cache.allocate()
    cache.store(table, layer=0, positions=list(range(seq)), keys=k, values=v)
    q = RNG.normal(size=(heads, dim)).astype(np.float32)
    out = paged_attention_forward(q, cache, table, layer=0, seq_len=seq,
                                  num_heads=heads, num_kv_heads=kv_heads)
    assert out.shape == (heads, dim)
    rk, rv = cache.get_kv(table, layer=0, seq_len=seq)
    k_exp = np.repeat(rk, heads // kv_heads, axis=0)
    v_exp = np.repeat(rv, heads // kv_heads, axis=0)
    scores = np.einsum("hkd,hd->hk", k_exp, q)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    ref = np.einsum("hk,hkd->hd", p, v_exp)
    assert np.allclose(out, ref, atol=1e-5)
    print("  test_paged_attention_forward_math PASS")


def test_paged_attention_causal():
    heads, dim, seq = 2, 8, 5
    config = PagedAttentionConfig(num_layers=1, num_kv_heads=1, head_dim=dim,
                                  block_size=4, num_blocks=8)
    cache = PagedKVCache(config)
    k = _kv(seq, 1, dim)
    v = _kv(seq, 1, dim)
    table = cache.allocate()
    cache.store(table, layer=0, positions=list(range(seq)), keys=k, values=v)
    q = RNG.normal(size=(seq, heads, dim)).astype(np.float32)
    out = paged_attention_forward(q, cache, table, layer=0, seq_len=seq,
                                  num_heads=heads, causal=True)
    assert out.shape == (seq, heads, dim)
    rk, rv = cache.get_kv(table, layer=0, seq_len=seq)
    k_exp = np.repeat(rk, heads // 1, axis=0)
    v_exp = np.repeat(rv, heads // 1, axis=0)
    scores = np.einsum("thd,hkd->thk", q, k_exp)
    mask = np.triu(np.ones((seq, seq), dtype=bool), 1)
    scores = np.where(mask[:, None, :], -np.inf, scores)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    ref = np.einsum("thk,hkd->thd", p, v_exp)
    assert np.allclose(out, ref, atol=1e-5)
    # causal row 0 attends only to token 0 == decode against a single-key cache.
    q0 = q[0]
    out0 = paged_attention_forward(q0, cache, table, layer=0, seq_len=1,
                                   num_heads=heads, causal=True)
    assert np.allclose(out[0], out0, atol=1e-5)
    print("  test_paged_attention_causal PASS")


def test_incremental_attention():
    heads, dim, seq = 2, 8, 5
    q = RNG.normal(size=(heads, dim)).astype(np.float32)
    k = RNG.normal(size=(heads, seq, dim)).astype(np.float32)
    v = RNG.normal(size=(heads, seq, dim)).astype(np.float32)
    state = IncrementalAttentionState(heads, dim)
    for i in range(seq):
        out, state = incremental_attention_forward(
            q, k[:, i:i + 1], v[:, i:i + 1], state)
        scores = np.einsum("hd,hkd->hk", q, k[:, : i + 1])
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        ref = np.einsum("hk,hkd->hd", p, v[:, : i + 1])
        assert np.allclose(out, ref, atol=1e-5)
    print("  test_incremental_attention PASS")


def test_chunked_prefill():
    heads, dim, seq, chunk = 4, 16, 12, 4
    q = RNG.normal(size=(heads, dim)).astype(np.float32)
    k = RNG.normal(size=(heads, seq, dim)).astype(np.float32)
    v = RNG.normal(size=(heads, seq, dim)).astype(np.float32)
    chunks = range(0, seq, chunk)
    keys = [k[:, i:i + chunk] for i in chunks]
    values = [v[:, i:i + chunk] for i in chunks]
    out = chunked_prefill_forward(q, keys, values)
    scores = np.einsum("hd,hkd->hk", q, k)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    ref = np.einsum("hk,hkd->hd", p, v)
    assert np.allclose(out, ref, atol=1e-5)
    print("  test_chunked_prefill PASS")


def test_config_defaults():
    cfg = PagedAttentionConfig()
    assert cfg.block_size == 16
    assert cfg.num_blocks == 64
    print("  test_config_defaults PASS")


if __name__ == "__main__":
    test_block_pool_alloc_free()
    test_block_table_basics()
    test_paged_kv_store_get()
    test_paged_kv_fork_cow()
    test_prefix_cache()
    test_paged_attention_forward_math()
    test_paged_attention_causal()
    test_incremental_attention()
    test_chunked_prefill()
    test_config_defaults()
    print("ALL paged_attention TESTS PASS")
