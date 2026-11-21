"""Tests for c_attention.py — attention kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import Attention, DifferentialAttention, FlexAttention


def test_attention_forward():
    attn = Attention()
    q = np.random.randn(2, 4, 8, 16).astype(np.float32)
    k = np.random.randn(2, 4, 8, 16).astype(np.float32)
    v = np.random.randn(2, 4, 8, 16).astype(np.float32)
    out = attn.forward(q, k, v)
    assert out.shape == (2, 4, 8, 16)


def test_attention_causal_mask():
    attn = Attention()
    q = np.random.randn(1, 1, 4, 8).astype(np.float32)
    k = np.random.randn(1, 1, 4, 8).astype(np.float32)
    v = np.random.randn(1, 1, 4, 8).astype(np.float32)
    mask = np.triu(-1e9 * np.ones((4, 4)), k=1).astype(np.float32)
    out = attn.forward(q, k, v, mask=mask)
    assert out.shape == (1, 1, 4, 8)
    assert not np.any(np.isnan(out))


def test_attention_scale():
    attn = Attention()
    q = np.random.randn(1, 1, 2, 4).astype(np.float32)
    k = np.random.randn(1, 1, 2, 4).astype(np.float32)
    v = np.random.randn(1, 1, 2, 4).astype(np.float32)
    out = attn.forward(q, k, v, scale=1.0)
    assert out.shape == (1, 1, 2, 4)


def test_differential_attention():
    da = DifferentialAttention()
    q = np.random.randn(2, 1, 4, 16).astype(np.float32)
    k = np.random.randn(2, 1, 4, 16).astype(np.float32)
    v = np.random.randn(2, 1, 4, 8).astype(np.float32)
    out = da.forward(q, k, v, lambd=0.9)
    assert out.shape == (2, 1, 4, 8)


def test_differential_attention_identity():
    da = DifferentialAttention()
    q = np.random.randn(1, 1, 2, 8).astype(np.float32)
    k = np.random.randn(1, 1, 2, 8).astype(np.float32)
    v = np.random.randn(1, 1, 2, 4).astype(np.float32)
    out_lambda0 = da.forward(q, k, v, lambd=0.0)
    out_lambda1 = da.forward(q, k, v, lambd=0.5)
    assert out_lambda0.shape == out_lambda1.shape


def test_flex_attention():
    fa = FlexAttention()
    q = np.random.randn(1, 2, 3, 8).astype(np.float32)
    k = np.random.randn(1, 2, 3, 8).astype(np.float32)
    v = np.random.randn(1, 2, 3, 8).astype(np.float32)
    out = fa.forward(q, k, v)
    assert out.shape == (1, 2, 3, 8)


def test_flex_attention_block_mask():
    fa = FlexAttention()
    q = np.random.randn(1, 1, 4, 4).astype(np.float32)
    k = np.random.randn(1, 1, 4, 4).astype(np.float32)
    v = np.random.randn(1, 1, 4, 4).astype(np.float32)
    mask = np.ones((4, 4), dtype=np.float32)
    mask[2:, :] = 0
    out = fa.forward(q, k, v, block_mask=mask)
    assert out.shape == (1, 1, 4, 4)


if __name__ == "__main__":
    test_attention_forward()
    test_attention_causal_mask()
    test_attention_scale()
    test_differential_attention()
    test_differential_attention_identity()
    test_flex_attention()
    test_flex_attention_block_mask()
    print("All c_attention tests passed.")
