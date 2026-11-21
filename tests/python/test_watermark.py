"""Tests for watermark.py — model fingerprinting and watermarking."""

import numpy as np
import pytest

from SneppX_ALG.interface_bindings import Tensor, Module, AdamW
from SneppX_ALG.interface_bindings.watermark import (
    ModelHasher,
    WeightWatermark,
    BackdoorWatermark,
)


class _Model(Module):
    def __init__(self, seed=0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.w = Tensor(rng.randn(8, 4).astype(np.float32), requires_grad=True)
        self.b = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        return x @ self.w + self.b


def _loss_fn(out, labels):
    logits = out if isinstance(out, Tensor) else Tensor(out)
    return logits.cross_entropy(labels)


# ===========================================================================
#  ModelHasher
# ===========================================================================


def test_hash_changes_with_weights():
    m = _Model()
    h = ModelHasher()
    fp1 = h.fingerprint(m)
    m.w.data = m.w.data + 0.01
    fp2 = h.fingerprint(m)
    assert fp1 != fp2


def test_hash_deterministic():
    m1 = _Model(seed=3)
    m2 = _Model(seed=3)
    h = ModelHasher()
    assert h.fingerprint(m1) == h.fingerprint(m2)


def test_hmac_verify_roundtrip():
    m = _Model()
    hs = ModelHasher(secret="my-secret")
    sig = hs.fingerprint(m)
    assert hs.verify(m, sig)
    # Tampered model fails verification.
    m.w.data = m.w.data + 0.5
    assert not hs.verify(m, sig)


def test_hmac_requires_secret_to_verify():
    m = _Model()
    hs = ModelHasher(secret="s")
    sig = hs.fingerprint(m)
    with pytest.raises(ValueError):
        ModelHasher().verify(m, sig)


# ===========================================================================
#  WeightWatermark
# ===========================================================================


def test_weight_watermark_embed_detect():
    m = _Model()
    wm = WeightWatermark(key="owner-key", strength=0.1, ratio=0.1)
    wm.embed(m)
    ok, score = wm.detect(m)
    assert ok
    assert score > 0.9


def test_weight_watermark_absent_on_fresh_model():
    m = _Model()
    ok, score = WeightWatermark(key="owner-key", strength=0.1, ratio=0.1).detect(m)
    assert not ok
    assert score < 0.5


def test_weight_watermark_key_dependent():
    m = _Model()
    WeightWatermark(key="key-A", strength=0.1, ratio=0.1).embed(m)
    # Wrong key should not detect the watermark.
    ok, score = WeightWatermark(key="key-B", strength=0.1, ratio=0.1).detect(m)
    assert not ok


def test_weight_watermark_degrades_after_retrain():
    m = _Model()
    wm = WeightWatermark(key="k", strength=0.1, ratio=0.1)
    wm.embed(m)
    ok_before, _ = wm.detect(m)
    assert ok_before
    # Simulate fine-tuning: perturb many weights away from the pattern.
    for _, p in m.named_parameters():
        arr = np.array(p.data)
        arr = arr + np.random.RandomState(0).normal(0, 0.5, arr.shape).astype(arr.dtype)
        p.data = arr
    ok_after, score = wm.detect(m, threshold=0.7)
    assert score < 1.0


def test_weight_watermark_bad_ratio():
    with pytest.raises(ValueError):
        WeightWatermark(key="k", strength=0.1, ratio=0.0)


# ===========================================================================
#  BackdoorWatermark
# ===========================================================================


def test_backdoor_train_increases_match():
    m = _Model()
    bw = BackdoorWatermark(
        key="bk", input_shape=(8,), target_label=1, num_keys=60, threshold=0.9
    )
    before = bw.verify(m)[1]
    opt = AdamW(m.parameters(), lr=0.05)
    bw.train(m, opt, _loss_fn, epochs=60)
    after_ok, after = bw.verify(m)
    assert after > before
    assert after_ok


def test_backdoor_untrained_low_match():
    m = _Model()
    bw = BackdoorWatermark(
        key="bk", input_shape=(8,), target_label=3, num_keys=40
    )
    ok, frac = bw.verify(m)
    assert not ok
    assert frac < 0.5


def test_backdoor_key_reproducible():
    bw1 = BackdoorWatermark(key="k", input_shape=(8,), target_label=0, num_keys=10)
    bw2 = BackdoorWatermark(key="k", input_shape=(8,), target_label=0, num_keys=10)
    k1 = np.asarray(bw1._keys[0].data)
    k2 = np.asarray(bw2._keys[0].data)
    assert np.allclose(k1, k2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
