"""Regression: S5 privacy defenses must use fresh OS-seeded randomness.

The old ``RandomState(0)`` produced identical "noise" on every call. Repeated
queries cancelled it out, defeating both model-inversion and membership-inference
defenses while still reporting that noise had been applied.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import s5_safety


def test_model_inversion_noise_is_not_repeated():
    g = np.zeros(32, dtype=np.float64)
    a = s5_safety._py_model_inversion_apply(g, noise_scale=1.0, clip_norm=0.0)
    b = s5_safety._py_model_inversion_apply(g, noise_scale=1.0, clip_norm=0.0)

    assert not np.array_equal(a, b)
    assert np.all(np.abs(a) <= 0.5)
    assert np.all(np.abs(b) <= 0.5)


def test_model_inversion_clip_still_applied():
    g = np.array([100.0, 0.0])
    out = s5_safety._py_model_inversion_apply(g, noise_scale=0.0, clip_norm=1.0)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_membership_inference_noise_is_not_repeated():
    logits = np.array([2.0, -2.0, 0.0])
    a = s5_safety._py_membership_inference_defense(logits, epsilon=1.0)
    b = s5_safety._py_membership_inference_defense(logits, epsilon=1.0)

    assert not np.array_equal(a, b)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))


def test_adversarial_smoothing_noise_is_not_repeated():
    safety = s5_safety.S5RLHFSafety()
    x = np.zeros(16, dtype=np.float64)
    a = safety.adversarial_smooth(x, epsilon=0.25)
    b = safety.adversarial_smooth(x, epsilon=0.25)

    assert not np.array_equal(a, b)
    assert np.all(np.abs(a) <= 0.25)
    assert np.all(np.abs(b) <= 0.25)
