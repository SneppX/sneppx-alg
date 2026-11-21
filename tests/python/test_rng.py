"""Tests for sneppx_serve.rng (sneppx-alg interface_bindings.rng)."""

import os
import random

from SneppX_ALG.interface_bindings.rng import set_global_seed, make_rng


def test_set_global_seed_int_validation():
    try:
        set_global_seed("not-an-int")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")


def test_set_global_seed_deterministic():
    set_global_seed(42)
    a = random.random()
    set_global_seed(42)
    b = random.random()
    assert a == b


def test_set_global_seed_env():
    set_global_seed(7)
    assert os.environ.get("SNEAPX_SEED") == "7"


def test_make_rng_returns_independent_instance():
    r1 = make_rng(1)
    r2 = make_rng(1)
    assert isinstance(r1, random.Random)
    assert r1 is not r2
    assert r1.random() == r2.random()


def test_make_rng_different_seeds():
    r1 = make_rng(1)
    r2 = make_rng(2)
    assert r1.random() != r2.random()
