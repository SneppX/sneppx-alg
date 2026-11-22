"""Regression: multi-rank collectives must not pretend to reduce data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import distributed
from SneppX_ALG.interface_bindings.tensor import Tensor


@pytest.fixture
def multi_rank(monkeypatch):
    ctx = distributed._global_context
    monkeypatch.setattr(ctx, "world_size", 2)
    monkeypatch.setattr(ctx, "rank", 0)
    return ctx


def test_all_reduce_identity_for_single_rank(monkeypatch):
    monkeypatch.setattr(distributed._global_context, "world_size", 1)
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    assert distributed.all_reduce(t) is t


def test_all_reduce_fails_closed_for_multi_rank(multi_rank):
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="multi-rank collective"):
        distributed.all_reduce(t)


def test_all_gather_fails_closed_for_multi_rank(multi_rank):
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="multi-rank collective"):
        distributed.all_gather(t)


def test_reduce_scatter_fails_closed_for_multi_rank(multi_rank):
    t = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(RuntimeError, match="multi-rank collective"):
        distributed.reduce_scatter(t)
