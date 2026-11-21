"""Tests for sneppx_serve.usage_tracker (sneppx-alg interface_bindings.usage_tracker)."""

import os
import tempfile

from SneppX_ALG.interface_bindings.usage_tracker import UsageTracker


def _tracker():
    d = tempfile.mkdtemp()
    return UsageTracker(db_path=os.path.join(d, "usage.db"))


def test_record_and_get_stats():
    t = _tracker()
    t.record("key1", tokens_in=10, tokens_out=5)
    stats = t.get_stats("key1")
    assert stats is not None


def test_check_rate_limit():
    t = _tracker()
    ok, remaining = t.check_rate_limit("key1", rpm=100)
    assert isinstance(ok, bool)
    assert isinstance(remaining, int)


def test_get_stats_empty():
    t = _tracker()
    stats = t.get_stats("nonexistent")
    # should return something (empty or None)
    assert stats is None or isinstance(stats, dict)
