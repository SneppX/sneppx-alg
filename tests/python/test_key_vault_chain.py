"""Regression: key-vault audit chains must recompute and verify every digest."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import key_vault as kv


def _logger(tmp_path):
    path = tmp_path / "audit.log"
    logger = kv.AuditLogger(str(path))
    for i in range(3):
        logger.log("encrypt", "alice", f"key-{i}", "success", {"n": i})
    return logger, path


def test_valid_chain_verifies(tmp_path):
    logger, _ = _logger(tmp_path)
    assert logger.verify_chain() is True


def test_in_memory_tampering_detected(tmp_path):
    logger, _ = _logger(tmp_path)
    logger._entries[1]["details"] = {"n": 999}
    assert logger.verify_chain() is False


def test_persisted_log_tampering_detected(tmp_path):
    logger, path = _logger(tmp_path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[1]["details"] = {"n": 999}
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    assert logger.verify_chain() is False


def test_stored_digest_must_match_recomputed_value(tmp_path):
    logger, _ = _logger(tmp_path)
    entry = logger._entries[0]
    entry["hash"] = "0" * 64
    assert logger.verify_chain() is False


def test_chain_ends_at_expected_head(tmp_path):
    logger, path = _logger(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    # In-memory and persisted views disagree.
    assert logger.verify_chain() is False
