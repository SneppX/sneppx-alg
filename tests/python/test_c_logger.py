"""Tests for c_logger.py — structured logger."""

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import Logger


def test_logger_defaults():
    log = Logger("test")
    assert log.name == "test"
    assert log.level == "INFO"


def test_logger_levels():
    log = Logger("test", level="DEBUG")
    assert log._should_log("DEBUG")
    assert log._should_log("INFO")
    assert log._should_log("ERROR")
    log.set_level("ERROR")
    assert not log._should_log("INFO")
    assert log._should_log("ERROR")


def test_logger_format():
    log = Logger("fmt", level="DEBUG")
    msg = log._format("INFO", "hello world")
    assert "hello world" in msg
    assert "INFO" in msg
    assert "fmt" in msg


def test_logger_json():
    log = Logger("json", level="INFO", json_output=True)
    msg = log._format("WARN", "test", {"extra": 42})
    import json
    parsed = json.loads(msg)
    assert parsed["level"] == "WARN"
    assert parsed["message"] == "test"
    assert parsed["extra"] == 42


def test_logger_rank():
    log = Logger("ranked", level="INFO", rank=3)
    msg = log._format("INFO", "hello")
    assert "rank" in msg
    assert log.rank == 3


def test_logger_named():
    parent = Logger("parent")
    child = parent.named("child")
    assert child.name == "parent.child"
    assert child.level == parent.level


def test_logger_output():
    log = Logger("out", level="INFO")
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        log.info("test message")
        output = captured.getvalue()
        assert "test message" in output
    finally:
        sys.stdout = old_stdout


def test_logger_error_stderr():
    log = Logger("err")
    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured
    try:
        log.error("error msg")
        output = captured.getvalue()
        assert "error msg" in output
    finally:
        sys.stderr = old_stderr


if __name__ == "__main__":
    test_logger_defaults()
    test_logger_levels()
    test_logger_format()
    test_logger_json()
    test_logger_rank()
    test_logger_named()
    test_logger_output()
    test_logger_error_stderr()
    print("All c_logger tests passed.")
