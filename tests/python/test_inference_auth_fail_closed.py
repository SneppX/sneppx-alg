"""Regression: inference API must fail closed when security is uninitialized."""

import ast
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import inference_server as srv


class _Request:
    headers = {}
    client = None


def test_require_auth_rejects_when_security_uninitialized(monkeypatch):
    monkeypatch.setattr(srv, "get_security", lambda: None)
    with pytest.raises(HTTPException) as exc:
        srv.require_auth(_Request())
    assert exc.value.status_code == 503


def test_app_registers_lifespan():
    tree = ast.parse(Path(srv.__file__).read_text(encoding="utf-8"))
    app_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FastAPI"
    ]
    assert app_calls
    assert any(keyword.arg == "lifespan" for keyword in app_calls[0].keywords), (
        "FastAPI app must register the lifespan that initializes security"
    )
