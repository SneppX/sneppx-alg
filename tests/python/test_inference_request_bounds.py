"""M-14 regressions: bounded inference requests and enforced filter statuses."""

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from SneppX_ALG.interface_bindings import inference_server as server
from SneppX_ALG.interface_bindings.inference_server import (
    BatchGenerateRequest,
    ChatCompletionRequest,
    GenerateRequest,
)


class _Filter:
    def __init__(self, status, text):
        self._result = (status, text)

    def filter_prompt(self, prompt):
        return self._result


def test_request_models_reject_oversize_inputs():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x" * 40_000)
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="ok", max_new_tokens=999_999)
    with pytest.raises(ValidationError):
        BatchGenerateRequest(prompts=["ok"], max_new_tokens=999_999)
    with pytest.raises(ValidationError):
        BatchGenerateRequest(prompts=["x" * 40_000])
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "x" * 40_000}])


@pytest.mark.parametrize(
    "status,status_code",
    [("injection", 400), ("too_long", 413)],
)
def test_prepare_prompt_rejects_unsafe_statuses(status, status_code):
    with patch.object(server, "get_security", return_value=_Filter(status, "x")):
        with pytest.raises(HTTPException) as exc:
            server._prepare_prompt("x")
    assert exc.value.status_code == status_code


def test_prepare_prompt_uses_sanitized_text():
    with patch.object(
        server, "get_security", return_value=_Filter("sanitized", "safe")
    ):
        assert server._prepare_prompt("raw") == "safe"
    with patch.object(server, "get_security", return_value=_Filter("clean", "ignored")):
        assert server._prepare_prompt("raw") == "raw"


@pytest.mark.parametrize(
    "endpoint,req_obj",
    [
        (server.generate_endpoint, GenerateRequest(prompt="hello")),
        (server.batch_generate_endpoint, BatchGenerateRequest(prompts=["hello"])),
        (server.generate_stream_endpoint, GenerateRequest(prompt="hello")),
    ],
)
def test_generate_endpoints_honor_too_long_status(endpoint, req_obj):
    with patch.object(
        server, "get_security", return_value=_Filter("too_long", "hello")
    ):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(endpoint(req_obj))
    assert exc.value.status_code == 413
