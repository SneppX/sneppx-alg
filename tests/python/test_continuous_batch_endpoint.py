"""M-13 regression tests for the continuous-batch endpoint (mocked, CPU-only)."""

import asyncio
import types

from SneppX_ALG.interface_bindings import inference_server as server
from SneppX_ALG.interface_bindings.continuous_batching import BatchResult


def test_continuous_batch_endpoint_uses_request_scoped_scheduler_and_drains_results(
    monkeypatch,
):
    entry = types.SimpleNamespace(model_id="test-model", model=object())
    monkeypatch.setattr(server, "get_model", lambda model_id: entry)
    monkeypatch.setattr(server, "_filter_prompt", lambda prompt: ("clean", prompt))
    monkeypatch.setattr(server, "_encode_prompt", lambda entry, prompt: [1])

    def fail_get_scheduler():
        raise AssertionError("endpoint must not use the global scheduler")

    monkeypatch.setattr(server, "get_scheduler", fail_get_scheduler)

    def fake_loop(model_fn, scheduler, decode_fn, max_steps):
        scheduler.get_batch()
        ids = list(scheduler._running)
        for rid in ids:
            yield BatchResult(
                request_id=rid,
                generated_text="ok",
                token_ids=[1, 2],
                prompt_tokens=1,
                completion_tokens=1,
                total_time_ms=1.0,
            )

    monkeypatch.setattr(server, "continuous_generate_loop", fake_loop)

    req = server.BatchGenerateRequest(prompts=["hello", "world"], max_new_tokens=1, do_sample=False)
    response = asyncio.run(server.continuous_batch_endpoint(req))

    assert len(response.results) == 2
    assert [item.generated_text for item in response.results] == ["ok", "ok"]
    assert all(item.completion_tokens == 1 for item in response.results)
