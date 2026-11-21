# Cookbook — Serving & Inference

## 1. Start the FastAPI inference server

**Intent:** Spin up `sneppx-serve` with security + TLS.

```powershell
sneppx-serve --port 8000 --host 0.0.0.0 --auth-mode api-key --api-keys "k1,k2"
```

Flags mirror `serve_cli.py`: `--model-config`, `--checkpoint`,
`--tokenizer`, `--workers`, `--tls-certfile/--tls-keyfile`,
`--cidr-allow/--cidr-deny`, `--rate-limit`, `--port-knock`,
`--api-db / --usage-db` (SQLite), `--disable-prompt-injection`,
`--disable-output-verify`, `--enable-firewall`.

**Notes:** Requires `pip install "sneppx-alg[serve]"` for `fastapi`/`uvicorn`/
`pydantic`. CPU-safe.

## 2. Register a loaded model

**Intent:** Hot-swap which model the server serves.

```python
from SneppX_ALG import register_model, get_model, list_models
from SneppX_ALG import Transformer

model = Transformer(vocab_size=1000, dim=256, num_heads=4, num_layers=4, ffn_dim=1024, max_seq_len=128)
register_model("gpt-tiny", model)
print(list_models())     # ['gpt-tiny']
m = get_model("gpt-tiny")
```

**Notes:** The server registers a passthrough by default; `register_model`
swaps in your real module. CPU-safe.

## 3. Generate via the HTTP API

**Intent:** Call `/v1/generate` like an OpenAI-compatible endpoint.

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer k1" \
  -d '{"prompt":"Hello, SneppX","max_new_tokens":64,"temperature":0.7,"top_p":0.9}'
```

Responses are `GenerateResponse`-shaped: `generated_text`, `token_ids`,
`prompt_tokens`, `completion_tokens`, `total_tokens`.

## 4. Continuous batching

**Intent:** High-throughput token-by-token scheduling.

```python
from SneppX_ALG import ContinuousBatchScheduler, ScheduledRequest, SchedulerConfig

cfg = SchedulerConfig(max_batch_size=64, max_seq_len=2048, schedule_policy="fcfs")
sched = ContinuousBatchScheduler(cfg)
reqs = [ScheduledRequest(ids=[1,2,3], req_id=0, max_new_tokens=32)]
result = sched.step(reqs)       # one scheduler step
```

**Notes:** `ContinuousBatchScheduler` is backed by `inference_server.py`'s
`/v1/generate/continuous-batch` endpoint. GPU recommended for throughput.
