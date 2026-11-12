# SNEPPX Model Serving Guide

## Overview

SNEPPX Serving provides a high-performance inference engine with continuous batching, paged KV cache, quantized execution, and a REST API compatible with OpenAI endpoint specifications. It includes both a Python inference server and a native C HTTP control plane.

## Python Inference Server

```python
from sneppx.serving import InferenceServer

server = InferenceServer(
    model=model,
    quantize="int4",
    max_batch_size=64,
    max_seq_len=8192,
)

server.start(port=8080)
```

### Endpoints
- `GET /v1/health`: Server health check.
- `GET /v1/models`: List loaded models.
- `POST /v1/generate`: Text generation with streaming and token counts.
- `POST /v1/generate/continuous-batch`: Continuous batching generation queue.

## Python Server Security

For remote access, bind the server with authentication enabled:

```bash
sneppx-serve --host 0.0.0.0 --auth-mode bearer \
  --api-keys "replace-with-a-long-random-secret" \
  --rate-limit-rpm 120
```

The Python server enforces authentication on generation, continuous-batch
generation, model quantization, deployment/traffic controls, and the metrics
endpoint. Admin key-management endpoints read credentials from the
`Authorization` HTTP header; do not put credentials in query strings.

Authenticated requests are rate-limited per user or API key. Managed API keys
use their stored per-key RPM limit; static keys use `--rate-limit-rpm`.
When authentication is disabled, protected local endpoints are rate-limited
by client IP. Set `--rate-limit-rpm 0` to disable rate limiting. The CLI
rejects unauthenticated non-loopback binds.

## C HTTP Control Plane

Built in C (`net/http/http_api.c`) with zero external dependencies (Winsock / BSD sockets):
- Bearer token authentication middleware (`http_auth.c`).
- Static file serving and route parameter capture (`{param}`).
- Demo binary: `examples/http_server_demo.c`.
