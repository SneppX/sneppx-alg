# SneppX API Reference

**Version:** 1.1.1  
**Base URL:** `http://localhost:8080`  
**OpenAPI:** Auto-generated at `/docs` (Swagger UI) when FastAPI server is running.

---

## Authentication

All API endpoints (except auth/public ones) require an API key in the `Authorization` header:

```
Authorization: Bearer sk-sneppx-<hex>
```

API keys are generated with the `sk-sneppx-` prefix. Keys are hashed (SHA-256) at rest; only the first 18 characters (`sk-sneppx-` + 8 hex) are displayed in list responses.

### Header Parameters

| Header | Value | Required |
|--------|-------|----------|
| `Authorization` | `Bearer sk-sneppx-<hex>` | Yes (except auth/health) |
| `Content-Type` | `application/json` | For POST/PATCH requests |

---

## Error Responses

All errors return JSON with the same structure:

```json
{
  "detail": "Human-readable error message"
}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (missing/invalid parameters) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not Found |
| 409 | Conflict (e.g. email already registered) |
| 429 | Too Many Requests (rate limit exceeded) |
| 500 | Internal Server Error |

---

## Health

### `GET /v1/health`

Check server status.

**Auth:** None

**Response:**

```json
{
  "status": "ok",
  "version": "1.1.1",
  "models_loaded": 0,
  "uptime_seconds": 123.45
}
```

**Example:**

```bash
curl http://localhost:8080/v1/health
```

---

## Models

### `GET /v1/models`

List all registered models.

**Auth:** API key (Bearer token)

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-7b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "sneppx",
      "meta": {}
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

### `GET /v1/models/{model_id}`

Get details for a specific model.

**Auth:** API key (Bearer token)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | string | Model identifier |

**Example:**

```bash
curl http://localhost:8080/v1/models/llama-7b \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

## Text Generation

### `POST /v1/generate`

Generate text from a prompt.

**Auth:** API key (Bearer token)

**Request Body:**

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 256,
  "do_sample": true,
  "temperature": 1.0,
  "top_k": 0,
  "top_p": 1.0,
  "repetition_penalty": 1.0,
  "num_beams": 1,
  "stop_strings": null,
  "model": null
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | — | Input text prompt (required) |
| `max_new_tokens` | int | 256 | Max tokens to generate (1–4096) |
| `do_sample` | bool | true | Use sampling vs greedy |
| `temperature` | float | 1.0 | Sampling temperature (0.01–5.0) |
| `top_k` | int | 0 | Top-k sampling (0 = disabled) |
| `top_p` | float | 1.0 | Nucleus sampling (0.0–1.0) |
| `repetition_penalty` | float | 1.0 | Repetition penalty (0.0–10.0) |
| `num_beams` | int | 1 | Beam search width (1–8) |
| `stop_strings` | [string] | null | Stop sequences |
| `model` | string | null | Model ID (uses default if null) |

**Response:**

```json
{
  "generated_text": "...",
  "token_ids": [1, 2, 3],
  "prompt_tokens": 5,
  "completion_tokens": 128,
  "total_tokens": 133,
  "model": "llama-7b",
  "created": 1700000000
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/v1/generate \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_new_tokens": 100}'
```

---

### `POST /v1/generate/batch`

Generate text for multiple prompts in a single request.

**Auth:** API key (Bearer token)

**Request Body:**

```json
{
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
  "max_new_tokens": 256,
  "do_sample": true,
  "temperature": 1.0,
  "top_k": 0,
  "top_p": 1.0,
  "repetition_penalty": 1.0,
  "model": null
}
```

**Constraints:** 1–64 prompts per request.

**Example:**

```bash
curl -X POST http://localhost:8080/v1/generate/batch \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello", "World"]}'
```

---

### `POST /v1/generate/stream`

Stream generated tokens via Server-Sent Events (SSE).

**Auth:** API key (Bearer token)

**Request Body:** Same as `POST /v1/generate`.

**Response:** SSE stream:

```
data: {"choices":[{"index":0,"delta":{"content":"42"},"finish_reason":null}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}

data: {"choices":[{"index":0,"delta":{"content":"43"},"finish_reason":null}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

data: [DONE]
```

**Example:**

```bash
curl -N -X POST http://localhost:8080/v1/generate/stream \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "max_new_tokens": 50}'
```

---

### `POST /v1/generate/continuous-batch`

Submit prompts to the continuous batching scheduler. Supports dynamic request queue with priority scheduling.

**Auth:** API key (Bearer token)

**Request Body:** Same as `POST /v1/generate/batch`.

**Example:**

```bash
curl -X POST http://localhost:8080/v1/generate/continuous-batch \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Explain quantum computing"]}'
```

---

## Chat Completions

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Auth:** API key (Bearer token)

**Request Body:**

```json
{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is ML?"}
  ],
  "max_tokens": 256,
  "temperature": 1.0,
  "top_k": 0,
  "top_p": 1.0,
  "stream": false,
  "stop": null
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | "default" | Model ID |
| `messages` | [object] | — | Chat messages (required, min 1) |
| `max_tokens` | int | 256 | Max tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_k` | int | 0 | Top-k filtering |
| `top_p` | float | 1.0 | Nucleus sampling |
| `stream` | bool | false | Enable SSE streaming |
| `stop` | [string] | null | Stop sequences |

Each message has `role` (system/user/assistant) and `content` (string).

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "ML stands for Machine Learning..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

---

## Quantization

### `POST /v1/models/quantize`

Quantize a loaded model to reduce memory footprint.

**Auth:** API key (Bearer token)

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | — | Model ID to quantize (required) |
| `quant_mode` | string | "int4" | Quantization mode: `int4`, `int8`, `fp8` |

**Response:**

```json
{
  "status": "ok",
  "model_id": "llama-7b",
  "quant_mode": "int4",
  "layers_quantized": 32,
  "size_mb": 3500.5
}
```

**Example:**

```bash
curl -X POST "http://localhost:8080/v1/models/quantize?model_id=llama-7b&quant_mode=int4" \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

## User Authentication (Public Gateway)

### `POST /v1/auth/register`

Register a new user account. Automatically creates a free-tier API key.

**Auth:** None

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Constraints:**

| Field | Rule |
|-------|------|
| `email` | Valid email format |
| `password` | 8–128 characters |

**Response:**

```json
{
  "user_id": "uuid-string",
  "email": "user@example.com",
  "tier": "free",
  "api_key": "sk-sneppx-a1b2c3d4e5f6g7h8",
  "message": "Account created. Save your API key — it will not be shown again."
}
```

**⚠️ Important:** The API key is shown **only once** at registration. Store it securely.

**Example:**

```bash
curl -X POST http://localhost:8080/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "password": "my-strong-pw-123"}'
```

---

### `POST /v1/auth/login`

Authenticate with email and password.

**Auth:** None

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**

```json
{
  "user_id": "uuid-string",
  "email": "user@example.com",
  "tier": "free",
  "token": "sk-sneppx-a1b2c3d4e5f6g7h8"
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com", "password": "my-strong-pw-123"}'
```

---

## Self-Service Key Management

### `GET /v1/keys`

List all API keys belonging to the authenticated user.

**Auth:** API key (Bearer token)

**Response:**

```json
{
  "keys": [
    {
      "id": "key-uuid",
      "key_prefix": "sk-sneppx-a1b2c3d4",
      "name": "Default key for alice@example.com",
      "tier": "free",
      "is_active": true,
      "created_at": 1700000000.0,
      "expires_at": null
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8080/v1/keys \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

### `POST /v1/keys`

Create a new API key for the authenticated user.

**Auth:** API key (Bearer token)

**Request Body:**

```json
{
  "name": "My new key",
  "expires_in_days": 90
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | "" | Key nickname (max 64 chars) |
| `expires_in_days` | int | null | Optional expiration (1–365) |

**Rate Limits by Tier:**

| Tier | Max Keys | RPM |
|------|----------|-----|
| Free | 2 | 60 |
| Pro | 10 | 1,000 |
| Enterprise | 100 | 10,000 |

**Response:**

```json
{
  "key": "sk-sneppx-<full-key>",
  "id": "key-uuid",
  "name": "My new key",
  "tier": "free",
  "is_active": true,
  "created_at": 1700000000.0,
  "expires_at": 1707593600.0
}
```

**Example:**

```bash
curl -X POST http://localhost:8080/v1/keys \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8" \
  -H "Content-Type: application/json" \
  -d '{"name": "dev-key", "expires_in_days": 30}'
```

---

### `DELETE /v1/keys/{key_id}`

Revoke (delete) one of the authenticated user's API keys.

**Auth:** API key (Bearer token)

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `key_id` | string | Key UUID to revoke |

**Response:**

```json
{
  "status": "revoked",
  "key_id": "key-uuid"
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8080/v1/keys/key-uuid \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

### `GET /v1/usage`

Get aggregated usage statistics for the authenticated user's keys.

**Auth:** API key (Bearer token)

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `since` | float | null | Unix timestamp to filter from |

**Response:**

```json
{
  "requests": 142,
  "tokens_in": 5230,
  "tokens_out": 12800,
  "avg_response_ms": 245.3
}
```

**Example:**

```bash
curl "http://localhost:8080/v1/usage?since=1700000000" \
  -H "Authorization: Bearer sk-sneppx-a1b2c3d4e5f6g7h8"
```

---

## Admin API

All admin endpoints require the admin key set at server startup (`--admin-key <key>`). Pass it in the `Authorization` header.

### `GET /v1/admin/keys`

List all API keys in the system.

**Auth:** Admin key (Bearer token)

**Response:**

```json
{
  "keys": [
    {
      "id": "key-uuid",
      "key_prefix": "sk-sneppx-a1b2c3d4",
      "name": "Default key for ...",
      "tier": "free",
      "is_active": true,
      "user_id": "user-uuid",
      "created_at": 1700000000.0,
      "expires_at": null
    }
  ],
  "count": 5
}
```

**Example:**

```bash
curl http://localhost:8080/v1/admin/keys \
  -H "Authorization: Bearer my-admin-key"
```

---

### `POST /v1/admin/keys`

Create an API key for any user or tier.

**Auth:** Admin key (Bearer token)

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "" | Key nickname |
| `tier` | string | "free" | Tier: free, pro, enterprise |
| `expires_in_days` | int | null | Optional expiration |
| `user_id` | string | null | Owner user ID |

**Example:**

```bash
curl -X POST "http://localhost:8080/v1/admin/keys?name=prod-key&tier=enterprise&user_id=user-uuid" \
  -H "Authorization: Bearer my-admin-key"
```

---

### `GET /v1/admin/keys/{key_id}`

Get details for a specific key.

**Auth:** Admin key (Bearer token)

**Example:**

```bash
curl http://localhost:8080/v1/admin/keys/key-uuid \
  -H "Authorization: Bearer my-admin-key"
```

---

### `DELETE /v1/admin/keys/{key_id}`

Revoke any key.

**Auth:** Admin key (Bearer token)

**Example:**

```bash
curl -X DELETE http://localhost:8080/v1/admin/keys/key-uuid \
  -H "Authorization: Bearer my-admin-key"
```

---

### `PATCH /v1/admin/keys/{key_id}`

Update key properties.

**Auth:** Admin key (Bearer token)

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | New nickname |
| `tier` | string | New tier |
| `rate_limit_rpm` | int | Custom rate limit |
| `expires_in_days` | int | Extend/set expiration |

**Example:**

```bash
curl -X PATCH "http://localhost:8080/v1/admin/keys/key-uuid?tier=pro&rate_limit_rpm=2000" \
  -H "Authorization: Bearer my-admin-key"
```

---

### `GET /v1/admin/keys/{key_id}/usage`

Get usage statistics for a specific key.

**Auth:** Admin key (Bearer token)

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `since` | float | Unix timestamp filter |

**Response:**

```json
{
  "requests": 50,
  "tokens_in": 1200,
  "tokens_out": 3400,
  "avg_response_ms": 183.2,
  "by_endpoint": {
    "/v1/generate": {"count": 30, "avg_ms": 200.0},
    "/v1/chat/completions": {"count": 20, "avg_ms": 150.0}
  }
}
```

**Example:**

```bash
curl "http://localhost:8080/v1/admin/keys/key-uuid/usage" \
  -H "Authorization: Bearer my-admin-key"
```

---

## Tier Comparison

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Max API keys | 2 | 10 | 100 |
| Rate limit (RPM) | 60 | 1,000 | 10,000 |
| Max tokens/request | 2,048 | 4,096 | 8,192 |
| Concurrent requests | 1 | 5 | Unlimited |
| Access | Public | Public | Private deployment |
| Support | Community | Email | Dedicated |

---

## C HTTP Server API

For C/C++ applications that embed SneppX directly, the `net/http/` module provides a lightweight HTTP server.

### Lifecycle

```c
#include "net/http/http_server.h"
#include "net/http/http_auth.h"

/* Create server on port 8080 with 4 worker threads */
SNEPPX_HttpServer* srv = SNEPPX_http_server_create(8080, 4);

/* Add auth middleware */
SNEPPX_HttpAuth* auth = SNEPPX_http_auth_create(NULL);
SNEPPX_http_auth_add_key(auth, getenv("SNEPPX_HTTP_API_KEY"));
SNEPPX_http_auth_add_public_path(auth, "/v1/health");
SNEPPX_http_server_add_middleware(srv,
    SNEPPX_http_auth_middleware(auth), auth);

/* Register routes */
SNEPPX_http_server_add_route(srv, "GET", "/v1/health", health_handler, NULL);
SNEPPX_http_server_add_route(srv, "POST", "/v1/generate", generate_handler, my_model);

/* Start (blocking) */
SNEPPX_http_server_start(srv);

/* Cleanup */
SNEPPX_http_server_stop(srv);
SNEPPX_http_auth_destroy(auth);
SNEPPX_http_server_destroy(srv);
```

### Handler Signature

```c
int my_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata);
```

Return 0 on success. Use the accessor/setter functions from `http_server.h`.

### Auth Middleware

```c
/* Create auth state */
SNEPPX_HttpAuth* auth = SNEPPX_http_auth_create(NULL);
SNEPPX_http_auth_add_key(auth, getenv("SNEPPX_HTTP_API_KEY"));

/* Mark public paths (no auth required) */
SNEPPX_http_auth_add_public_path(auth, "/v1/health");
SNEPPX_http_auth_add_public_path(auth, "/v1/auth");

/* Get middleware function pointer */
SNEPPX_http_middleware_fn mw = SNEPPX_http_auth_middleware(auth);

/* Register on server */
SNEPPX_http_server_add_middleware(srv, mw, auth);
```

In stub mode (no SQLite), any key with the `sk-sneppx-` prefix is accepted for development.

### High-Level REST API (`http_api.h`)

The `net/http/http_api.h` module registers the standard serving endpoints on a server
in one call. It mirrors the Python `inference_server.py` surface for the core routes.

```c
#include "net/http/http_api.h"

SNEPPX_HttpApi* api = SNEPPX_http_api_create("1.1.1");
SNEPPX_http_api_register(srv, api);   /* registers the /v1/* routes */
```

Registered endpoints:

| Method | Path                 | Auth    | Description                          |
|--------|----------------------|---------|--------------------------------------|
| GET    | `/v1/health`         | public  | Status, version, uptime, model count |
| GET    | `/v1/models`         | public  | List known model presets             |
| GET    | `/v1/models/{id}`    | public  | Single model configuration           |
| POST   | `/v1/generate`       | Bearer  | Generate text from a prompt          |

The generate handler parses a JSON body (`model`, `prompt`, optional
`max_new_tokens`, `temperature`, `top_p`, `top_k`) and returns the same
`GenerateResponse` shape as the Python server. Without loaded model weights it
produces a deterministic continuation; responses include `generated_text`,
`token_ids`, and token counts.

A runnable demo is provided at `examples/http_server_demo.c`
(builds when `SNEPPX_BUILD_HTTP=ON`):

```powershell
# build
cmake -B build_http -G Ninja -DSNEPPX_BUILD_HTTP=ON
cmake --build build_http --config Release --target http_server_demo

# run (port, optional key DB)
build_http\examples\http_server_demo.exe 8080

# exercise
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models
curl -H "Authorization: Bearer sk-sneppx-... \
     -H "Content-Type: application/json" \
     -d '{"model":"llama3-8B","prompt":"Hello"}' \
     http://127.0.0.1:8080/v1/generate
```

Routes may also use `{param}` patterns; captured values are read with
`SNEPPX_http_request_param(req, "name")`.

---

## Quick Reference

```
# Public (no auth)
GET  /v1/health                                    Health check
POST /v1/auth/register                             Register account
POST /v1/auth/login                                Login

# Model & Generation (auth required)
GET  /v1/models                                    List models
GET  /v1/models/{id}                               Get model info
POST /v1/generate                                  Generate text
POST /v1/generate/batch                            Batch generate
POST /v1/generate/stream                           Stream generate
POST /v1/generate/continuous-batch                 Continuous batch
POST /v1/chat/completions                          Chat completions
POST /v1/models/quantize                           Quantize model

# Self-service keys (auth required)
GET  /v1/keys                                      List my keys
POST /v1/keys                                      Create key
DELETE /v1/keys/{id}                               Revoke key
GET  /v1/usage                                     My usage stats

# Admin (admin key required)
GET  /v1/admin/keys                                List all keys
POST /v1/admin/keys                                Create any key
GET  /v1/admin/keys/{id}                           Get key details
DELETE /v1/admin/keys/{id}                         Revoke any key
PATCH /v1/admin/keys/{id}                          Update key
GET  /v1/admin/keys/{id}/usage                     Key usage stats
```
