"""FastAPI inference server for SneppX models — generate, chat, stream."""

import os
import time
import json
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, Header, Request, Depends
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    BaseModel = object

from .tensor import Tensor
from .nn import Module
from .generation import (
    GenerationConfig,
    generate,
    batch_generate,
    Streamer,
    TextStreamer,
    TokenStreamer,
)
from .tokenizer import Tokenizer
from .security_middleware import SecurityConfig, SecurityMiddleware
from .api_key_manager import ApiKeyManager, get_key_manager
from .usage_tracker import UsageTracker, get_usage_tracker, set_usage_tracker
from .api_gateway import router as auth_router, self_service_router as keys_router
from .firewall import create_firewall as _create_firewall
from .continuous_batching import ContinuousBatchScheduler, SchedulerConfig, ScheduledRequest, continuous_generate_loop
from .quantized_serve import QuantizedModelConfig, quantize_model_weights, dequantize_weights
from .serving_engine import ServingEngine, ServingConfig, MetricsCollector

logger = logging.getLogger(__name__)


# ===========================================================================
#  Serving control plane (metrics / readiness / versioning / A-B)
# ===========================================================================

_serving_engine: Optional[ServingEngine] = None


def get_serving_engine() -> ServingEngine:
    """Shared production-grade serving engine for metrics, readiness, A/B routing."""
    global _serving_engine
    if _serving_engine is None:
        _serving_engine = ServingEngine(config=ServingConfig(),
                                        model_fn=lambda batch: batch)
        _serving_engine.set_ready(True)
    return _serving_engine


# ===========================================================================
#  Pydantic Models
# ===========================================================================


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    do_sample: bool = True
    temperature: float = Field(1.0, ge=0.01, le=5.0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.0, le=10.0)
    num_beams: int = Field(1, ge=1, le=8)
    stop_strings: Optional[List[str]] = None
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    generated_text: str
    token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    created: int


class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=64)
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    model: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: List[GenerateResponse]
    model: str
    created: int


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.01, le=5.0)
    top_k: int = 0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "sneppx"
    meta: Dict[str, Any] = {}


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.5.2"
    models_loaded: int = 0
    uptime_seconds: float = 0.0


# ===========================================================================
#  Model Registry
# ===========================================================================


class ModelEntry:
    def __init__(
        self,
        model_id: str,
        model: Module,
        tokenizer: Optional[Tokenizer] = None,
        default_config: Optional[GenerationConfig] = None,
        meta: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.default_config = default_config or GenerationConfig()
        self.meta = meta or {}
        self.created = int(time.time())


_models: Dict[str, ModelEntry] = {}
_start_time: float = time.time()
_scheduler: Optional[ContinuousBatchScheduler] = None


def get_scheduler() -> ContinuousBatchScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ContinuousBatchScheduler()
    return _scheduler


def _filter_prompt(prompt: str) -> Tuple[str, str]:
    sec = get_security()
    if sec is not None:
        return sec.filter_prompt(prompt)
    return "clean", prompt


def _verify_output(text: str) -> Tuple[str, str]:
    sec = get_security()
    if sec is not None:
        return sec.verify_output(text)
    return "clean", text


def create_firewall(**kwargs):
    return _create_firewall(**kwargs)


def register_model(
    model_id: str,
    model: Module,
    tokenizer: Optional[Tokenizer] = None,
    default_config: Optional[GenerationConfig] = None,
    meta: Optional[dict] = None,
):
    _models[model_id] = ModelEntry(model_id, model, tokenizer, default_config, meta)
    # Mirror into the production serving engine (model versioning surface).
    eng = get_serving_engine()
    if not eng.versions(model_id):
        eng.register_model(model_id, "v1", meta.get("description", "") if isinstance(meta, dict) else "")


def get_model(model_id: Optional[str] = None) -> ModelEntry:
    if model_id is None:
        if not _models:
            raise HTTPException(503, "No models loaded")
        return list(_models.values())[0]
    if model_id not in _models:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return _models[model_id]


def list_models() -> List[ModelEntry]:
    return list(_models.values())


def _encode_prompt(entry: "ModelEntry", prompt: str) -> List[int]:
    """Encode prompt to token IDs. Requires a tokenizer for real text generation."""
    if entry.tokenizer:
        return entry.tokenizer.encode(prompt)
    raise HTTPException(
        422,
        "No tokenizer registered for this model. Register a tokenizer to encode text prompts, "
        "or provide pre-tokenized input via a custom integration.",
    )


def _decode_tokens(entry: "ModelEntry", token_ids: List[int]) -> str:
    """Decode token IDs to text. Requires a tokenizer."""
    if entry.tokenizer:
        return entry.tokenizer.decode(token_ids)
    raise HTTPException(
        422,
        "No tokenizer registered for this model. Register a tokenizer to decode generated tokens.",
    )


def _make_generation_config(req: GenerateRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        num_beams=req.num_beams,
        stop_strings=req.stop_strings,
    )


def _make_chat_config(req: ChatCompletionRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature or 1.0,
        top_k=req.top_k,
        top_p=req.top_p,
        stop_strings=req.stop,
    )


# ===========================================================================
#  FastAPI App
# ===========================================================================


_security: Optional[SecurityMiddleware] = None
_admin_key: Optional[str] = None


def get_security() -> Optional[SecurityMiddleware]:
    return _security


def set_security(config: Optional[SecurityConfig] = None, firewall_config: Optional[dict] = None,
                admin_key: Optional[str] = None, key_store_path: Optional[str] = None,
                usage_db_path: Optional[str] = None):
    global _security, _admin_key
    if key_store_path:
        from .api_key_store import ApiKeyStore
        store = ApiKeyStore(db_path=key_store_path)
        km = ApiKeyManager(store=store)
        if admin_key:
            km.set_admin_key(admin_key)
        set_key_manager(km)
    if usage_db_path:
        set_usage_tracker(UsageTracker(db_path=usage_db_path))
    if config is None:
        _security = SecurityMiddleware(SecurityConfig.from_env(), key_manager=get_key_manager())
    else:
        _security = SecurityMiddleware(config, key_manager=get_key_manager())
    _admin_key = admin_key
    if firewall_config is not None:
        _security.firewall = _create_firewall(**firewall_config)
    elif os.environ.get("FIREWALL_CONFIG"):
        _security.firewall = _create_firewall(os.environ["FIREWALL_CONFIG"])


def require_auth(request: "Request"):
    """Enforce authentication on protected endpoints.

    When security is disabled (auth mode ``"none"``) requests are allowed
    through. When auth is enabled, a missing/invalid credential raises 401.
    """
    sec = get_security()
    if sec is None:
        return None
    auth = sec.authenticator
    if not auth.enabled:
        return None
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    info = auth.authenticate(header)
    if info is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    user_id = info.get("user_id") or info.get("id") or "authenticated"
    allowed, reason = sec.check_rate_limit(
        user_id,
        tier=info.get("tier", "free"),
        rpm_override=info.get("rate_limit_rpm"),
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=reason,
            headers={"Retry-After": "60"},
        )
    return info


def _verify_admin_auth(authorization: Optional[str]) -> bool:
    """Validate an admin credential supplied in the Authorization header."""
    if not authorization or not _admin_key:
        return False
    value = authorization.strip()
    scheme, separator, credential = value.partition(" ")
    if separator and scheme.casefold() == "bearer":
        value = credential.strip()
    return bool(value) and value == _admin_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SneppX inference server")
    _models.clear()
    global _security
    if _security is None:
        _security = SecurityMiddleware(SecurityConfig.from_env())
    app.state.security = _security
    yield
    _models.clear()
    app.state.security = None
    logger.info("Shutting down SneppX inference server")


import hashlib


def _key_hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


app = FastAPI(
    title="SneppX Inference API",
    version="1.5.2",
)

app.include_router(auth_router)
app.include_router(keys_router)


@app.middleware("http")
async def usage_middleware(request: callable, call_next):
    import time as time_module
    start = time_module.time()
    response = await call_next(request)
    elapsed_ms = (time_module.time() - start) * 1000
    try:
        auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth_header:
            key = auth_header.strip()
            scheme, separator, credential = key.partition(" ")
            if separator and scheme.casefold() == "bearer":
                key = credential.strip()
            if key.startswith("sk-sneppx-"):
                tracker = get_usage_tracker()
                kh = _key_hash(key)
                tracker.record(
                    key_hash=kh,
                    endpoint=request.url.path,
                    response_ms=elapsed_ms,
                    status_code=response.status_code,
                )
    except Exception:
        pass
    # Feed the production serving engine's metrics (latency/throughput/errors).
    try:
        get_serving_engine().metrics.record(
            latency_us=elapsed_ms * 1000.0, tokens=0, error=response.status_code >= 500)
    except Exception:
        pass
    return response


@app.get("/v1/health", response_model=HealthResponse)
async def health_endpoint():
    return HealthResponse(
        status="ok",
        version="1.5.2",
        models_loaded=len(_models),
        uptime_seconds=time.time() - _start_time,
    )


# ---------------------------------------------------------------------------
# Production serving surface: liveness, readiness, metrics, versioning, A/B
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz_endpoint():
    eng = get_serving_engine()
    return {"status": "ok", "ready": eng.ready,
            "uptime_ms": int((time.time() - _start_time) * 1000)}


@app.get("/readyz")
async def readyz_endpoint():
    eng = get_serving_engine()
    if not eng.ready:
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True, "uptime_ms": int((time.time() - _start_time) * 1000)}


@app.get("/metrics", dependencies=[Depends(require_auth)])
async def metrics_endpoint(format: str = Query(default=""), accept: Optional[str] = Header(default=None)):
    from fastapi.responses import PlainTextResponse
    eng = get_serving_engine()
    if format == "json" or (accept and "application/json" in accept):
        return JSONResponse(eng.metrics.to_dict())
    return PlainTextResponse(eng.metrics.render_prometheus(),
                             media_type="text/plain; version=0.0.4")


@app.get("/v1/models/{model_id}/versions")
async def model_versions_endpoint(model_id: str):
    eng = get_serving_engine()
    versions = eng.versions(model_id)
    if versions is None:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return {"name": model_id, "versions": versions}


@app.post("/v1/deploy", dependencies=[Depends(require_auth)])
async def deploy_endpoint(model: str = Query(...), version: str = Query("v1"),
                          description: str = Query(""), promote: bool = Query(False)):
    eng = get_serving_engine()
    ok = eng.deploy(model, version, description, weight=100 if promote else 0, promote=promote)
    return {"status": "deployed" if ok else "error"}


@app.post("/v1/traffic", dependencies=[Depends(require_auth)])
async def traffic_endpoint(model: str = Query(...), split: str = Query(...)):
    eng = get_serving_engine()
    legs = []
    for part in split.split(","):
        if ":" in part:
            vid, w = part.split(":", 1)
            legs.append((vid.strip(), int(w.strip())))
    ok = eng.set_traffic(model, legs)
    return {"status": "ok" if ok else "error"}


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models_endpoint():
    data = []
    for entry in _models.values():
        data.append(
            ModelInfo(
                id=entry.model_id,
                created=entry.created,
                meta=entry.meta,
            )
        )
    return ModelListResponse(data=data)


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    entry = get_model(model_id)
    return ModelInfo(
        id=entry.model_id,
        created=entry.created,
        meta=entry.meta,
    )


@app.post("/v1/generate", response_model=GenerateResponse, dependencies=[Depends(require_auth)])
async def generate_endpoint(req: GenerateRequest):
    filter_status, sanitized = _filter_prompt(req.prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else req.prompt

    entry = get_model(req.model)
    gen_config = _make_generation_config(req)

    input_ids = _encode_prompt(entry, prompt_to_use)

    result = generate(
        entry.model,
        input_ids,
        generation_config=gen_config,
    )
    output_ids = result["output_ids"].flatten().tolist()
    prompt_len = len(input_ids)
    completion_len = len(output_ids) - prompt_len

    generated_text = _decode_tokens(entry, output_ids[prompt_len:])

    verify_status, verified_text = _verify_output(generated_text)
    if verify_status == "blocked":
        generated_text = verified_text

    return GenerateResponse(
        generated_text=generated_text,
        token_ids=output_ids,
        prompt_tokens=prompt_len,
        completion_tokens=completion_len,
        total_tokens=len(output_ids),
        model=entry.model_id,
        created=int(time.time()),
    )


@app.post("/v1/generate/batch", response_model=BatchGenerateResponse, dependencies=[Depends(require_auth)])
async def batch_generate_endpoint(req: BatchGenerateRequest):
    sanitized_prompts = []
    for prompt in req.prompts:
        filter_status, sanitized = _filter_prompt(prompt)
        if filter_status == "injection":
            raise HTTPException(400, "Prompt blocked by content filter")
        sanitized_prompts.append(sanitized if filter_status == "sanitized" else prompt)

    entry = get_model(req.model)
    gen_config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )

    all_prompts = []
    for prompt in sanitized_prompts:
        ids = _encode_prompt(entry, prompt)
        all_prompts.append(np.array(ids, dtype=np.int64))

    result = batch_generate(entry.model, all_prompts, generation_config=gen_config)
    outputs = result["output_ids"]
    results_list = []
    for i, prompt in enumerate(req.prompts):
        output_ids = outputs[i].tolist() if outputs.ndim > 1 else outputs.tolist()
        prompt_len = len(all_prompts[i])
        completion_len = len(output_ids) - prompt_len
        generated_text = _decode_tokens(entry, output_ids[prompt_len:])
        results_list.append(
            GenerateResponse(
                generated_text=generated_text,
                token_ids=output_ids,
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=len(output_ids),
                model=entry.model_id,
                created=int(time.time()),
            )
        )
    return BatchGenerateResponse(
        results=results_list,
        model=entry.model_id,
        created=int(time.time()),
    )


@app.post("/v1/generate/continuous-batch", response_model=BatchGenerateResponse, dependencies=[Depends(require_auth)])
async def continuous_batch_endpoint(req: BatchGenerateRequest):
    sanitized_prompts = []
    for prompt in req.prompts:
        filter_status, sanitized = _filter_prompt(prompt)
        if filter_status == "injection":
            raise HTTPException(400, "Prompt blocked by content filter")
        sanitized_prompts.append(sanitized if filter_status == "sanitized" else prompt)

    entry = get_model(req.model)
    scheduler = get_scheduler()
    req_ids = []
    for prompt in sanitized_prompts:
        ids = _encode_prompt(entry, prompt)
        req_id = scheduler.submit(
            prompt=prompt, prompt_tokens=ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        )
        req_ids.append(req_id)

    def model_fn(input_ids_list, positions):
        gen_config = GenerationConfig(
            max_new_tokens=1, do_sample=req.do_sample,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        )
        batch_input = [np.array(ids, dtype=np.int64) for ids in input_ids_list]
        result = batch_generate(entry.model, batch_input, generation_config=gen_config)
        output_ids = result["output_ids"]
        return output_ids, None

    def decode_fn(token_ids):
        return _decode_tokens(entry, token_ids)

    for batch_result in continuous_generate_loop(model_fn, scheduler, decode_fn, max_steps=req.max_new_tokens * len(req.prompts)):
        pass

    results_list = []
    for i, rid in enumerate(req_ids):
        prompt_len = len(sanitized_prompts[i])
        completed = [r for r in scheduler._completed if r.request_id == rid]
        if completed:
            row = completed[-1]
            results_list.append(GenerateResponse(
                generated_text=row.generated_text,
                token_ids=row.token_ids,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_tokens=row.prompt_tokens + row.completion_tokens,
                model=entry.model_id,
                created=int(time.time()),
            ))
    return BatchGenerateResponse(
        results=results_list,
        model=entry.model_id,
        created=int(time.time()),
    )


@app.post("/v1/models/quantize", dependencies=[Depends(require_auth)])
async def quantize_model_endpoint(model_id: str = Query(...), quant_mode: str = Query("int4")):
    entry = get_model(model_id)
    mode_map = {"int4": "INT4_SYM", "int8": "INT8_SYM", "fp8": "INT4_SYM"}
    from .quantization import QuantMode
    mode = getattr(QuantMode, mode_map.get(quant_mode, "INT4_SYM"), QuantMode.INT4_SYM)
    cfg = QuantizedModelConfig(quant_mode=mode)
    params = {}
    for name, param in entry.model.named_parameters().items() if hasattr(entry.model, "named_parameters") else []:
        from .tensor import Tensor
        p = param.data if hasattr(param, "data") else param
        params[name] = np.asarray(p, dtype=np.float32)
    if not params:
        return JSONResponse({"status": "skipped", "message": "No parameters to quantize"})
    quantized = quantize_model_weights(params, cfg)
    from .quantized_serve import estimate_model_size_mb
    size_mb = estimate_model_size_mb(quantized)
    entry.meta["quantized"] = True
    entry.meta["quant_mode"] = quant_mode
    entry.meta["quantized_size_mb"] = round(size_mb, 2)
    return JSONResponse({
        "status": "ok",
        "model_id": model_id,
        "quant_mode": quant_mode,
        "layers_quantized": len(quantized),
        "size_mb": round(size_mb, 2),
    })


@app.post("/v1/generate/stream", dependencies=[Depends(require_auth)])
async def generate_stream_endpoint(req: GenerateRequest):
    filter_status, sanitized = _filter_prompt(req.prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else req.prompt

    entry = get_model(req.model)
    gen_config = _make_generation_config(req)

    input_ids = _encode_prompt(entry, prompt_to_use)

    async def event_stream() -> AsyncGenerator[str, None]:
        token_buffer = []

        class AsyncStreamer(Streamer):
            def put(self, token_id: int):
                token_buffer.append(token_id)

            def put_text(self, text: str):
                pass

            def end(self):
                pass

        streamer = AsyncStreamer()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate(
                entry.model,
                input_ids,
                generation_config=gen_config,
                streamer=streamer,
            ),
        )

        prompt_len = len(input_ids)
        for i, tid in enumerate(token_buffer):
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": str(tid)},
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": i + 1,
                    "total_tokens": prompt_len + i + 1,
                },
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
async def chat_completions(req: ChatCompletionRequest):
    entry = get_model(req.model)
    gen_config = _make_chat_config(req)

    if entry.tokenizer and hasattr(entry.tokenizer, "apply_chat_template"):
        prompt = entry.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages]
        )
    else:
        prompt = (
            "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
        )

    filter_status, sanitized = _filter_prompt(prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else prompt

    if entry.tokenizer:
        input_ids = entry.tokenizer.encode(prompt_to_use)
    else:
        input_ids = _encode_prompt(entry, prompt_to_use)

    if req.stream:
        return await _chat_stream(entry, prompt_to_use, input_ids, gen_config)

    result = generate(entry.model, input_ids, generation_config=gen_config)
    output_ids = result["output_ids"].flatten().tolist()
    prompt_len = len(input_ids)
    completion_len = len(output_ids) - prompt_len

    content = _decode_tokens(entry, output_ids[prompt_len:])

    verify_status, verified_content = _verify_output(content)
    if verify_status == "blocked":
        content = verified_content

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return ChatCompletionResponse(
        id=resp_id,
        created=int(time.time()),
        model=entry.model_id,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": len(output_ids),
        },
    )


async def _chat_stream(entry, prompt: str, input_ids, gen_config):
    async def event_stream() -> AsyncGenerator[str, None]:
        token_buffer = []

        class AsyncStreamer(Streamer):
            def put(self, token_id: int):
                token_buffer.append(token_id)

            def put_text(self, text: str):
                pass

            def end(self):
                pass

        streamer = AsyncStreamer()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate(
                entry.model,
                input_ids,
                generation_config=gen_config,
                streamer=streamer,
            ),
        )

        prompt_len = len(input_ids)
        for i, tid in enumerate(token_buffer):
            token_text = _decode_tokens(entry, [tid])
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/v1/admin/keys")
async def admin_list_keys(authorization: Optional[str] = Header(default=None, alias="Authorization")):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    keys = km.list_keys()
    return {"keys": keys, "count": len(keys)}


@app.post("/v1/admin/keys")
async def admin_create_key(
    name: str = Query(default=""),
    tier: str = Query(default="free"),
    expires_in_days: Optional[int] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    result = km.create_key(
        name=name, tier=tier,
        expires_in_days=expires_in_days, user_id=user_id,
    )
    if result is None:
        raise HTTPException(400, "Failed to create key")
    return result


@app.get("/v1/admin/keys/{key_id}")
async def admin_get_key(key_id: str, authorization: Optional[str] = Header(default=None, alias="Authorization")):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    key = km.get_key(key_id)
    if not key:
        raise HTTPException(404, "Key not found")
    return key


@app.delete("/v1/admin/keys/{key_id}")
async def admin_revoke_key(key_id: str, authorization: Optional[str] = Header(default=None, alias="Authorization")):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    if not km.revoke_key(key_id):
        raise HTTPException(404, "Key not found")
    return {"status": "revoked", "key_id": key_id}


@app.patch("/v1/admin/keys/{key_id}")
async def admin_update_key(
    key_id: str,
    name: Optional[str] = Query(default=None),
    tier: Optional[str] = Query(default=None),
    rate_limit_rpm: Optional[int] = Query(default=None),
    expires_in_days: Optional[int] = Query(default=None),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    if not km.update_key(key_id, name=name, tier=tier,
                         rate_limit_rpm=rate_limit_rpm, expires_in_days=expires_in_days):
        raise HTTPException(404, "Key not found")
    return {"status": "updated", "key_id": key_id}


@app.get("/v1/admin/keys/{key_id}/usage")
async def admin_key_usage(
    key_id: str,
    since: Optional[float] = Query(default=None),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    if not _verify_admin_auth(authorization):
        raise HTTPException(401, "Admin authentication required")
    km = get_key_manager()
    stats = km.get_usage_stats(key_id, since=since)
    if not stats:
        raise HTTPException(404, "Key not found")
    return stats


__all__ = [
    "app",
    "register_model",
    "get_model",
    "list_models",
    "set_security",
    "get_security",
    "GenerateRequest",
    "GenerateResponse",
    "BatchGenerateRequest",
    "BatchGenerateResponse",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelListResponse",
    "create_firewall",
]
