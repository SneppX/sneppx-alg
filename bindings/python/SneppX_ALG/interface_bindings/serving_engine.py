"""Production-grade model serving engine for SNEPPX.

Covers:
  * Model versioning + rolling updates (deploy/promote/rollback)
  * A/B testing with consistent-request traffic splitting
  * Dynamic batching with configurable max batch size + timeout
  * /metrics (Prometheus) + /healthz + /readyz endpoints
  * Health checks + readiness probes with warm-up on startup
  * Horizontal scaling via a worker pool
  * YAML configuration with hot-reload (mtime polling)

The C core lives in ``net/http/serving_engine.c``; this module is the
numpy/stdlib mirror used by the Python inference server and a lightweight
standalone :class:`ServingServer` (stdlib ``http.server``) for integration
without FastAPI.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "ServingConfig",
    "ModelVersion",
    "TrafficSplit",
    "MetricsCollector",
    "DynamicBatchQueue",
    "ConfigReloader",
    "ServingEngine",
    "WorkerPool",
    "ServingServer",
    "serve_from_config",
]


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ServingConfig:
    max_batch_size: int = 8
    batch_timeout_ms: float = 5.0
    n_workers: int = 1
    warmup_iters: int = 8
    ready_after_warmup: bool = True
    metrics_prefix: str = "sneppx"
    listen: str = "127.0.0.1"
    port: int = 8300


@dataclass
class ModelVersion:
    name: str
    version_id: str
    description: str = ""
    weight: int = 100
    deployed_at: float = field(default_factory=time.time)
    loaded: bool = True


@dataclass
class TrafficSplit:
    """A/B (or A/B/n) traffic policy for a model."""
    model: str
    legs: List[Tuple[str, int]] = field(default_factory=list)  # [(version_id, weight)]

    def pick(self, request_id: str) -> Optional[str]:
        if not self.legs:
            return None
        total = sum(w for _, w in self.legs)
        if total <= 0:
            return self.legs[0][0] if self.legs else None
        h = int(hashlib.sha256(f"{self.model}:{request_id}".encode()).hexdigest(), 16)
        r = (h % total) + 1
        cum = 0
        for vid, w in self.legs:
            cum += w
            if r <= cum:
                return vid
        return self.legs[-1][0]


# ===========================================================================
# Metrics
# ===========================================================================

_LATENCY_BUCKETS_MS = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]


class MetricsCollector:
    """Latency + throughput + error counters with Prometheus export."""

    def __init__(self, prefix: str = "sneppx"):
        self.prefix = prefix
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.tokens = 0
        self.lat_sum_us = 0.0
        self.lat_sum_sq_us = 0.0
        self.lat_n = 0
        self.buckets = {b: 0 for b in _LATENCY_BUCKETS_MS}
        self.start = time.monotonic()

    def record(self, latency_us: float, tokens: int = 0, error: bool = False) -> None:
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            self.tokens += tokens
            self.lat_sum_us += latency_us
            self.lat_sum_sq_us += latency_us * latency_us
            self.lat_n += 1
            ms = latency_us / 1000.0
            for b in _LATENCY_BUCKETS_MS:
                if ms <= b:
                    self.buckets[b] += 1
                    break
            else:
                self.tokens  # inf bucket implicit

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "errors": self.errors,
                "tokens": self.tokens,
                "lat_sum_us": self.lat_sum_us,
                "lat_sum_sq_us": self.lat_sum_sq_us,
                "lat_n": self.lat_n,
                "buckets": dict(self.buckets),
            }

    def percentile(self, p: float) -> float:
        s = self._snapshot()
        n = s["lat_n"]
        if n <= 0:
            return 0.0
        target = max(1, int(n * p))
        cum = 0
        for b in _LATENCY_BUCKETS_MS:
            cum += s["buckets"][b]
            if cum >= target:
                return b
        return _LATENCY_BUCKETS_MS[-1]

    @property
    def p50(self) -> float:
        return self.percentile(0.50)

    @property
    def p95(self) -> float:
        return self.percentile(0.95)

    @property
    def p99(self) -> float:
        return self.percentile(0.99)

    @property
    def throughput_rps(self) -> float:
        elapsed = max(1e-6, time.monotonic() - self.start)
        s = self._snapshot()
        return s["requests"] / elapsed

    def to_dict(self) -> Dict[str, Any]:
        s = self._snapshot()
        elapsed = max(1e-6, time.monotonic() - self.start)
        return {
            "uptime_ms": int((time.monotonic() - self.start) * 1000),
            "requests": s["requests"],
            "errors": s["errors"],
            "tokens": s["tokens"],
            "throughput_rps": s["requests"] / elapsed,
            "latency_ms": {"p50": self.p50, "p95": self.p95, "p99": self.p99},
            "buckets_ms": s["buckets"],
        }

    def render_prometheus(self) -> str:
        s = self._snapshot()
        elapsed = max(1e-6, time.monotonic() - self.start)
        lines = [
            f"# HELP {self.prefix}_serving_requests_total Total requests.",
            f"# TYPE {self.prefix}_serving_requests_total counter",
            f"{self.prefix}_serving_requests_total {s['requests']}",
            f"# HELP {self.prefix}_serving_errors_total Total errors.",
            f"# TYPE {self.prefix}_serving_errors_total counter",
            f"{self.prefix}_serving_errors_total {s['errors']}",
            f"# HELP {self.prefix}_serving_tokens_total Tokens generated.",
            f"# TYPE {self.prefix}_serving_tokens_total counter",
            f"{self.prefix}_serving_tokens_total {s['tokens']}",
            f"# HELP {self.prefix}_latency_ms Latency by bucket.",
            f"# TYPE {self.prefix}_latency_ms histogram",
        ]
        for b in _LATENCY_BUCKETS_MS:
            lines.append(f'{self.prefix}_latency_ms_bucket{{le="{b:.3f}"}} {s["buckets"][b]}')
        lines.append(f'{self.prefix}_latency_ms_bucket{{le="+Inf"}} {s["lat_n"]}')
        lines.append(f"{self.prefix}_latency_ms_sum {s['lat_sum_us'] / 1000.0:.6f}")
        lines.append(f"{self.prefix}_latency_ms_count {s['lat_n']}")
        lines += [
            f"# HELP {self.prefix}_serving_ready 1 if ready.",
            f"# TYPE {self.prefix}_serving_ready gauge",
            f"{self.prefix}_serving_throughput_rps {s['requests'] / elapsed:.6f}",
        ]
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self.requests = self.errors = self.tokens = self.lat_n = 0
            self.lat_sum_us = self.lat_sum_sq_us = 0.0
            for b in self.buckets:
                self.buckets[b] = 0


# ===========================================================================
# Dynamic batching
# ===========================================================================

@dataclass
class BatchItem:
    request_id: str
    key: str
    submitted_at: float


class DynamicBatchQueue:
    """Thread-safe batch collector. Drains when ``max_batch_size`` is reached
    or ``batch_timeout_ms`` elapses since the first queued item."""

    def __init__(self, max_batch_size: int = 8, batch_timeout_ms: float = 5.0):
        self.max_batch_size = max(1, max_batch_size)
        self.batch_timeout_ms = batch_timeout_ms
        self._lock = threading.Lock()
        self._items: List[BatchItem] = []

    def submit(self, request_id: str, key: str = "") -> int:
        with self._lock:
            self._items.append(BatchItem(request_id=request_id, key=key,
                                         submitted_at=time.monotonic()))
            return len(self._items)

    def drain(self, max_items: int = 1024) -> List[BatchItem]:
        """Return a ready batch, or [] if not ready (size/timeout)."""
        with self._lock:
            n = len(self._items)
            if n == 0:
                return []
            elapsed_ms = (time.monotonic() - self._items[0].submitted_at) * 1000.0
            ready = (n >= self.max_batch_size) or (elapsed_ms >= self.batch_timeout_ms)
            if not ready:
                return []
            take = min(n, max_batch_size_for(n) if False else n, max_items)
            take = min(n, max_items)
            ready_items = self._items[:take]
            self._items = self._items[take:]
            return ready_items

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._items)

    def reconfigure(self, max_batch_size: int, batch_timeout_ms: float) -> None:
        self.max_batch_size = max(1, max_batch_size)
        self.batch_timeout_ms = batch_timeout_ms


def max_batch_size_for(_):
    return 0


# ===========================================================================
# Config hot-reload
# ===========================================================================

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _load_yaml_fallback(path)


def _load_yaml_fallback(path: str) -> Dict[str, Any]:
    """Minimal key: value / dotted-key parser (no external deps)."""
    out: Dict[str, Any] = {}
    try:
        with open(path, "r") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                if ":" in line and not line.endswith(":"):
                    k, v = line.split(":", 1)
                    k = k.strip().lstrip("- ")
                    v = v.strip().strip('"').strip("'")
                    if v == "":
                        out[k] = {}
                    elif v.lower() in ("true", "false"):
                        out[k] = v.lower() == "true"
                    else:
                        try:
                            out[k] = int(v)
                        except ValueError:
                            try:
                                out[k] = float(v)
                            except ValueError:
                                out[k] = v
    except FileNotFoundError:
        return {}
    return out


class ConfigReloader:
    """Polls a YAML file for changes and applies a callback on reload."""

    def __init__(self, path: str, on_reload: Callable[[Dict[str, Any]], None]):
        self.path = path
        self.on_reload = on_reload
        self._mtime = 0.0  # force a reload on the first check()
        self._lock = threading.Lock()

    def check(self) -> bool:
        """Returns True if a reload happened."""
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            return False
        with self._lock:
            if mt == self._mtime:
                return False
            self._mtime = mt
        cfg = _load_yaml(self.path)
        self.on_reload(cfg)
        return True


def _serving_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized serving.* params from either flat-dotted or nested YAML."""
    svc = cfg.get("serving") or {}
    if isinstance(svc, dict):
        out = dict(svc)
    else:
        out = {}
    for k, v in list(cfg.items()):
        if isinstance(k, str) and k.startswith("serving."):
            out[k.split(".", 1)[1]] = v
    return out


def _legs_from_str(val: str) -> List[Tuple[str, int]]:
    legs = []
    for part in str(val).split(","):
        if ":" in part:
            vid, w = part.split(":", 1)
            legs.append((vid.strip(), int(w.strip())))
    return legs


def _traffic_splits(cfg: Dict[str, Any]) -> Dict[str, List[Tuple[str, int]]]:
    out = {}
    for k, v in cfg.items():
        if isinstance(k, str) and k.startswith("traffic."):
            model = k.split(".", 1)[1]
            legs = _legs_from_str(v)
            if legs:
                out[model] = legs
    return out


# ===========================================================================
# Serving engine
# ===========================================================================

class ServingEngine:
    """Python serving control plane (mirrors net/http/serving_engine.c)."""

    def __init__(self, config: Optional[ServingConfig] = None,
                 model_fn: Optional[Callable[..., Any]] = None):
        self.config = config or ServingConfig()
        self.model_fn = model_fn
        self._lock = threading.RLock()
        self._models: Dict[str, List[ModelVersion]] = {}
        self._active: Dict[str, str] = {}          # model -> active version_id
        self._traffic: Dict[str, TrafficSplit] = {}
        self.metrics = MetricsCollector(self.config.metrics_prefix)
        self.batcher = DynamicBatchQueue(
            self.config.max_batch_size, self.config.batch_timeout_ms)
        self._ready = False
        self._warmup_remaining = self.config.warmup_iters
        self._warmup_done = False
        self.started_at = time.monotonic()
        self.config_reloader: Optional[ConfigReloader] = None

    @property
    def active(self) -> Dict[str, str]:
        """Snapshot of active version per model."""
        with self._lock:
            return dict(self._active)

    # ---------- versioning / rolling updates ----------
    def register_model(self, name: str, version_id: str = "v1",
                       description: str = "", weight: Optional[int] = None) -> bool:
        with self._lock:
            versions = self._models.setdefault(name, [])
            for v in versions:
                if v.version_id == version_id:
                    return False
            v = ModelVersion(name=name, version_id=version_id, description=description,
                             weight=100 if weight is None and not versions else (weight or 0))
            versions.append(v)
            if name not in self._active:
                self._active[name] = version_id
            if weight is not None:
                self._traffic.setdefault(name, TrafficSplit(model=name,
                                                            legs=[(version_id, weight)]))
            return True

    def deploy(self, name: str, version_id: str, description: str = "",
               weight: int = 0, promote: bool = False) -> bool:
        """Rolling update: add a new version at `weight` (default 0 = canary)."""
        if not self.register_model(name, version_id, description, weight=weight):
            return False
        with self._lock:
            split = self._traffic.setdefault(name, TrafficSplit(model=name, legs=[]))
            exists = any(lid == version_id for lid, _ in split.legs)
            if not exists:
                split.legs.append((version_id, weight))
            if promote:
                for i, (lid, _) in enumerate(split.legs):
                    split.legs[i] = (lid, 100 if lid == version_id else 0)
                self._active[name] = version_id
        return True

    def set_traffic(self, name: str, legs: List[Tuple[str, int]]) -> bool:
        with self._lock:
            if name not in self._models:
                return False
            total = sum(w for _, w in legs)
            if total > 100:
                return False
            self._traffic[name] = TrafficSplit(model=name, legs=list(legs))
            return True

    def promote(self, name: str, version_id: str) -> bool:
        with self._lock:
            if name not in self._models:
                return False
            versions = self._models[name]
            if not any(v.version_id == version_id for v in versions):
                return False
            self._active[name] = version_id
            self._traffic[name] = TrafficSplit(model=name,
                                               legs=[(version_id, 100)])
            return True

    def rollback(self, name: str) -> bool:
        with self._lock:
            versions = self._models.get(name)
            if not versions or len(versions) < 2:
                return False
            # drop the most recently added version (the canary)
            versions.pop()
            # keep the active version if it still exists, else fall back to head
            active = self._active.get(name)
            if not any(v.version_id == active for v in versions):
                active = versions[0].version_id
            self._active[name] = active
            self._traffic[name] = TrafficSplit(model=name, legs=[(active, 100)])
            return True

    def route(self, name: str, request_id: str) -> Optional[str]:
        with self._lock:
            split = self._traffic.get(name)
            if split:
                return split.pick(request_id) or self._active.get(name)
            return self._active.get(name)

    def set_config_path(self, path: str) -> None:
        self.config_reloader = ConfigReloader(path, self._apply_config)

    def _apply_config(self, cfg: Dict[str, Any]) -> None:
        svc = _serving_params(cfg)
        self.config.max_batch_size = int(svc.get("max_batch_size", self.config.max_batch_size))
        self.config.batch_timeout_ms = float(svc.get("batch_timeout_ms", self.config.batch_timeout_ms))
        self.config.n_workers = int(svc.get("n_workers", self.config.n_workers))
        self.config.warmup_iters = int(svc.get("warmup_iters", self.config.warmup_iters))
        self.config.listen = str(svc.get("listen", self.config.listen))
        self.config.port = int(svc.get("port", self.config.port))
        self.batcher.reconfigure(self.config.max_batch_size, self.config.batch_timeout_ms)
        for model, legs in _traffic_splits(cfg).items():
            self._traffic[model] = TrafficSplit(model=model, legs=legs)

    def reload(self) -> bool:
        if not self.config_reloader:
            return False
        return self.config_reloader.check()

    # ---------- inference ----------
    def serve(self, inputs, name: Optional[str] = None,
              request_id: Optional[str] = None,
              version: Optional[str] = None) -> Any:
        if request_id is None:
            request_id = f"{time.time_ns()}"
        if name is None:
            name = next(iter(self._models)) if self._models else "default"
        target = version or self.route(name, request_id) or name
        t0 = time.perf_counter()
        try:
            result = self.model_fn(inputs) if self.model_fn else inputs
            latency_us = int((time.perf_counter() - t0) * 1e6)
            toks = self._token_count(inputs, result)
            self.metrics.record(latency_us, toks, error=False)
            return result
        except Exception:
            latency_us = int((time.perf_counter() - t0) * 1e6)
            self.metrics.record(latency_us, 0, error=True)
            raise

    @staticmethod
    def _token_count(inp, out) -> int:
        try:
            return max(0, len(inp))
        except TypeError:
            return 0

    # ---------- batching convenience ----------
    def submit_batch(self, request_id: str) -> int:
        return self.batcher.submit(request_id)

    def drain_batch(self, max_items: int = 1024) -> List[BatchItem]:
        items = self.batcher.drain(max_items)
        self.metrics.requests += 0
        if items:
            self._record_batch(len(items))
        return items

    def _record_batch(self, n: int) -> None:
        self.metrics.record(0, n, error=False)  # placeholder latency

    # ---------- health / warm-up ----------
    def warmup(self, name: Optional[str] = None) -> None:
        self._warmup_remaining = self.config.warmup_iters
        self._warmup_done = False
        self._ready = bool(self.config.ready_after_warmup) and self.config.warmup_iters <= 0
        if self.config.warmup_iters <= 0:
            self._ready = True

    def warmup_tick(self, latency_us: float = 100.0, tokens: int = 1) -> bool:
        if self._warmup_done:
            return self._ready
        self.metrics.record(latency_us, tokens, error=False)
        self._warmup_remaining -= 1
        if self._warmup_remaining <= 0:
            self._warmup_done = True
            self._ready = True
            return True
        return False

    @property
    def ready(self) -> bool:
        return self._ready

    def set_ready(self, ready: bool) -> None:
        self._ready = bool(ready)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ready": self._ready,
            "uptime_ms": int((time.monotonic() - self.started_at) * 1000),
            "models_loaded": len(self._models),
        }

    def versions(self, name: Optional[str] = None) -> Any:
        with self._lock:
            if name is None:
                return [
                    {"name": n, "versions": [
                        {"version_id": v.version_id, "description": v.description,
                         "weight": v.weight,
                         "active": v.version_id == self._active.get(n),
                         "deployed_at": v.deployed_at}
                        for v in vs
                    ]}
                    for n, vs in self._models.items()
                ]
            if name in self._models:
                return [{
                    "version_id": v.version_id,
                    "description": v.description,
                    "weight": v.weight,
                    "active": v.version_id == self._active.get(name),
                    "deployed_at": v.deployed_at,
                } for v in self._models[name]]
        return None if name else []


class WorkerPool:
    """Horizontal-scaling worker pool: each worker drains the dynamic batch
    queue and calls ``run_batch``."""

    def __init__(self, engine: ServingEngine, run_batch: Callable[[List[BatchItem]], Any],
                 n_workers: int = 1):
        self.engine = engine
        self.run_batch = run_batch
        self.n_workers = max(1, n_workers)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._pool = ThreadPoolExecutor(max_workers=self.n_workers,
                                         thread_name_prefix="sneppx-worker")
        for _ in range(self.n_workers):
            self._pool.submit(self._loop)

    def _loop(self) -> None:
        while not self._stop.is_set():
            items = self.engine.batcher.drain()
            if items:
                try:
                    self.run_batch(items)
                except Exception as e:  # noqa: BLE001
                    print(f"[sneppx serving] worker error: {e!r}")
            else:
                time.sleep(min(self.engine.config.batch_timeout_ms / 2000.0, 0.005))

    def stop(self) -> None:
        self._stop.set()
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=True)


# ===========================================================================
# Standalone HTTP server (stdlib) — no FastAPI required
# ===========================================================================

class ServingServer:
    """Lightweight standalone server over stdlib http.server."""

    def __init__(self, engine: ServingEngine, run_fn: Optional[Callable] = None,
                 host: Optional[str] = None, port: Optional[int] = None):
        self.engine = engine
        self.run_fn = run_fn or (lambda inputs: inputs)
        self.host = host or engine.config.listen
        self.port = int(port if port is not None else engine.config.port)
        self._httpd: Any = None

    def _handle(self, handler):
        eng = self.engine

        def dispatch(handler, method, path, body):
            if path in ("/", "/v1/health", "/healthz"):
                return 200, {"Content-Type": "application/json"}, \
                    json.dumps({"status": "ok", **eng.health()})
            if path in ("/readyz", "/v1/ready"):
                if eng.ready:
                    return 200, {"Content-Type": "application/json"}, \
                        json.dumps({"ready": True, "uptime_ms": int((time.monotonic() - eng.started_at) * 1000)})
                return 503, {"Content-Type": "application/json"}, \
                    json.dumps({"ready": False})
            if path == "/metrics":
                accept = handler.headers.get("Accept", "")
                want_json = ("application/json" in accept) or \
                    ("format=json" in urlparse(handler.path).query)
                if want_json:
                    return 200, {"Content-Type": "application/json"}, \
                        json.dumps(eng.metrics.to_dict())
                return 200, {"Content-Type": "text/plain; version=0.0.4"}, \
                    eng.metrics.render_prometheus()
            if path.startswith("/v1/models/versions"):
                name = handler.path_query("model")
                if name:
                    v = eng.versions(name)
                    return 200, {"Content-Type": "application/json"}, \
                        json.dumps({"name": name, "versions": v or []})
                return 200, {"Content-Type": "application/json"}, \
                    json.dumps({"models": eng.versions()})
            if path == "/v1/deploy" and method == "POST":
                data = json.loads(body or "{}")
                ok = eng.deploy(data.get("model", "default"),
                                data.get("version", "v1"),
                                data.get("description", ""),
                                weight=int(data.get("weight", 0)),
                                promote=bool(data.get("promote", False)))
                return (200 if ok else 409), {"Content-Type": "application/json"}, \
                    json.dumps({"status": "deployed" if ok else "error"})
            if path == "/v1/traffic" and method == "POST":
                data = json.loads(body or "{}")
                legs = []
                for part in str(data.get("split", "")).split(","):
                    if ":" in part:
                        vid, w = part.split(":", 1)
                        legs.append((vid.strip(), int(w.strip())))
                ok = eng.set_traffic(data.get("model", "default"), legs)
                return (200 if ok else 404), {"Content-Type": "application/json"}, \
                    json.dumps({"status": "ok" if ok else "error"})
            if path == "/v1/generate" and method == "POST":
                data = json.loads(body or "{}")
                rid = data.get("request_id") or f"{time.time_ns()}"
                version = data.get("version") or eng.route(data.get("model", "default"), rid)
                result = eng.serve(data.get("inputs"), name=data.get("model"),
                                   request_id=rid, version=version)
                out = result if isinstance(result, (dict, list)) else {"output": str(result)}
                return 200, {"Content-Type": "application/json"}, json.dumps({"version": version, **out})
            return 404, {"Content-Type": "application/json"}, \
                json.dumps({"error": "not_found", "path": path})

        handler._dispatch = staticmethod(dispatch)
        return handler

    def _build_handler(self):
        from http.server import BaseHTTPRequestHandler

        eng = self.engine

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence
                pass

            def path_query(self, key):
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                return q.get(key, [None])[0]

            def _body(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                return self.rfile.read(length).decode("utf-8") if length else ""

            def _respond(self):
                dispatch = getattr(self, "_dispatch", None)
                if not dispatch:
                    self.send_error(404)
                    return
                status, headers, body = dispatch(self, self.command,
                                                 urlparse(self.path).path, self._body())
                self.send_response(status)
                self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def do_GET(self):    self._respond()
            def do_POST(self):   self._respond()
            def do_PATCH(self):  self._respond()

        return self._handle(Handler)

    def serve_forever(self) -> None:
        from http.server import ThreadingHTTPServer
        Handler = self._build_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        try:
            self.engine.warmup()
            self._httpd.serve_forever()
        finally:
            self._httpd.server_close()

    def start_background(self) -> "ServingServer":
        """Bind on an OS-assigned port (port=0) and serve in a daemon thread."""
        from http.server import ThreadingHTTPServer
        Handler = self._build_handler()
        # port 0 -> OS chooses
        self._httpd = ThreadingHTTPServer((self.host, 0), Handler)
        self.port = self._httpd.server_address[1]
        self.engine.warmup()
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def serve_from_config(path: str, run_fn: Optional[Callable] = None) -> ServingEngine:
    """Load a YAML config, build the engine + optional worker pool, and start."""
    cfg = _load_yaml(path)
    svc = _serving_params(cfg)
    serving_cfg = ServingConfig(
        max_batch_size=int(svc.get("max_batch_size", 8)),
        batch_timeout_ms=float(svc.get("batch_timeout_ms", 5.0)),
        n_workers=int(svc.get("n_workers", 1)),
        warmup_iters=int(svc.get("warmup_iters", 8)),
        listen=svc.get("listen", "127.0.0.1"),
        port=int(svc.get("port", 8300)),
    )
    engine = ServingEngine(config=serving_cfg, model_fn=run_fn)
    engine.set_config_path(path)
    engine.warmup()
    for name, spec in (cfg.get("models") or {}).items():
        if isinstance(spec, dict):
            engine.register_model(name, spec.get("version_id", "v1"),
                                 spec.get("description", ""),
                                 weight=int(spec.get("weight", 100)))
        else:
            engine.register_model(name, str(spec))
    for model, legs in _traffic_splits(cfg).items():
        engine._traffic[model] = TrafficSplit(model=model, legs=legs)
    return engine
