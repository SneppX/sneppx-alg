"""MLflow-class MLOps surface: model registry, run lineage, prompt registry,
AI gateway routing and eval monitoring.

This module fills the ``mlflow.lifecycle`` gaps with a dependency-free, local
JSON backend (mirrors the ``mlflow.client.MlflowClient`` surface):

- :class:`ModelRegistry` — registered models, versions, aliases and
  stage transitions (None -> Staging -> Production -> Archived).
- :class:`LineageStore` — run lineage DAG (parent/child runs, model sources,
  code version, dataset hash).
- :class:`PromptRegistry` — versioned prompts/templates with variables.
- :class:`AIGateway` — endpoint routing, API-key management, usage metrics
  and fallback policies.
- :class:`EvalMonitor` — eval summary tracking, delta computation and
  regression alerts.
- :class:`SneppxTrackingClient` — single facade tying the above together.

Everything is pure Python + stdlib; no LLM, no CUDA, no network required.
"""

import json
import os
import re
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "ModelVersion",
    "RegisteredModel",
    "ModelRegistry",
    "LineageStore",
    "PromptVersion",
    "PromptRegistry",
    "GatewayRoute",
    "AIGateway",
    "EvalSummary",
    "EvalMonitor",
    "SneppxTrackingClient",
]

STAGES = ("None", "Staging", "Production", "Archived")


def _now() -> float:
    return time.time()


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


class JSONStore:
    """Thread-safe JSON persistence keyed by (collection, record_id)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, collection: str, record_id: str) -> str:
        cdir = os.path.join(self.root, collection)
        os.makedirs(cdir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", record_id)
        return os.path.join(cdir, f"{safe}.json")

    def put(self, collection: str, record_id: str, data: dict) -> None:
        with self._lock:
            path = self._path(collection, record_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    def get(self, collection: str, record_id: str) -> Optional[dict]:
        with self._lock:
            path = self._path(collection, record_id)
            if not os.path.exists(path):
                return None
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    def list(self, collection: str) -> List[dict]:
        with self._lock:
            cdir = os.path.join(self.root, collection)
            if not os.path.isdir(cdir):
                return []
            out = []
            for name in sorted(os.listdir(cdir)):
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(cdir, name), encoding="utf-8") as f:
                    out.append(json.load(f))
            return out

    def delete(self, collection: str, record_id: str) -> bool:
        with self._lock:
            path = self._path(collection, record_id)
            if os.path.exists(path):
                os.remove(path)
                return True
            return False


# ===========================================================================
#  Model registry
# ===========================================================================


@dataclass
class ModelVersion:
    model_name: str
    version: int
    run_id: str = ""
    stage: str = "None"
    aliases: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    source: str = ""
    created_at: float = 0.0
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RegisteredModel:
    name: str
    versions: List[ModelVersion] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    created_at: float = 0.0

    @property
    def latest_version(self) -> Optional[ModelVersion]:
        return self.versions[-1] if self.versions else None


class ModelRegistry:
    """Create registered models, register versions, alias and promote them."""

    def __init__(self, store: Optional[JSONStore] = None, root: str = "mlflow"):
        self._store = store or JSONStore(root)

    # -- persistence helpers -------------------------------------------------

    def _model(self, name: str) -> Optional[RegisteredModel]:
        data = self._store.get("models", name)
        if data is None:
            return None
        model = RegisteredModel(name=name, tags=data.get("tags", {}),
                                description=data.get("description", ""),
                                created_at=data.get("created_at", 0.0))
        model.versions = [ModelVersion.from_dict(v) for v in data.get("versions", [])]
        return model

    def _save(self, model: RegisteredModel) -> None:
        self._store.put("models", model.name, {
            "name": model.name,
            "tags": model.tags,
            "description": model.description,
            "created_at": model.created_at,
            "versions": [v.to_dict() for v in model.versions],
        })

    # -- public API ----------------------------------------------------------

    def create_registered_model(self, name: str, tags: Optional[Dict[str, str]] = None,
                                description: str = "") -> RegisteredModel:
        if self._model(name) is not None:
            raise ValueError(f"registered model {name!r} already exists")
        model = RegisteredModel(name=name, tags=tags or {}, description=description,
                                created_at=_now())
        self._save(model)
        return model

    def get_registered_model(self, name: str) -> Optional[RegisteredModel]:
        return self._model(name)

    def list_registered_models(self) -> List[str]:
        return [m["name"] for m in self._store.list("models")]

    def delete_registered_model(self, name: str) -> bool:
        return self._store.delete("models", name)

    def register_model(self, name: str, run_id: str = "", params: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, float]] = None, source: str = "") -> ModelVersion:
        model = self._model(name)
        if model is None:
            model = self.create_registered_model(name)
        version = ModelVersion(
            model_name=name,
            version=(model.latest_version.version + 1) if model.latest_version else 1,
            run_id=run_id,
            params=params or {},
            metrics=metrics or {},
            source=source,
            created_at=_now(),
        )
        model.versions.append(version)
        self._save(model)
        return version

    def get_model_version(self, name: str, version: Union[int, str]) -> Optional[ModelVersion]:
        model = self._model(name)
        if model is None:
            return None
        if version == "latest":
            return model.latest_version
        for v in model.versions:
            if str(v.version) == str(version):
                return v
        return None

    def update_model_version(self, version: ModelVersion) -> None:
        model = self._model(version.model_name)
        if model is None:
            raise ValueError(f"no registered model {version.model_name!r}")
        for i, v in enumerate(model.versions):
            if v.version == version.version:
                model.versions[i] = version
                self._save(model)
                return
        raise ValueError(f"version {version.version} not found for {version.model_name!r}")

    def transition_model_version_stage(self, name: str, version: Union[int, str],
                                       stage: str) -> ModelVersion:
        if stage not in STAGES:
            raise ValueError(f"invalid stage {stage!r}, expected one of {STAGES}")
        v = self.get_model_version(name, version)
        if v is None:
            raise ValueError(f"version {version} not found for model {name!r}")
        old = v.stage
        if old in ("Production", "Archived") and stage in ("Production", "Archived"):
            raise ValueError(f"cannot transition {name}@{v.version} from {old!r} to {stage!r}")
        v.stage = stage
        self.update_model_version(v)
        return v

    def set_registered_model_alias(self, name: str, alias: str,
                                   version: Union[int, str]) -> None:
        v = self.get_model_version(name, version)
        if v is None:
            raise ValueError(f"version {version} not found for model {name!r}")
        for other in self._model(name).versions:
            if alias in other.aliases:
                other.aliases.remove(alias)
        v.aliases.append(alias)
        self.update_model_version(v)

    def get_model_version_by_alias(self, name: str, alias: str) -> Optional[ModelVersion]:
        model = self._model(name)
        if model is None:
            return None
        for v in model.versions:
            if alias in v.aliases:
                return v
        return None

    def get_latest_versions(self, name: str, stages: Optional[List[str]] = None) -> List[ModelVersion]:
        model = self._model(name)
        if model is None:
            return []
        if stages is None:
            return [model.latest_version] if model.latest_version else []
        return [v for v in model.versions if v.stage in stages]


# ===========================================================================
#  Run lineage
# ===========================================================================


@dataclass
class LineageRecord:
    run_id: str
    experiment_name: str = "default"
    parent_run_id: str = ""
    child_run_ids: List[str] = field(default_factory=list)
    source_uri: str = ""
    code_version: str = ""
    dataset_hash: str = ""
    model_names: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LineageRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LineageStore:
    """Record and query the training/eval lineage DAG (MLflow "runs" graph)."""

    def __init__(self, store: Optional[JSONStore] = None, root: str = "mlflow"):
        self._store = store or JSONStore(root)

    def _record(self, run_id: str) -> Optional[LineageRecord]:
        data = self._store.get("lineage", run_id)
        return LineageRecord.from_dict(data) if data else None

    def create_run(self, run_id: str, experiment_name: str = "default",
                   parent_run_id: str = "", source_uri: str = "",
                   code_version: str = "", dataset_hash: str = "",
                   tags: Optional[Dict[str, str]] = None) -> LineageRecord:
        if self._record(run_id) is not None:
            raise ValueError(f"lineage record for run {run_id!r} already exists")
        rec = LineageRecord(run_id=run_id, experiment_name=experiment_name,
                            parent_run_id=parent_run_id, source_uri=source_uri,
                            code_version=code_version, dataset_hash=dataset_hash,
                            tags=tags or {}, created_at=_now())
        self._store.put("lineage", run_id, rec.to_dict())
        if parent_run_id and self._record(parent_run_id) is not None:
            parent = self._record(parent_run_id)
            if run_id not in parent.child_run_ids:
                parent.child_run_ids.append(run_id)
                self._store.put("lineage", parent.run_id, parent.to_dict())
        return rec

    def get_run(self, run_id: str) -> Optional[LineageRecord]:
        return self._record(run_id)

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        rec = self._record(run_id)
        if rec is None:
            raise ValueError(f"no lineage record for run {run_id!r}")
        rec.tags.update(tags)
        self._store.put("lineage", run_id, rec.to_dict())

    def associate_model(self, run_id: str, model_name: str) -> None:
        rec = self._record(run_id)
        if rec is None:
            raise ValueError(f"no lineage record for run {run_id!r}")
        if model_name not in rec.model_names:
            rec.model_names.append(model_name)
            self._store.put("lineage", run_id, rec.to_dict())

    def lineage_of(self, run_id: str) -> List[LineageRecord]:
        """Return the ancestor chain (oldest first) of ``run_id``."""
        chain: List[LineageRecord] = []
        cur = self._record(run_id)
        seen = set()
        while cur is not None and cur.run_id not in seen:
            seen.add(cur.run_id)
            chain.append(cur)
            parent = cur.parent_run_id
            cur = self._record(parent) if parent else None
        return list(reversed(chain))

    def descendants_of(self, run_id: str) -> List[LineageRecord]:
        out: List[LineageRecord] = []
        frontier = [run_id]
        while frontier:
            cur = self._record(frontier.pop(0))
            if cur is None:
                continue
            for c in cur.child_run_ids:
                child = self._record(c)
                if child is not None:
                    out.append(child)
                    frontier.append(child.run_id)
        return out


# ===========================================================================
#  Prompt registry
# ===========================================================================


@dataclass
class PromptVersion:
    name: str
    version: int
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def render(self, **values: Any) -> str:
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise ValueError(f"missing prompt variables: {missing}")
        out = self.template
        for key, val in values.items():
            out = out.replace("{" + key + "}", str(val))
        return out


class PromptRegistry:
    """Versioned prompt/template store with rendering (LangSmith-style)."""

    def __init__(self, store: Optional[JSONStore] = None, root: str = "mlflow"):
        self._store = store or JSONStore(root)
        self._vars_re = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def _versions(self, name: str) -> List[PromptVersion]:
        data = self._store.get("prompts", name)
        return [PromptVersion.from_dict(p) for p in data.get("versions", [])] if data else []

    def register_prompt(self, name: str, template: str, description: str = "",
                        tags: Optional[Dict[str, str]] = None) -> PromptVersion:
        versions = self._versions(name)
        version = len(versions) + 1
        variables = sorted(set(self._vars_re.findall(template)))
        prompt = PromptVersion(name=name, version=version, template=template,
                               variables=variables, description=description,
                               tags=tags or {}, created_at=_now())
        versions.append(prompt)
        self._store.put("prompts", name, {"versions": [p.to_dict() for p in versions]})
        return prompt

    def get_prompt(self, name: str, version: Union[int, str] = "latest") -> Optional[PromptVersion]:
        versions = self._versions(name)
        if not versions:
            return None
        if version == "latest":
            return versions[-1]
        for p in versions:
            if p.version == version:
                return p
        return None

    def list_prompts(self) -> List[str]:
        return [p["versions"][-1]["name"] for p in self._store.list("prompts")]

    def render(self, name: str, version: Union[int, str] = "latest", **values: Any) -> str:
        prompt = self.get_prompt(name, version)
        if prompt is None:
            raise ValueError(f"prompt {name!r} not found")
        return prompt.render(**values)


# ===========================================================================
#  AI gateway
# ===========================================================================


@dataclass
class GatewayRoute:
    name: str
    model: str
    provider: str = "local"
    api_key: str = ""
    priority: int = 0
    enabled: bool = True
    usage: Dict[str, float] = field(default_factory=lambda: {"calls": 0.0, "latency": 0.0})


class AIGateway:
    """Route inference requests to named model endpoints with usage tracking.

    ``backend`` is an optional callable ``backend(model, inputs, **kw)``; when
    omitted the gateway serves a deterministic echo fallback so routing logic
    stays testable without an LLM.
    """

    def __init__(self, backend: Optional[Callable] = None, store: Optional[JSONStore] = None,
                 root: str = "mlflow"):
        self._backend = backend
        self._store = store or JSONStore(root)
        self._routes: Dict[str, GatewayRoute] = {}
        self._lock = threading.RLock()
        for data in self._store.list("routes"):
            r = GatewayRoute(**{k: v for k, v in data.items()})
            self._routes[r.name] = r

    def add_route(self, name: str, model: str, provider: str = "local",
                  api_key: str = "", priority: int = 0) -> GatewayRoute:
        route = GatewayRoute(name=name, model=model, provider=provider,
                             api_key=api_key, priority=priority)
        with self._lock:
            self._routes[name] = route
            self._store.put("routes", name, asdict(route))
        return route

    def set_api_key(self, name: str, api_key: str) -> None:
        with self._lock:
            if name not in self._routes:
                raise ValueError(f"unknown route {name!r}")
            self._routes[name].api_key = api_key
            self._store.put("routes", name, asdict(self._routes[name]))

    def enable(self, name: str, enabled: bool = True) -> None:
        with self._lock:
            if name not in self._routes:
                raise ValueError(f"unknown route {name!r}")
            self._routes[name].enabled = enabled
            self._store.put("routes", name, asdict(self._routes[name]))

    def list_routes(self) -> List[GatewayRoute]:
        return sorted(self._routes.values(), key=lambda r: (-r.priority, r.name))

    def get_route(self, name: str) -> Optional[GatewayRoute]:
        return self._routes.get(name)

    def query(self, inputs: Any, route: str = "", model: Optional[str] = None) -> Any:
        """Route ``inputs`` to an endpoint and record usage.

        Resolution order: explicit ``model``/``route`` name, then highest
        priority enabled route matching ``model``, else any enabled route.
        """
        with self._lock:
            candidates = [r for r in self._routes.values() if r.enabled]
            if route:
                match = [r for r in candidates if r.name == route]
                candidates = match or candidates
            if model:
                match = [r for r in candidates if r.model == model]
                if match:
                    candidates = match
            if not candidates:
                raise RuntimeError("no enabled gateway routes")
            target = candidates[0]

        import time as _t
        t0 = _t.perf_counter()
        try:
            if self._backend is not None:
                out = self._backend(target.model, inputs)
            else:
                out = inputs
            return out
        finally:
            dt = _t.perf_counter() - t0
            with self._lock:
                target.usage["calls"] += 1
                target.usage["latency"] += dt
                self._store.put("routes", target.name, asdict(target))

    def usage_report(self) -> Dict[str, Dict[str, float]]:
        return {r.name: dict(r.usage) for r in self._routes.values()}


# ===========================================================================
#  Eval monitoring
# ===========================================================================


@dataclass
class EvalSummary:
    run_id: str
    model_name: str = ""
    metric: str = "accuracy"
    value: float = 0.0
    dataset: str = ""
    extra: Dict[str, float] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EvalMonitor:
    """Track eval summaries per model, detect regressions over time."""

    def __init__(self, store: Optional[JSONStore] = None, root: str = "mlflow"):
        self._store = store or JSONStore(root)

    def log_eval(self, run_id: str, metric: str, value: float, model_name: str = "",
                 dataset: str = "", extra: Optional[Dict[str, float]] = None) -> EvalSummary:
        summary = EvalSummary(run_id=run_id, model_name=model_name, metric=metric,
                              value=value, dataset=dataset, extra=extra or {},
                              created_at=_now())
        self._store.put("evals", f"{run_id}_{metric}_{int(_now() * 1000)}",
                        summary.to_dict())
        return summary

    def history(self, model_name: Optional[str] = None, metric: Optional[str] = None) -> List[EvalSummary]:
        out = []
        for data in self._store.list("evals"):
            s = EvalSummary.from_dict(data)
            if model_name and s.model_name != model_name:
                continue
            if metric and s.metric != metric:
                continue
            out.append(s)
        return sorted(out, key=lambda s: s.created_at)

    def deltas(self, model_name: str, metric: str = "accuracy") -> Dict[str, Optional[float]]:
        hist = self.history(model_name, metric)
        if len(hist) < 2:
            return {"latest": hist[-1].value if hist else None,
                    "delta": None, "trend": "flat"}
        latest = hist[-1].value
        prev = hist[-2].value
        delta = latest - prev
        trend = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        return {"latest": latest, "delta": delta, "trend": trend}

    def regressions(self, threshold: float = -0.01) -> List[EvalSummary]:
        """Eval points that dropped at least ``threshold`` below the best so far."""
        best: Dict[str, float] = {}
        alerts = []
        for s in self.history():
            key = f"{s.model_name}:{s.metric}"
            if s.value < best.get(key, s.value) + threshold and best.get(key) is not None:
                alerts.append(s)
            best[key] = max(best.get(key, -float("inf")), s.value)
        return alerts


# ===========================================================================
#  Unified client
# ===========================================================================


class SneppxTrackingClient:
    """One facade over registry + lineage + prompts + gateway + eval monitor.

    Mirrors the ``MlflowClient`` surface used by platform scripts::

        client = SneppxTrackingClient(root=".sneppx/mlflow")
        mv = client.register_model("gpt-lite", run_id="run_1", metrics={"acc": 0.9})
        client.transition_model_version_stage("gpt-lite", mv.version, "Production")
    """

    def __init__(self, root: str = "mlflow", backend: Optional[Callable] = None):
        store = JSONStore(root)
        self.registry = ModelRegistry(store)
        self.lineage = LineageStore(store)
        self.prompts = PromptRegistry(store)
        self.gateway = AIGateway(backend, store)
        self.evals = EvalMonitor(store)
        self._root = root

    # registry passthroughs
    def create_registered_model(self, name: str, tags=None, description: str = "") -> RegisteredModel:
        return self.registry.create_registered_model(name, tags, description)

    def get_registered_model(self, name: str) -> Optional[RegisteredModel]:
        return self.registry.get_registered_model(name)

    def list_registered_models(self) -> List[str]:
        return self.registry.list_registered_models()

    def register_model(self, name: str, run_id: str = "", params=None, metrics=None,
                       source: str = "") -> ModelVersion:
        return self.registry.register_model(name, run_id, params, metrics, source)

    def get_model_version(self, name: str, version) -> Optional[ModelVersion]:
        return self.registry.get_model_version(name, version)

    def transition_model_version_stage(self, name: str, version, stage: str) -> ModelVersion:
        return self.registry.transition_model_version_stage(name, version, stage)

    def set_registered_model_alias(self, name: str, alias: str, version) -> None:
        self.registry.set_registered_model_alias(name, alias, version)

    def get_model_version_by_alias(self, name: str, alias: str) -> Optional[ModelVersion]:
        return self.registry.get_model_version_by_alias(name, alias)

    def get_latest_versions(self, name: str, stages=None) -> List[ModelVersion]:
        return self.registry.get_latest_versions(name, stages)

    # lineage passthroughs
    def create_run(self, run_id: str, experiment_name: str = "default",
                   parent_run_id: str = "", source_uri: str = "", code_version: str = "",
                   dataset_hash: str = "", tags=None) -> LineageRecord:
        return self.lineage.create_run(run_id, experiment_name, parent_run_id,
                                       source_uri, code_version, dataset_hash, tags)

    def get_run(self, run_id: str) -> Optional[LineageRecord]:
        return self.lineage.get_run(run_id)

    def set_run_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        self.lineage.set_tags(run_id, tags)

    def associate_model(self, run_id: str, model_name: str) -> None:
        self.lineage.associate_model(run_id, model_name)

    def lineage_of(self, run_id: str) -> List[LineageRecord]:
        return self.lineage.lineage_of(run_id)

    def descendants_of(self, run_id: str) -> List[LineageRecord]:
        return self.lineage.descendants_of(run_id)

    # prompt passthroughs
    def register_prompt(self, name: str, template: str, description: str = "", tags=None) -> PromptVersion:
        return self.prompts.register_prompt(name, template, description, tags)

    def get_prompt(self, name: str, version="latest") -> Optional[PromptVersion]:
        return self.prompts.get_prompt(name, version)

    def render_prompt(self, name: str, version="latest", **values: Any) -> str:
        return self.prompts.render(name, version, **values)

    # gateway passthroughs
    def add_route(self, name: str, model: str, provider: str = "local",
                  api_key: str = "", priority: int = 0) -> GatewayRoute:
        return self.gateway.add_route(name, model, provider, api_key, priority)

    def set_api_key(self, name: str, api_key: str) -> None:
        self.gateway.set_api_key(name, api_key)

    def enable_route(self, name: str, enabled: bool = True) -> None:
        self.gateway.enable(name, enabled)

    def query(self, inputs: Any, route: str = "", model: Optional[str] = None) -> Any:
        return self.gateway.query(inputs, route, model)

    def usage_report(self) -> Dict[str, Dict[str, float]]:
        return self.gateway.usage_report()

    # eval passthroughs
    def log_eval(self, run_id: str, metric: str, value: float, model_name: str = "",
                 dataset: str = "", extra=None) -> EvalSummary:
        return self.evals.log_eval(run_id, metric, value, model_name, dataset, extra)

    def eval_deltas(self, model_name: str, metric: str = "accuracy") -> Dict[str, Optional[float]]:
        return self.evals.deltas(model_name, metric)

    def eval_regressions(self, threshold: float = -0.01) -> List[EvalSummary]:
        return self.evals.regressions(threshold)
