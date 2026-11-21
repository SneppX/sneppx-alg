"""Experiment / run tracking.

Structured run-level experiment model with `metadata.json` + `metrics.jsonl`
persistence and parameter/artifact versioning (cf. TensorBoard ``runs/``).

Typical usage::

    tracker = ExperimentStore("runs")
    exp = tracker.create_experiment("cnn_sweep")
    with exp.run({"lr": 0.01, "batch": 64}) as run:
        for step in range(3):
            run.log_metric("loss", 1.0 / (step + 1), step=step)
            run.log_metric("acc", 0.5 + 0.1 * step, step=step)
        run.log_params({"optimizer": "adamw"})
        run.log_artifact("weights.bin")

    best = exp.best_run("acc")
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

__all__ = [
    "Run",
    "Experiment",
    "ExperimentStore",
    "load_experiment",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


class Run:
    """A single training run with metadata, params, metrics and artifacts."""

    def __init__(
        self,
        experiment_name: str = "default",
        run_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_id = run_id or _new_run_id()
        self.params: Dict[str, Any] = dict(params or {})
        self.metrics: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        self.status = "created"
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self.tags: List[str] = []

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> "Run":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.finish()
        return False

    def start(self) -> "Run":
        self.status = "running"
        self.started_at = self.started_at or _now_iso()
        return self

    def finish(self) -> "Run":
        self.status = "completed"
        self.ended_at = _now_iso()
        return self

    def fail(self, error: str) -> "Run":
        self.status = "failed"
        self.ended_at = _now_iso()
        self.metrics.append({"step": -1, "error": error})
        return self

    # ---- logging ---------------------------------------------------------
    def log_params(self, params: Dict[str, Any]) -> "Run":
        self.params.update(params)
        return self

    def log_metric(self, key: str, value: Any, step: Optional[int] = None,
                   **extra: Any) -> "Run":
        entry = {"metric": key, "value": value}
        if step is not None:
            entry["step"] = step
        entry.update(extra)
        self.metrics.append(entry)
        return self

    def log_artifact(self, path: str) -> "Run":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.artifacts.append(path)
        return self

    def add_tag(self, tag: str) -> "Run":
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    # ---- queries ---------------------------------------------------------
    def metric_history(self, key: str) -> List[Dict[str, Any]]:
        return [m for m in self.metrics if m.get("metric") == key]

    def best(self, key: str, mode: str = "max") -> Optional[float]:
        vals = [m["value"] for m in self.metric_history(key)
                if isinstance(m["value"], (int, float, np.floating))]
        if not vals:
            return None
        return float(max(vals)) if mode == "max" else float(min(vals))

    def last(self, key: str) -> Optional[Any]:
        history = self.metric_history(key)
        return history[-1]["value"] if history else None

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment_name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tags": self.tags,
            "params": self.params,
            "artifacts": self.artifacts,
        }

    def metrics_jsonl(self) -> str:
        return "\n".join(json.dumps(m) for m in self.metrics)

    def save(self, base_dir: str) -> str:
        run_dir = os.path.join(base_dir, self.experiment_name, self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
            f.write(self.metrics_jsonl())
            if self.metrics:
                f.write("\n")
        return run_dir

    @classmethod
    def from_dir(cls, run_dir: str) -> "Run":
        with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        run = cls(meta.get("experiment", "default"), meta.get("run_id"))
        run.status = meta.get("status", "created")
        run.started_at = meta.get("started_at")
        run.ended_at = meta.get("ended_at")
        run.tags = meta.get("tags", [])
        run.params = meta.get("params", {})
        run.artifacts = meta.get("artifacts", [])
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        run.metrics.append(json.loads(line))
        return run


class Experiment:
    """A named collection of runs."""

    def __init__(self, name: str, runs: Optional[List[Run]] = None):
        self.name = name
        self.runs: List[Run] = list(runs or [])

    def run(self, params: Optional[Dict[str, Any]] = None,
            run_id: Optional[str] = None) -> Run:
        run = Run(self.name, run_id=run_id, params=params)
        self.runs.append(run)
        return run

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def best_run(self, key: str, mode: str = "max") -> Optional[Run]:
        scored = [(r.best(key, mode), r) for r in self.runs]
        scored = [(v, r) for v, r in scored if v is not None]
        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=(mode == "max"))
        return scored[0][1]

    def save(self, base_dir: str) -> str:
        for run in self.runs:
            run.save(base_dir)
        return os.path.join(base_dir, self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "run_ids": [r.run_id for r in self.runs],
        }

    @classmethod
    def from_dir(cls, base_dir: str) -> "Experiment":
        runs = []
        if os.path.isdir(base_dir):
            for entry in sorted(os.listdir(base_dir)):
                run_dir = os.path.join(base_dir, entry)
                if os.path.isfile(os.path.join(run_dir, "metadata.json")):
                    try:
                        runs.append(Run.from_dir(run_dir))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return cls(os.path.basename(os.path.normpath(base_dir)), runs)


class ExperimentStore:
    """Top-level manager over a root directory of experiments."""

    def __init__(self, root: str = "runs"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def create_experiment(self, name: str) -> Experiment:
        return Experiment(name)

    def save_experiment(self, experiment: Experiment) -> str:
        return experiment.save(self.root)

    def load_experiment(self, name: str) -> Experiment:
        return Experiment.from_dir(os.path.join(self.root, name))

    def list_experiments(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            e for e in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, e))
        )


def load_experiment(path: str) -> Experiment:
    """Load an experiment directory (or a single run directory)."""
    return Experiment.from_dir(path)
