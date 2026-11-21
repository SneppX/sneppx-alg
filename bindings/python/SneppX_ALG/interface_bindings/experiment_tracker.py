"""Experiment tracking — local JSON backend + optional W&B/MLflow integration."""

import os
import json
import time
import uuid
import glob
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Callable

# ===========================================================================
#  Data models
# ===========================================================================


@dataclass
class ExperimentRun:
    run_id: str = ""
    experiment_name: str = "default"
    run_name: str = ""
    status: str = "running"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def log_metric(self, key: str, value: float, step: int = 0):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({"step": step, "value": value, "time": time.time()})

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_param(self, key: str, value: Any):
        self.params[key] = value

    def log_params(self, params: Dict[str, Any]):
        self.params.update(params)

    def log_artifact(self, path: str):
        if path not in self.artifacts:
            self.artifacts.append(path)

    def set_tag(self, key: str, value: str):
        self.tags[key] = value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ===========================================================================
#  Local backend
# ===========================================================================


class LocalBackend:
    """Stores runs as JSON files in a directory."""

    def __init__(self, storage_dir: str = "experiments"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _run_path(self, run_id: str) -> str:
        return os.path.join(self.storage_dir, f"{run_id}.json")

    def save(self, run: ExperimentRun):
        path = self._run_path(run.run_id)
        with open(path, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def load(self, run_id: str) -> Optional[ExperimentRun]:
        path = self._run_path(run_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return ExperimentRun.from_dict(data)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        runs = []
        for path in sorted(glob.glob(os.path.join(self.storage_dir, "*.json"))):
            with open(path) as f:
                data = json.load(f)
            run = ExperimentRun.from_dict(data)
            if experiment_name is None or run.experiment_name == experiment_name:
                runs.append(run)
        return sorted(runs, key=lambda r: r.start_time, reverse=True)

    def delete_run(self, run_id: str):
        path = self._run_path(run_id)
        if os.path.exists(path):
            os.remove(path)


# ===========================================================================
#  Tracker base + implementations
# ===========================================================================


class Tracker:
    """Base experiment tracker. All methods are no-ops by default."""

    def __init__(self):
        self._run: Optional[ExperimentRun] = None

    @property
    def run(self) -> Optional[ExperimentRun]:
        return self._run

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        pass

    def log_metric(self, key: str, value: float, step: int = 0):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_param(self, key: str, value: Any):
        pass

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            self.log_param(k, v)

    def log_artifact(self, path: str):
        pass

    def set_tag(self, key: str, value: str):
        pass

    def set_status(self, status: str):
        pass

    def end_run(self):
        pass


class LogTracker(Tracker):
    """Logs metrics to stdout via print."""

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        print(
            f"[Experiment] Starting run — experiment={experiment_name} name={run_name}"
        )

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        parts = [f"  Step {step}"] + [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(" | ".join(parts))

    def end_run(self):
        print("[Experiment] Run ended")


class ExperimentTracker(Tracker):
    """Full experiment tracker with local JSON storage.

    Usage:
        tracker = ExperimentTracker(storage_dir="experiments")
        tracker.start_run("my_experiment", params={"lr": 0.001})
        tracker.log_metrics({"loss": 0.5, "acc": 0.8}, step=10)
        tracker.end_run()
    """

    def __init__(self, storage_dir: str = "experiments"):
        super().__init__()
        self._backend = LocalBackend(storage_dir)
        self._lock = threading.Lock()

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        with self._lock:
            run_id = f"{experiment_name}_{uuid.uuid4().hex[:8]}"
            self._run = ExperimentRun(
                run_id=run_id,
                experiment_name=experiment_name,
                run_name=run_name or run_id,
                status="running",
                params=params or {},
                config=config or {},
                tags=tags or {},
                start_time=time.time(),
            )
            self._backend.save(self._run)
            print(f"[Experiment] Started run {run_id} ({experiment_name})")

    def log_metric(self, key: str, value: float, step: int = 0):
        with self._lock:
            if self._run is not None:
                self._run.log_metric(key, value, step)
                self._backend.save(self._run)

    def log_param(self, key: str, value: Any):
        with self._lock:
            if self._run is not None:
                self._run.log_param(key, value)
                self._backend.save(self._run)

    def log_params(self, params: Dict[str, Any]):
        with self._lock:
            if self._run is not None:
                self._run.log_params(params)
                self._backend.save(self._run)

    def log_artifact(self, path: str):
        with self._lock:
            if self._run is not None:
                self._run.log_artifact(path)
                self._backend.save(self._run)

    def set_tag(self, key: str, value: str):
        with self._lock:
            if self._run is not None:
                self._run.set_tag(key, value)
                self._backend.save(self._run)

    def set_status(self, status: str):
        with self._lock:
            if self._run is not None:
                self._run.status = status
                self._backend.save(self._run)

    def end_run(self):
        with self._lock:
            if self._run is not None:
                self._run.end_time = time.time()
                if self._run.status == "running":
                    self._run.status = "completed"
                self._backend.save(self._run)
                print(f"[Experiment] Run {self._run.run_id} ended ({self._run.status})")

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        return self._backend.list_runs(experiment_name)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self._backend.load(run_id)


class WandbTracker(Tracker):
    """W&B integration. Falls back to LogTracker if wandb not installed."""

    def __init__(self, **init_kwargs):
        super().__init__()
        self._init_kwargs = init_kwargs
        self._wandb = None
        self._available = False

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        try:
            import wandb

            self._wandb = wandb
            self._available = True
        except ImportError:
            print("[WandbTracker] wandb not installed, skipping")
            return

        wandb_kwargs = {**self._init_kwargs}
        wandb_kwargs.setdefault("project", experiment_name)
        if run_name:
            wandb_kwargs["name"] = run_name
        if config:
            wandb_kwargs["config"] = config
        self._wandb.init(**wandb_kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_param(self, key: str, value: Any):
        if self._wandb is not None:
            self._wandb.config[key] = value

    def log_params(self, params: Dict[str, Any]):
        if self._wandb is not None:
            self._wandb.config.update(params)

    def log_artifact(self, path: str):
        if self._wandb is not None and os.path.exists(path):
            art = self._wandb.Artifact("artifact", type="data")
            art.add_file(path)
            self._wandb.log_artifact(art)

    def set_tag(self, key: str, value: str):
        if self._wandb is not None:
            self._wandb.tags[key] = value

    def set_status(self, status: str):
        pass

    def end_run(self):
        if self._wandb is not None:
            self._wandb.finish()


class MLflowTracker(Tracker):
    """MLflow integration. Falls back to LogTracker if mlflow not installed."""

    def __init__(self, tracking_uri: Optional[str] = None, **init_kwargs):
        super().__init__()
        self._tracking_uri = tracking_uri
        self._init_kwargs = init_kwargs
        self._mlflow = None
        self._available = False

    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str = "",
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        try:
            import mlflow

            self._mlflow = mlflow
            self._available = True
        except ImportError:
            print("[MLflowTracker] mlflow not installed, skipping")
            return

        if self._tracking_uri:
            self._mlflow.set_tracking_uri(self._tracking_uri)

        self._mlflow.set_experiment(experiment_name)
        run_args = {}
        if run_name:
            run_args["run_name"] = run_name
        if tags:
            run_args["tags"] = tags
        self._mlflow.start_run(**run_args)

        if params:
            self._mlflow.log_params(params)
        if config:
            self._mlflow.log_params({f"config.{k}": v for k, v in config.items()})

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any):
        if self._mlflow is not None:
            self._mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_artifact(self, path: str):
        if self._mlflow is not None and os.path.exists(path):
            self._mlflow.log_artifact(path)

    def set_tag(self, key: str, value: str):
        if self._mlflow is not None:
            self._mlflow.set_tag(key, value)

    def set_status(self, status: str):
        if self._mlflow is not None:
            self._mlflow.set_tag("status", status)

    def end_run(self):
        if self._mlflow is not None:
            self._mlflow.end_run()


# ===========================================================================
#  Composite tracker
# ===========================================================================


class CompositeTracker(Tracker):
    """Combines multiple trackers into one. Dispatches to all children."""

    def __init__(self, trackers: Optional[List[Tracker]] = None):
        super().__init__()
        self._trackers = trackers or []

    def add(self, tracker: Tracker):
        self._trackers.append(tracker)

    def start_run(self, **kwargs):
        for t in self._trackers:
            t.start_run(**kwargs)

    def log_metric(self, key: str, value: float, step: int = 0):
        for t in self._trackers:
            t.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        for t in self._trackers:
            t.log_metrics(metrics, step)

    def log_param(self, key: str, value: Any):
        for t in self._trackers:
            t.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        for t in self._trackers:
            t.log_params(params)

    def log_artifact(self, path: str):
        for t in self._trackers:
            t.log_artifact(path)

    def set_tag(self, key: str, value: str):
        for t in self._trackers:
            t.set_tag(key, value)

    def set_status(self, status: str):
        for t in self._trackers:
            t.set_status(status)

    def end_run(self):
        for t in self._trackers:
            t.end_run()


__all__ = [
    "ExperimentRun",
    "LocalBackend",
    "Tracker",
    "LogTracker",
    "ExperimentTracker",
    "WandbTracker",
    "MLflowTracker",
    "CompositeTracker",
]
