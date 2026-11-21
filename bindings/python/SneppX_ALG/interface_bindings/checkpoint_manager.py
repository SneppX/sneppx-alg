"""Checkpoint Manager — high-level checkpoint save/load with best-k retention."""

import os
import re
import json
import time
import glob
import pickle
import random
import shutil
import threading
from typing import Optional, Dict, Any, List, Callable

import numpy as np

from .tensor import Tensor
from .nn import Module
from .optim import Optimizer
from .checkpoint import CheckpointWriter, CheckpointReader, validate_checkpoint
from .amp import GradScaler


def _serialize_tensor_data(t: Tensor) -> np.ndarray:
    return t.data.copy()


def _deserialize_tensor_data(arr: np.ndarray, dtype: str = "float32") -> Tensor:
    return Tensor(arr, dtype=dtype)


class CheckpointManager:
    """High-level checkpoint manager with best-k retention and async save.

    Usage:
        ckpt_mgr = CheckpointManager("checkpoints", keep_last=5, keep_best=3)
        trainer = Trainer(model, config)
        ckpt_mgr.save(trainer, metric=val_loss, is_best=(val_loss < best))
        ckpt_mgr.load_latest(trainer)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last: int = 5,
        keep_best: int = 3,
        async_save: bool = True,
        metric_mode: str = "min",
        save_format: str = "pickle",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.async_save = async_save
        self.metric_mode = metric_mode
        self.save_format = save_format
        self._save_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._best_metrics: List[Dict[str, Any]] = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _latest_path(self) -> str:
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.ckpt")
        files = glob.glob(pattern)
        if not files:
            return ""

        def _step_from_path(p):
            m = re.search(r"checkpoint_(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else 0

        return max(files, key=_step_from_path)

    def _all_checkpoints(self) -> List[str]:
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.ckpt")
        return sorted(
            glob.glob(pattern),
            key=lambda p: (
                int(re.search(r"checkpoint_(\d+)", os.path.basename(p)).group(1))
                if re.search(r"checkpoint_(\d+)", os.path.basename(p))
                else 0
            ),
        )

    def _best_checkpoints(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.checkpoint_dir, "best_*.ckpt")))

    def _build_trainer_state(self, trainer) -> dict:
        import random as _random

        state = {
            "step": trainer._step,
            "epoch": trainer._epoch,
            "best_metric": trainer._best_metric,
            "config": trainer.config.to_dict(),
            "model_state": trainer.model.state_dict(),
            "optimizer_state": (
                trainer.optimizer.state_dict()
                if hasattr(trainer.optimizer, "state_dict")
                else {}
            ),
            "rng_state": {
                "random": _random.getstate(),
                "numpy": np.random.get_state(),
            },
        }
        if hasattr(trainer, "scaler") and trainer.scaler is not None:
            state["scaler_state"] = {
                "scale": trainer.scaler.scale,
                "growth_factor": trainer.scaler.growth_factor,
                "backoff_factor": trainer.scaler.backoff_factor,
                "growth_interval": trainer.scaler.growth_interval,
                "_growth_tracker": getattr(trainer.scaler, "_growth_tracker", 0),
            }
        return state

    def _restore_trainer_state(self, trainer, state: dict):
        import random as _random

        trainer._step = state.get("step", 0)
        trainer._epoch = state.get("epoch", 0)
        trainer._best_metric = state.get("best_metric", None)
        rng = state.get("rng_state", {})
        if "random" in rng:
            _random.setstate(rng["random"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        model_sd = state.get("model_state", {})
        if model_sd:
            trainer.model.load_state_dict(model_sd)
        opt_sd = state.get("optimizer_state", {})
        if opt_sd and hasattr(trainer.optimizer, "load_state_dict"):
            trainer.optimizer.load_state_dict(opt_sd)
        scaler_sd = state.get("scaler_state")
        if scaler_sd and hasattr(trainer, "scaler") and trainer.scaler is not None:
            trainer.scaler.scale = scaler_sd["scale"]
            trainer.scaler.growth_factor = scaler_sd["growth_factor"]
            trainer.scaler.backoff_factor = scaler_sd["backoff_factor"]
            trainer.scaler.growth_interval = scaler_sd["growth_interval"]
            trainer.scaler._growth_tracker = scaler_sd.get("_growth_tracker", 0)

    def _do_save(self, path: str, state: dict):
        if self.save_format == "binary":
            self._save_binary(path, state)
        else:
            self._save_pickle(path, state)

    def _save_pickle(self, path: str, state: dict):
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _save_binary(self, path: str, state: dict):
        model_sd = state.get("model_state", {})
        meta = {k: v for k, v in state.items() if k != "model_state"}
        meta_pickled = pickle.dumps(meta)
        w = CheckpointWriter(path)
        for name, arr in model_sd.items():
            w.write_tensor(arr.tobytes(), shape=arr.shape, dtype=0)
        w.write_tensor(meta_pickled, shape=(len(meta_pickled),), dtype=0)
        w.write_metadata(
            {
                "tensor_names": list(model_sd.keys()),
                "tensor_shapes": {n: list(a.shape) for n, a in model_sd.items()},
                "meta_tensor_idx": len(model_sd),
            }
        )
        w.close()

    def _do_load(self, path: str) -> dict:
        if self.save_format == "binary":
            return self._load_binary(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_binary(self, path: str) -> dict:
        r = CheckpointReader(path)
        meta = r.read_metadata()
        tensor_names = meta.get("tensor_names", [])
        tensor_shapes_map = meta.get("tensor_shapes", {})
        model_sd = {}
        for idx, name in enumerate(tensor_names):
            raw = r.read_tensor(idx)
            shape = tuple(tensor_shapes_map.get(name, (len(raw),)))
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            model_sd[name] = arr
        meta_tensor_idx = meta.get("meta_tensor_idx", len(tensor_names))
        meta_raw = r.read_tensor(meta_tensor_idx)
        main_meta = pickle.loads(meta_raw) if meta_raw else {}
        main_meta["model_state"] = model_sd
        r.close()
        return main_meta

    def save(
        self,
        trainer,
        metric: Optional[float] = None,
        is_best: Optional[bool] = None,
        tag: str = "",
    ):
        step = trainer._step
        state = self._build_trainer_state(trainer)
        if metric is not None:
            state["metric"] = metric

        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.ckpt")
        if tag:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}_{tag}.ckpt")

        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()

        if self.async_save:
            self._save_thread = threading.Thread(
                target=self._do_save, args=(path, state)
            )
            self._save_thread.start()
        else:
            self._do_save(path, state)

        _best = is_best
        if _best is None and metric is not None:
            if self.metric_mode == "min":
                best_val = min(
                    (s.get("metric", float("inf")) for s in self._best_metrics),
                    default=float("inf"),
                )
                _best = metric < best_val - 1e-8
            else:
                best_val = max(
                    (s.get("metric", float("-inf")) for s in self._best_metrics),
                    default=float("-inf"),
                )
                _best = metric > best_val + 1e-8

        if _best:
            best_tag = f"metric={metric:.6f}" if metric is not None else "best"
            best_path = os.path.join(
                self.checkpoint_dir, f"best_{step}_{best_tag}.ckpt"
            )
            self._do_save(best_path, state)
            self._best_metrics.append(
                {"path": best_path, "metric": metric, "step": step}
            )
            self._prune_best()

        self._prune_last()

    def load_latest(self, trainer) -> bool:
        path = self._latest_path()
        if not path or not os.path.exists(path):
            return False
        return self.load(trainer, path)

    def load_best(self, trainer) -> bool:
        def _extract_metric(p):
            m = re.search(r"metric=([^\.]+)", os.path.basename(p))
            if m:
                return float(m.group(1))
            return 0.0

        best_files = sorted(
            self._best_checkpoints(),
            key=_extract_metric,
            reverse=(self.metric_mode == "max"),
        )
        if not best_files:
            return self.load_latest(trainer)
        return self.load(trainer, best_files[0])

    def load(self, trainer, path: str) -> bool:
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False
        try:
            state = self._do_load(path)
            self._restore_trainer_state(trainer, state)
            print(f"Checkpoint loaded: {path} (step {state.get('step', 0)})")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint {path}: {e}")
            return False

    def _prune_last(self):
        if self.keep_last <= 0:
            return
        ckpts = self._all_checkpoints()
        while len(ckpts) > self.keep_last:
            os.remove(ckpts.pop(0))

    def _prune_best(self):
        if self.keep_best <= 0:
            return
        best_sorted = sorted(
            self._best_metrics,
            key=lambda x: x["metric"] if x["metric"] is not None else float("inf"),
            reverse=(self.metric_mode == "max"),
        )
        while len(best_sorted) > self.keep_best:
            entry = best_sorted.pop()
            if os.path.exists(entry["path"]):
                os.remove(entry["path"])
            self._best_metrics.remove(entry)

    def summary(self) -> Dict[str, Any]:
        ckpts = self._all_checkpoints()
        bests = self._best_checkpoints()
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "total_checkpoints": len(ckpts),
            "best_checkpoints": len(bests),
            "latest": (
                os.path.basename(self._latest_path()) if self._latest_path() else None
            ),
            "last_checkpoints": [os.path.basename(p) for p in ckpts[-3:]],
            "best_metrics": list(self._best_metrics),
        }

    def destroy(self):
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()


__all__ = ["CheckpointManager"]
