"""Hyperparameter search — search spaces, samplers, and trial runner."""

import os
import math
import time
import json
import random
import itertools
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union

import numpy as np

from .experiment_tracker import ExperimentTracker, ExperimentRun

# ===========================================================================
#  Search Space
# ===========================================================================


class Param:
    """Base class for a hyperparameter distribution."""

    def sample(self, rng=None):
        rng = rng or random.Random()
        return self._sample(rng)

    def _sample(self, rng):
        raise NotImplementedError

    def grid(self, num=10):
        raise NotImplementedError


class UniformFloat(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def _sample(self, rng):
        return rng.uniform(self.low, self.high)

    def grid(self, num=10):
        step = (self.high - self.low) / max(num - 1, 1)
        return [self.low + i * step for i in range(num)]


class LogUniformFloat(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self._log_low = math.log(low) if low > 0 else -10
        self._log_high = math.log(high)

    def _sample(self, rng):
        return math.exp(rng.uniform(self._log_low, self._log_high))

    def grid(self, num=10):
        log_vals = [
            self._log_low + i * (self._log_high - self._log_low) / max(num - 1, 1)
            for i in range(num)
        ]
        return [math.exp(v) for v in log_vals]


class UniformInt(Param):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def _sample(self, rng):
        return rng.randint(self.low, self.high)

    def grid(self, num=None):
        return list(range(self.low, self.high + 1))


class Choice(Param):
    def __init__(self, values):
        self.values = values

    def _sample(self, rng):
        return rng.choice(self.values)

    def grid(self, num=None):
        return self.values[:]


class Fixed(Param):
    def __init__(self, value):
        self.value = value

    def _sample(self, rng):
        return self.value

    def grid(self, num=None):
        return [self.value]


# ===========================================================================
#  Search Space container
# ===========================================================================


class SearchSpace:
    """Collection of named hyperparameters with their distributions.

    Usage:
        space = SearchSpace()
        space.add("lr", LogUniformFloat(1e-5, 1e-1))
        space.add("batch_size", Choice([16, 32, 64]))
        space.add("optimizer", Choice(["adamw", "sgd"]))
    """

    def __init__(self):
        self._params: Dict[str, Param] = {}

    def add(self, name: str, param: Param):
        self._params[name] = param

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        rng = rng or random.Random()
        return {k: v.sample(rng) for k, v in self._params.items()}

    def grid_size(self) -> int:
        total = 1
        for p in self._params.values():
            total *= len(p.grid())
        return total

    def grid(self) -> List[Dict[str, Any]]:
        keys = list(self._params.keys())
        grids = [self._params[k].grid() for k in keys]
        results = []
        for combo in itertools.product(*grids):
            results.append(dict(zip(keys, combo)))
        return results

    @property
    def names(self) -> List[str]:
        return list(self._params.keys())

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"SearchSpace({items})"

    def to_dict(self) -> dict:
        return {k: repr(v) for k, v in self._params.items()}


# ===========================================================================
#  Trial
# ===========================================================================


@dataclass
class Trial:
    index: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[float] = None
    run_id: str = ""
    start_time: float = 0.0
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
#  Samplers
# ===========================================================================


class Sampler:
    """Base sampler. Generates trial parameter sets."""

    def __init__(self, space: SearchSpace, seed: int = 42):
        self.space = space
        self.rng = random.Random(seed)

    def sample(self) -> Dict[str, Any]:
        raise NotImplementedError

    def total_trials(self) -> Optional[int]:
        return None


class GridSampler(Sampler):
    """Exhaustive grid search over the search space."""

    def __init__(self, space: SearchSpace):
        super().__init__(space)
        self._grid = space.grid()
        self._index = 0

    def sample(self) -> Dict[str, Any]:
        if self._index >= len(self._grid):
            raise StopIteration("Grid exhausted")
        params = self._grid[self._index]
        self._index += 1
        return params

    def total_trials(self) -> Optional[int]:
        return len(self._grid)


class RandomSampler(Sampler):
    """Random search with optional max trials."""

    def __init__(self, space: SearchSpace, n_trials: int = 20, seed: int = 42):
        super().__init__(space, seed)
        self.n_trials = n_trials
        self._count = 0

    def sample(self) -> Dict[str, Any]:
        if self._count >= self.n_trials:
            raise StopIteration("Max trials reached")
        self._count += 1
        return self.space.sample(self.rng)

    def total_trials(self) -> Optional[int]:
        return self.n_trials


class BayesianSampler(Sampler):
    """Simple Bayesian optimization via random sampling with best-worst exploration.

    Uses a surrogate: samples candidates from the space, predicts via distance-weighted
    interpolation from observed results, picks the candidate with best predicted score.
    Falls back to random for early trials.
    """

    def __init__(
        self, space: SearchSpace, n_trials: int = 30, n_initial: int = 5, seed: int = 42
    ):
        super().__init__(space, seed)
        self.n_trials = n_trials
        self.n_initial = n_initial
        self._count = 0
        self._observed: List[Dict[str, Any]] = []
        self._scores: List[float] = []

    def sample(self) -> Dict[str, Any]:
        if self._count >= self.n_trials:
            raise StopIteration("Max trials reached")
        self._count += 1

        if self._count <= self.n_initial or not self._observed:
            return self.space.sample(self.rng)

        return self._acq_max()

    def _acq_max(self) -> Dict[str, Any]:
        candidates = [self.space.sample(self.rng) for _ in range(50)]
        best_score = float("inf")
        best_candidate = candidates[0]

        for cand in candidates:
            pred = self._predict(cand)
            if pred < best_score:
                best_score = pred
                best_candidate = cand
        return best_candidate

    def _predict(self, candidate: Dict[str, Any]) -> float:
        if not self._observed:
            return 0.0

        weights = []
        values = []
        for obs, score in zip(self._observed, self._scores):
            dist = self._distance(candidate, obs)
            w = math.exp(-dist)
            weights.append(w)
            values.append(score)

        total_w = sum(weights)
        if total_w < 1e-12:
            return sum(values) / len(values)
        return sum(w * v for w, v in zip(weights, values)) / total_w

    def _distance(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        dist = 0.0
        for k in a:
            if k not in b:
                continue
            va, vb = a[k], b[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                dist += (float(va) - float(vb)) ** 2
            else:
                dist += 0.0 if va == vb else 1.0
        return math.sqrt(dist)

    def add_observation(self, params: Dict[str, Any], score: float):
        self._observed.append(params)
        self._scores.append(score)

    def total_trials(self) -> Optional[int]:
        return self.n_trials


# ===========================================================================
#  Hyperparameter Search Runner
# ===========================================================================


class HPSearchRunner:
    """Runs hyperparameter search using a sampler and training function.

    Usage:
        def train_fn(params, trial):
            model = MyModel()
            trainer = Trainer(model, config.override(params))
            trainer.fit(train_loader, max_steps=1000)
            return trainer.evaluate(val_loader)

        runner = HPSearchRunner(space, sampler)
        results = runner.run(train_fn, experiment_tracker=tracker)
    """

    def __init__(
        self,
        space: SearchSpace,
        sampler: Union[Sampler, str] = "random",
        n_trials: int = 20,
        seed: int = 42,
        early_stopping_threshold: Optional[float] = None,
    ):
        self.space = space
        self.seed = seed

        if isinstance(sampler, str):
            if sampler == "grid":
                self.sampler = GridSampler(space)
            elif sampler == "random":
                self.sampler = RandomSampler(space, n_trials, seed)
            elif sampler == "bayesian":
                self.sampler = BayesianSampler(
                    space, n_trials, n_initial=max(5, n_trials // 5), seed=seed
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")
        else:
            self.sampler = sampler

        self.early_stopping_threshold = early_stopping_threshold
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

    def run(
        self,
        train_fn: Callable[[Dict[str, Any], int], float],
        experiment_tracker: Optional[ExperimentTracker] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Trial]:
        """Run the hyperparameter search.

        Args:
            train_fn: ``fn(params_dict, trial_index) -> float`` (lower is better).
            experiment_tracker: optional tracker for logging trials.
            progress_cb: optional ``fn(trial_index, total)`` called after each trial.

        Returns:
            List of completed ``Trial`` objects.
        """
        self.trials = []
        self.best_trial = None
        total = self.sampler.total_trials() or 9999
        trial_idx = 0

        while True:
            try:
                params = self.sampler.sample()
            except StopIteration:
                break

            trial_idx += 1
            trial = Trial(
                index=trial_idx,
                params=params,
                status="running",
                start_time=time.time(),
            )
            self.trials.append(trial)

            print(f"\n[Trial {trial_idx}/{total}] Params: {params}")

            metric = train_fn(params, trial_idx)

            trial.end_time = time.time()
            trial.result = metric
            trial.status = "completed"

            if self.best_trial is None or (
                self.best_trial.result is not None and metric < self.best_trial.result
            ):
                self.best_trial = trial
                print(f"  -> New best! metric={metric:.6f}")

            if isinstance(self.sampler, BayesianSampler):
                self.sampler.add_observation(params, metric)

            if experiment_tracker is not None:
                experiment_tracker.start_run(
                    experiment_name="hparams_search",
                    run_name=f"trial_{trial_idx}",
                    params={**params, "trial": trial_idx},
                )
                experiment_tracker.log_metrics({"metric": metric}, step=0)
                experiment_tracker.set_tag(
                    "best", "true" if trial is self.best_trial else "false"
                )
                experiment_tracker.set_status("completed")
                experiment_tracker.end_run()

            if progress_cb:
                progress_cb(trial_idx, total)

        print(f"\nHyperparameter search complete. {len(self.trials)} trials.")
        if self.best_trial:
            print(
                f"Best: {self.best_trial.params} -> metric={self.best_trial.result:.6f}"
            )
        return self.trials

    def best_params(self) -> Dict[str, Any]:
        if self.best_trial is None:
            return {}
        return self.best_trial.params

    def summary(self) -> dict:
        return {
            "total_trials": len(self.trials),
            "best_params": self.best_params(),
            "best_metric": self.best_trial.result if self.best_trial else None,
            "all_results": [
                {"index": t.index, "params": t.params, "result": t.result}
                for t in self.trials
                if t.result is not None
            ],
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, default=str)


__all__ = [
    "Param",
    "UniformFloat",
    "LogUniformFloat",
    "UniformInt",
    "Choice",
    "Fixed",
    "SearchSpace",
    "Trial",
    "Sampler",
    "GridSampler",
    "RandomSampler",
    "BayesianSampler",
    "HPSearchRunner",
]
