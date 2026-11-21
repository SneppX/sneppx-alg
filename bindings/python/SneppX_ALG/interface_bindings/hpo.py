"""Hyperparameter-search orchestrator — HPO over a config space.

Drives a user ``objective`` over a ``SearchSpace`` with pluggable samplers:
random search, exhaustive grid search, successive halving (early-stopped
budget scheduling), and Bayesian optimization via GP-UCB. The ``Study``
object offers both a streaming ``suggest``/``tell`` API (Optuna-style) and a
one-shot ``run`` driver; trials and results serialize to JSON.

Typical usage::

    space = {
        "lr": log_uniform(1e-4, 1e-1),
        "hidden": choice([32, 64, 128]),
        "epochs": int_(1, 10),
    }
    study = bayesian_search(space, train_and_eval, n_trials=20)
    print(study.best_params, study.best_value)
"""

import itertools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "Trial",
    "SearchSpace",
    "HPOConfig",
    "Study",
    "choice",
    "int_",
    "uniform",
    "log_uniform",
    "random_search",
    "grid_search",
    "halving_search",
    "bayesian_search",
]


# --------------------------------------------------------------------------
# parameter factories
# --------------------------------------------------------------------------

def choice(values: List[Any]) -> Dict[str, Any]:
    return {"type": "choice", "values": list(values)}


def int_(low: int, high: int) -> Dict[str, Any]:
    return {"type": "int", "low": int(low), "high": int(high)}


def uniform(low: float, high: float) -> Dict[str, Any]:
    return {"type": "float", "low": float(low), "high": float(high), "log": False}


def log_uniform(low: float, high: float) -> Dict[str, Any]:
    return {"type": "float", "low": float(low), "high": float(high), "log": True}


# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------

class SearchSpace:
    """A dict of parameter definitions with sampling / grid / encoding."""

    def __init__(self, space: Dict[str, Dict[str, Any]]):
        if not space:
            raise ValueError("search space must define at least one parameter")
        for name, spec in space.items():
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(
                    f"parameter {name!r} must be a spec dict from "
                    "choice/int_/uniform/log_uniform")
        self.space = dict(space)

    # ---- sampling --------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, spec in self.space.items():
            kind = spec["type"]
            if kind == "choice":
                vals = spec["values"]
                out[k] = vals[int(rng.integers(len(vals)))]
            elif kind == "int":
                out[k] = int(rng.integers(spec["low"], spec["high"] + 1))
            elif kind == "float":
                lo, hi = spec["low"], spec["high"]
                if spec.get("log"):
                    out[k] = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
                else:
                    out[k] = float(rng.uniform(lo, hi))
            else:
                raise ValueError(f"unknown parameter type {kind!r}")
        return out

    # ---- grid ------------------------------------------------------------
    def grid(self, float_steps: int = 5) -> List[Dict[str, Any]]:
        keys: List[str] = []
        axes: List[List[Any]] = []
        for k, spec in self.space.items():
            kind = spec["type"]
            if kind == "choice":
                vals = list(spec["values"])
            elif kind == "int":
                vals = list(range(spec["low"], spec["high"] + 1))
            elif kind == "float":
                lo, hi = spec["low"], spec["high"]
                n = max(1, int(spec.get("steps", float_steps)))
                if spec.get("log"):
                    vals = [float(np.exp(math.log(lo) + (math.log(hi) - math.log(lo))
                                         * i / max(1, n - 1))) for i in range(n)]
                else:
                    vals = [float(lo + (hi - lo) * i / max(1, n - 1)) for i in range(n)]
            else:
                raise ValueError(f"unknown parameter type {kind!r}")
            keys.append(k)
            axes.append(vals)
        return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]

    # ---- numeric encoding (for Bayesian surrogate) -----------------------
    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        row = np.empty(len(self.space))
        for i, (k, spec) in enumerate(self.space.items()):
            v = params[k]
            kind = spec["type"]
            if kind == "choice":
                row[i] = spec["values"].index(v) / max(1, len(spec["values"]) - 1)
            elif kind == "int":
                row[i] = (v - spec["low"]) / max(1, spec["high"] - spec["low"])
            elif kind == "float":
                lo, hi = spec["low"], spec["high"]
                if spec.get("log"):
                    t = (math.log(v) - math.log(lo)) / max(1e-9, math.log(hi) - math.log(lo))
                else:
                    t = (v - lo) / max(1e-9, hi - lo)
                row[i] = float(np.clip(t, 0.0, 1.0))
        return row

    def __len__(self) -> int:
        return len(self.space)

    def __repr__(self) -> str:
        return f"SearchSpace({list(self.space)})"


# --------------------------------------------------------------------------
# trial + config
# --------------------------------------------------------------------------

@dataclass
class Trial:
    """A single objective evaluation."""

    id: int
    params: Dict[str, Any]
    value: Optional[float] = None
    elapsed: float = 0.0
    status: str = "pending"   # pending | complete | failed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "params": self.params,
            "value": self.value,
            "elapsed": self.elapsed,
            "status": self.status,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trial":
        return Trial(
            id=d["id"],
            params=d["params"],
            value=d.get("value"),
            elapsed=d.get("elapsed", 0.0),
            status=d.get("status", "complete"),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Trial({self.id}, value={self.value})"


@dataclass
class HPOConfig:
    """Search configuration for a Study."""

    n_trials: int = 20
    sampler: str = "random"        # random | grid | halving | bayes
    direction: str = "minimize"    # minimize | maximize
    seed: int = 0
    budget_min: int = 1
    budget_max: int = 10
    eta: int = 3
    kappa: float = 2.5             # GP-UCB exploration constant
    initial_random: int = 4        # bayes warm-up trials
    float_steps: int = 5           # grid resolution for float params

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.sampler not in ("random", "grid", "halving", "bayes"):
            raise ValueError(
                f"unknown sampler {self.sampler!r}; "
                "expected random|grid|halving|bayes")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if self.eta < 2:
            raise ValueError("eta must be >= 2")


# --------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------

class _BaseSampler:
    name = "base"

    def __init__(self, study: "Study"):
        self.study = study

    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def observe(self, params: Dict[str, Any], value: float) -> None:
        pass


class _RandomSampler(_BaseSampler):
    name = "random"

    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        return self.study.space.sample(self.study._rng)


class _GridSampler(_BaseSampler):
    name = "grid"

    def __init__(self, study: "Study"):
        super().__init__(study)
        self._candidates = study.space.grid(study.config.float_steps)
        self._index = 0

    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        if self._index >= len(self._candidates):
            raise RuntimeError("grid search exhausted; raise n_trials")
        params = self._candidates[self._index]
        self._index += 1
        return params


class _HalvingSampler(_BaseSampler):
    """Successive halving: feeds the Study run loop a fixed pool; the
    objective is evaluated per budget round with top 1/eta promoted."""

    name = "halving"

    def __init__(self, study: "Study"):
        super().__init__(study)
        self._pool: List[Dict[str, Any]] = []

    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        if not self._pool:
            self._pool = [self.study.space.sample(self.study._rng)
                          for _ in range(self.study.config.n_trials)]
        return self._pool.pop()


class _BayesSampler(_BaseSampler):
    """GP-UCB Bayesian optimizer over the encoded search space."""

    name = "bayes"

    def __init__(self, study: "Study"):
        super().__init__(study)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        if len(self._X) < self.study.config.initial_random:
            return self.study.space.sample(self.study._rng)
        return self._acquire()

    def observe(self, params: Dict[str, Any], value: float) -> None:
        self._X.append(self.study.space.encode(params))
        self._y.append(float(value))

    def _acquire(self) -> Dict[str, Any]:
        X = np.asarray(self._X, dtype=np.float64).reshape(len(self._X), -1)
        y = np.asarray(self._y, dtype=np.float64)
        if self.study.config.direction == "minimize":
            y = -y
        if y.std() > 1e-9:
            y = (y - y.mean()) / y.std()
        candidates = [self.study.space.sample(self.study._rng)
                      for _ in range(64)]
        Xc = np.asarray([self.study.space.encode(p) for p in candidates])
        mu, sd = _gp_predict(X, y, Xc)
        score = mu + self.study.config.kappa * sd
        return candidates[int(np.argmax(score))]


def _rbf(X: np.ndarray, Y: np.ndarray, sigma_f: float,
         length_scale: float) -> np.ndarray:
    Xs = np.sum(X ** 2, axis=1, keepdims=True)
    Ys = np.sum(Y ** 2, axis=1, keepdims=True)
    d2 = Xs + Ys.T - 2.0 * (X @ Y.T)
    return sigma_f ** 2 * np.exp(-0.5 * d2 / (length_scale ** 2))


def _gp_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray,
                sigma_f: float = 1.0, length_scale: float = 1.0,
                sigma_n: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    K = _rbf(X, X, sigma_f, length_scale) + sigma_n ** 2 * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = _rbf(X, Xstar, sigma_f, length_scale)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.diag(_rbf(Xstar, Xstar, sigma_f, length_scale)) - np.einsum(
        "ij,ij->j", v, v)
    return mu, np.sqrt(np.maximum(var, 0.0))


# --------------------------------------------------------------------------
# study
# --------------------------------------------------------------------------

class Study:
    """Hyperparameter-search orchestrator with ask/tell and run drivers."""

    def __init__(self, space: Union[SearchSpace, Dict[str, Dict[str, Any]]],
                 config: Optional[HPOConfig] = None):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.config = config or HPOConfig()
        self.trials: List[Trial] = []
        self._rng = np.random.default_rng(self.config.seed)
        self._best_trial: Optional[Trial] = None
        self._sampler = self._make_sampler()

    def _make_sampler(self) -> _BaseSampler:
        if self.config.sampler == "random":
            return _RandomSampler(self)
        if self.config.sampler == "grid":
            return _GridSampler(self)
        if self.config.sampler == "halving":
            return _HalvingSampler(self)
        return _BayesSampler(self)

    # ---- ask / tell ------------------------------------------------------
    def suggest(self, budget: Optional[int] = None) -> Dict[str, Any]:
        return self._sampler.suggest(budget)

    def tell(self, params: Dict[str, Any], value: float, elapsed: float = 0.0,
             metadata: Optional[Dict[str, Any]] = None,
             status: str = "complete") -> Trial:
        trial = Trial(
            id=len(self.trials),
            params=dict(params),
            value=float(value),
            elapsed=float(elapsed),
            status=status,
            metadata=dict(metadata or {}),
        )
        self.trials.append(trial)
        self._sampler.observe(params, float(value))
        self._maybe_update_best(trial)
        return trial

    # ---- run -------------------------------------------------------------
    def run(self, objective: Callable[..., Any]) -> "Study":
        if self.config.sampler == "halving":
            self._run_halving(objective)
            return self
        budget = None
        if self.config.sampler == "grid":
            candidates = self.space.grid(self.config.float_steps)
            total = min(self.config.n_trials, len(candidates))
            self._sampler._candidates = candidates
            self._sampler._index = 0
        else:
            total = self.config.n_trials
        for _ in range(total):
            params = self.suggest(budget)
            res = objective(params, budget=budget) if budget is not None else objective(params)
            self._absorb(res, params)
        return self

    def _run_halving(self, objective: Callable[..., Any]) -> None:
        cfg = self.config
        budget = cfg.budget_min
        pool = [self.space.sample(self._rng) for _ in range(cfg.n_trials)]
        rounds = 0
        while budget <= cfg.budget_max and pool:
            results: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for params in pool:
                res = objective(params, budget=budget)
                value, meta = self._split_result(res)
                self.tell(params, value, metadata={**meta, "budget": budget})
                results.append((value, params, meta))
            results.sort(key=lambda t: t[0],
                         reverse=(cfg.direction == "maximize"))
            keep = max(1, len(results) // cfg.eta)
            pool = [r[1] for r in results[:keep]]
            budget = min(budget * cfg.eta, cfg.budget_max)
            rounds += 1
            if rounds > 32:
                break

    def _absorb(self, res: Any, params: Dict[str, Any]) -> None:
        value, meta = self._split_result(res)
        self.tell(params, value, metadata=meta)

    @staticmethod
    def _split_result(res: Any) -> Tuple[float, Dict[str, Any]]:
        if isinstance(res, dict):
            if "value" not in res:
                raise ValueError("objective dict results must include 'value'")
            return float(res["value"]), {k: v for k, v in res.items() if k != "value"}
        if isinstance(res, (int, float)):
            return float(res), {}
        raise ValueError("objective must return a number or {'value': ...} dict")

    def _maybe_update_best(self, trial: Trial) -> None:
        if trial.value is None or trial.status != "complete":
            return
        if self._best_trial is None:
            self._best_trial = trial
            return
        cur, cand = self._best_trial.value, trial.value
        if self.config.direction == "minimize":
            if cand < cur:
                self._best_trial = trial
        else:
            if cand > cur:
                self._best_trial = trial

    # ---- results ---------------------------------------------------------
    @property
    def best_trial(self) -> Trial:
        if self._best_trial is None:
            raise RuntimeError("no completed trials yet")
        return self._best_trial

    @property
    def best_params(self) -> Dict[str, Any]:
        return dict(self.best_trial.params)

    @property
    def best_value(self) -> float:
        return float(self.best_trial.value)

    @property
    def completed(self) -> int:
        return sum(1 for t in self.trials if t.status == "complete")

    def results(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.trials]

    def history(self) -> List[float]:
        return [float(t.value) for t in self.trials if t.value is not None]

    def to_json(self, path: str) -> None:
        payload = {
            "space": self.space.space,
            "config": {
                "n_trials": self.config.n_trials,
                "sampler": self.config.sampler,
                "direction": self.config.direction,
                "seed": self.config.seed,
            },
            "trials": self.results(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __repr__(self) -> str:
        return (f"Study({self.config.sampler}, {self.completed} trials, "
                f"best={self._best_trial.value if self._best_trial else None})")


# --------------------------------------------------------------------------
# convenience drivers
# --------------------------------------------------------------------------

def random_search(space: Union[SearchSpace, Dict[str, Dict[str, Any]]],
                  objective: Callable[..., Any], n_trials: int = 20,
                  direction: str = "minimize", seed: int = 0) -> Study:
    return Study(space, HPOConfig(n_trials=n_trials, sampler="random",
                                  direction=direction, seed=seed)).run(objective)


def grid_search(space: Union[SearchSpace, Dict[str, Dict[str, Any]]],
                objective: Callable[..., Any], n_trials: Optional[int] = None,
                direction: str = "minimize", float_steps: int = 5) -> Study:
    cfg = HPOConfig(sampler="grid", direction=direction, float_steps=float_steps)
    if n_trials is not None:
        cfg.n_trials = n_trials
    return Study(space, cfg).run(objective)


def halving_search(space: Union[SearchSpace, Dict[str, Dict[str, Any]]],
                   objective: Callable[..., Any], n_trials: int = 20,
                   direction: str = "minimize", budget_min: int = 1,
                   budget_max: int = 10, eta: int = 3, seed: int = 0) -> Study:
    cfg = HPOConfig(n_trials=n_trials, sampler="halving", direction=direction,
                    budget_min=budget_min, budget_max=budget_max, eta=eta,
                    seed=seed)
    return Study(space, cfg).run(objective)


def bayesian_search(space: Union[SearchSpace, Dict[str, Dict[str, Any]]],
                    objective: Callable[..., Any], n_trials: int = 20,
                    direction: str = "minimize", seed: int = 0,
                    kappa: float = 2.5) -> Study:
    cfg = HPOConfig(n_trials=n_trials, sampler="bayes", direction=direction,
                    seed=seed, kappa=kappa)
    return Study(space, cfg).run(objective)
