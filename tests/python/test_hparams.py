"""Tests for hparams.py — hyperparameter search."""

import math
import tempfile
import pytest
from SneppX_ALG.interface_bindings.hparams import (
    UniformFloat,
    LogUniformFloat,
    UniformInt,
    Choice,
    Fixed,
    SearchSpace,
    Trial,
    GridSampler,
    RandomSampler,
    BayesianSampler,
    HPSearchRunner,
)


def test_uniform_float():
    p = UniformFloat(0.0, 1.0)
    v = p.sample()
    assert 0.0 <= v <= 1.0
    g = p.grid(5)
    assert len(g) == 5
    assert g[0] == 0.0
    assert g[-1] == 1.0
    print("  test_uniform_float PASS")


def test_log_uniform_float():
    p = LogUniformFloat(1e-5, 1e-1)
    v = p.sample()
    assert 1e-5 <= v <= 1e-1
    g = p.grid(3)
    assert len(g) == 3
    assert abs(g[0] - 1e-5) < 1e-6
    assert abs(g[-1] - 1e-1) < 1e-6
    print("  test_log_uniform_float PASS")


def test_uniform_int():
    p = UniformInt(1, 10)
    for _ in range(50):
        v = p.sample()
        assert 1 <= v <= 10
    g = p.grid()
    assert len(g) == 10
    assert g[0] == 1
    assert g[-1] == 10
    print("  test_uniform_int PASS")


def test_choice():
    p = Choice(["adamw", "sgd", "lion"])
    vals = set()
    for _ in range(50):
        v = p.sample()
        assert v in ["adamw", "sgd", "lion"]
        vals.add(v)
    assert len(vals) == 3
    g = p.grid()
    assert g == ["adamw", "sgd", "lion"]
    print("  test_choice PASS")


def test_fixed():
    p = Fixed(42)
    assert p.sample() == 42
    assert p.grid() == [42]
    print("  test_fixed PASS")


def test_search_space():
    space = SearchSpace()
    space.add("lr", LogUniformFloat(1e-4, 1e-1))
    space.add("batch_size", Choice([16, 32]))
    space.add("dropout", UniformFloat(0.0, 0.5))

    s = space.sample()
    assert "lr" in s
    assert "batch_size" in s
    assert "dropout" in s
    assert s["batch_size"] in [16, 32]

    g = space.grid()
    assert len(g) == 10 * 2 * 10  # depends on grid sizes
    assert "lr" in g[0]
    print("  test_search_space PASS")


def test_grid_sampler():
    space = SearchSpace()
    space.add("lr", Choice([0.1, 0.01]))
    space.add("opt", Choice(["adamw", "sgd"]))

    sampler = GridSampler(space)
    assert sampler.total_trials() == 4

    trials = []
    while True:
        try:
            trials.append(sampler.sample())
        except StopIteration:
            break
    assert len(trials) == 4
    assert {"lr": 0.1, "opt": "adamw"} in trials
    assert {"lr": 0.01, "opt": "sgd"} in trials
    print("  test_grid_sampler PASS")


def test_random_sampler():
    space = SearchSpace()
    space.add("lr", LogUniformFloat(1e-4, 1e-1))
    space.add("opt", Choice(["adamw", "sgd"]))

    sampler = RandomSampler(space, n_trials=10)
    assert sampler.total_trials() == 10

    trials = []
    while True:
        try:
            trials.append(sampler.sample())
        except StopIteration:
            break
    assert len(trials) == 10
    print("  test_random_sampler PASS")


def test_bayesian_sampler():
    space = SearchSpace()
    space.add("x", UniformFloat(-5, 5))

    sampler = BayesianSampler(space, n_trials=15, n_initial=3, seed=42)

    for i in range(15):
        params = sampler.sample()
        # Objective: -(x-2)^2  → max at x=2
        score = -((params["x"] - 2) ** 2)
        sampler.add_observation(params, score)
    assert sampler._count == 15
    print("  test_bayesian_sampler PASS")


def test_hp_search_runner():
    space = SearchSpace()
    space.add("lr", Choice([0.1, 0.01]))
    space.add("opt", Choice(["adamw", "sgd"]))

    def train_fn(params, trial_idx):
        return params["lr"] * (0.9 if params["opt"] == "adamw" else 1.0)

    runner = HPSearchRunner(space, sampler="grid")
    results = runner.run(train_fn)
    assert len(results) == 4
    assert runner.best_trial is not None
    assert "lr" in runner.best_params()
    s = runner.summary()
    assert s["total_trials"] == 4
    print("  test_hp_search_runner PASS")


def test_hp_search_runner_random():
    space = SearchSpace()
    space.add("lr", UniformFloat(0.001, 0.1))

    def train_fn(params, trial_idx):
        return params["lr"]

    runner = HPSearchRunner(space, sampler="random", n_trials=5)
    results = runner.run(train_fn)
    assert len(results) == 5
    print("  test_hp_search_runner_random PASS")


def test_hp_search_runner_bayesian():
    space = SearchSpace()
    space.add("x", UniformFloat(-3, 3))

    def train_fn(params, trial_idx):
        return (params["x"] - 1.5) ** 2

    runner = HPSearchRunner(space, sampler="bayesian", n_trials=10)
    results = runner.run(train_fn)
    assert len(results) == 10
    assert runner.best_trial is not None
    print(f"  Best x={runner.best_params()}, metric={runner.best_trial.result:.4f}")
    print("  test_hp_search_runner_bayesian PASS")


def test_hp_search_runner_with_tracker():
    space = SearchSpace()
    space.add("lr", Choice([0.1, 0.01]))

    def train_fn(params, trial_idx):
        return params["lr"]

    with tempfile.TemporaryDirectory() as tmp:
        from SneppX_ALG.interface_bindings.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(storage_dir=tmp)
        runner = HPSearchRunner(space, sampler="grid")
        runner.run(train_fn, experiment_tracker=tracker)
        runs = tracker.list_runs("hparams_search")
        assert len(runs) == 2
    print("  test_hp_search_runner_with_tracker PASS")


def test_trial_dataclass():
    t = Trial(index=1, params={"lr": 0.01}, status="running")
    assert t.index == 1
    assert t.params["lr"] == 0.01
    t.result = 0.5
    t.status = "completed"
    assert t.status == "completed"
    print("  test_trial_dataclass PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
