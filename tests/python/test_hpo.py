import json
import math

import numpy as np
import pytest

from SneppX_ALG.interface_bindings.hpo import (
    HPOConfig,
    SearchSpace,
    Study,
    Trial,
    bayesian_search,
    choice,
    grid_search,
    halving_search,
    int_,
    log_uniform,
    random_search,
    uniform,
)


class TestSearchSpace:
    def test_sample_respects_choice(self):
        space = SearchSpace({"act": choice(["relu", "tanh", "gelu"])})
        seen = {space.sample(np.random.default_rng(0))["act"] for _ in range(50)}
        assert seen <= {"relu", "tanh", "gelu"}

    def test_sample_int_within_bounds(self):
        space = SearchSpace({"n": int_(3, 7)})
        for _ in range(50):
            v = space.sample(np.random.default_rng(1))["n"]
            assert 3 <= v <= 7
            assert isinstance(v, int)

    def test_sample_uniform_within_bounds(self):
        space = SearchSpace({"lr": uniform(0.001, 0.1)})
        for _ in range(50):
            v = space.sample(np.random.default_rng(2))["lr"]
            assert 0.001 <= v <= 0.1

    def test_sample_log_uniform_within_bounds(self):
        space = SearchSpace({"lr": log_uniform(1e-4, 1e-1)})
        rng = np.random.default_rng(3)
        vals = [space.sample(rng)["lr"] for _ in range(200)]
        assert all(1e-4 <= v <= 1e-1 for v in vals)
        assert max(vals) / min(vals) > 10

    def test_grid_enumerates_product(self):
        space = SearchSpace({
            "a": choice(["x", "y"]),
            "b": int_(1, 3),
        })
        grid = space.grid()
        assert len(grid) == 2 * 3
        combos = {tuple(sorted(g.items())) for g in grid}
        assert len(combos) == 6

    def test_grid_float_steps(self):
        space = SearchSpace({"lr": uniform(0.0, 1.0)})
        grid = space.grid(float_steps=4)
        assert len(grid) == 4
        assert grid[0]["lr"] == 0.0
        assert grid[-1]["lr"] == 1.0

    def test_encode_in_unit_box(self):
        space = SearchSpace({
            "a": choice(["lo", "hi"]),
            "b": int_(0, 10),
            "c": uniform(-2.0, 2.0),
        })
        p = space.sample(np.random.default_rng(4))
        enc = space.encode(p)
        assert enc.shape == (3,)
        assert enc.min() >= 0.0 and enc.max() <= 1.0

    def test_empty_space_raises(self):
        with pytest.raises(ValueError):
            SearchSpace({})

    def test_bad_spec_raises(self):
        with pytest.raises(ValueError):
            SearchSpace({"a": {"low": 1, "high": 2}})


class TestStudyBasics:
    def test_random_search_finds_minimum(self):
        def objective(params):
            return (params["x"] - 3.0) ** 2 + (params["y"] - 1.0) ** 2

        space = {"x": uniform(-10.0, 10.0), "y": uniform(-10.0, 10.0)}
        study = random_search(space, objective, n_trials=60, seed=7)
        assert study.completed == 60
        assert study.best_value < 1.0
        assert abs(study.best_params["x"] - 3.0) < 1.0
        assert abs(study.best_params["y"] - 1.0) < 1.0

    def test_direction_maximize(self):
        def objective(params):
            return -((params["x"] - 3.0) ** 2)

        study = random_search({"x": uniform(-10.0, 10.0)}, objective,
                              n_trials=60, direction="maximize", seed=1)
        assert abs(study.best_params["x"] - 3.0) < 1.0

    def test_seeded_determinism(self):
        space = {"x": uniform(-1.0, 1.0), "c": choice(["a", "b"])}

        def objective(params):
            return params["x"] * 2.0

        s1 = random_search(space, objective, n_trials=10, seed=42)
        s2 = random_search(space, objective, n_trials=10, seed=42)
        assert s1.history() == s2.history()

    def test_grid_search_exhaustive(self):
        space = {"a": choice([1.0, 10.0]), "b": int_(0, 2)}
        results = {}

        def objective(params):
            results[tuple(params.values())] = results.get(tuple(params.values()), 0) + 1
            return params["a"] * params["b"]

        study = grid_search(space, objective, direction="maximize")
        assert study.completed == 2 * 3
        for key, count in results.items():
            assert count == 1
        assert study.best_params["a"] == 10.0
        assert study.best_params["b"] == 2

    def test_grid_exhaustion_raises(self):
        space = {"a": choice([1, 2])}
        study = Study(space, HPOConfig(sampler="grid", n_trials=10))
        with pytest.raises(RuntimeError):
            for _ in range(10):
                study.suggest()

    def test_streaming_ask_tell(self):
        study = Study({"x": uniform(0.0, 1.0)},
                      HPOConfig(n_trials=5, sampler="random", seed=0))
        for i in range(5):
            params = study.suggest()
            study.tell(params, float(params["x"]) ** 2)
        assert study.completed == 5
        assert len(study.history()) == 5
        assert study.best_value == min(study.history())

    def test_objective_dict_metadata(self):
        calls = []

        def objective(params):
            calls.append(params)
            return {"value": params["x"], "epoch": 3, "loss": 0.5}

        study = random_search({"x": uniform(0.0, 1.0)}, objective, n_trials=5, seed=0)
        assert study.trials[0].metadata["epoch"] == 3
        assert study.trials[0].metadata["loss"] == 0.5

    def test_config_validation(self):
        with pytest.raises(ValueError):
            HPOConfig(n_trials=0)
        with pytest.raises(ValueError):
            HPOConfig(sampler="genetic")
        with pytest.raises(ValueError):
            HPOConfig(direction="sideways")

    def test_best_before_any_trial_raises(self):
        study = Study({"x": uniform(0.0, 1.0)})
        with pytest.raises(RuntimeError):
            _ = study.best_trial


class TestHalving:
    def test_halving_promotes_best_config(self):
        def objective(params, budget):
            # lr=0.1 dominates regardless of budget; harder configs need budget
            if params["lr"] == 0.1:
                return 0.1 + 1.0 / budget
            return 1.0 + 1.0 / budget

        space = {"lr": choice([0.1, 0.5, 1.0])}
        study = halving_search(space, objective, n_trials=12,
                               budget_min=1, budget_max=9, eta=3, seed=0)
        assert study.best_params["lr"] == 0.1
        assert study.best_value < 0.5
        budgets = {t.metadata.get("budget") for t in study.trials}
        assert len(budgets) >= 2

    def test_halving_records_budget_metadata(self):
        def objective(params, budget):
            return {"value": float(params["x"] * budget), "acc": 0.9}

        study = halving_search({"x": uniform(0.0, 1.0)}, objective,
                               n_trials=6, budget_min=1, budget_max=4, eta=2, seed=1)
        assert all("budget" in t.metadata for t in study.trials)
        assert all(t.status == "complete" for t in study.trials)


class TestBayesian:
    def test_bayesian_near_parabola_optimum(self):
        space = {"x": uniform(-5.0, 5.0)}

        def objective(params):
            return (params["x"] - 1.5) ** 2

        study = bayesian_search(space, objective, n_trials=15, seed=3)
        assert abs(study.best_params["x"] - 1.5) < 0.5
        assert study.completed == 15
        assert study.best_value < 0.25

    def test_bayesian_improves_over_random_baseline(self):
        space = {"x": uniform(-5.0, 5.0), "y": uniform(-5.0, 5.0)}

        def objective(params):
            return (params["x"] - 2.0) ** 2 + (params["y"] + 1.0) ** 2

        rand = random_search(space, objective, n_trials=15, seed=5).best_value
        bayes = bayesian_search(space, objective, n_trials=15, seed=5).best_value
        assert bayes < rand


class TestSerialization:
    def test_trial_roundtrip(self):
        t = Trial(id=3, params={"lr": 0.01}, value=0.5, elapsed=1.2,
                  metadata={"budget": 4})
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.id == 3
        assert t2.params == {"lr": 0.01}
        assert t2.value == 0.5
        assert t2.metadata == {"budget": 4}

    def test_study_json_roundtrip(self, tmp_path):
        def objective(params):
            return params["x"] ** 2

        study = random_search({"x": uniform(0.0, 1.0)}, objective,
                              n_trials=5, seed=0)
        path = tmp_path / "study.json"
        study.to_json(str(path))
        payload = Study.from_json(str(path))
        assert payload["config"]["sampler"] == "random"
        assert len(payload["trials"]) == 5
        assert {"id", "params", "value"} <= set(payload["trials"][0])
        # raw JSON file is valid and readable
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
