"""Tests for experiment/run tracking."""

import json
import os

import pytest

from SneppX_ALG.interface_bindings.experiment import (
    Run,
    Experiment,
    ExperimentStore,
    load_experiment,
)


def test_run_logging_and_best():
    run = Run("exp", run_id="run-1")
    run.start()
    run.log_metric("loss", 1.0, step=0)
    run.log_metric("loss", 0.5, step=1)
    run.log_metric("loss", 0.25, step=2)
    run.log_metric("acc", 0.6, step=0)
    run.finish()

    assert run.status == "completed"
    assert run.started_at is not None
    assert run.ended_at is not None
    assert run.best("loss", mode="min") == 0.25
    assert run.best("acc", mode="max") == 0.6
    assert run.last("loss") == 0.25
    assert len(run.metric_history("loss")) == 3


def test_run_context_manager():
    with Run("exp", run_id="run-2", params={"lr": 0.1}) as run:
        run.log_metric("loss", 1.0, step=0)
    assert run.status == "completed"
    assert run.params["lr"] == 0.1


def test_run_fail():
    run = Run("exp", run_id="run-3")
    run.fail("oom")
    assert run.status == "failed"
    assert run.ended_at is not None


def test_run_log_artifact_missing():
    run = Run("exp", run_id="run-4")
    with pytest.raises(FileNotFoundError):
        run.log_artifact("does-not-exist.bin")


def test_run_save_load_roundtrip(tmp_path):
    run = Run("exp", run_id="run-5", params={"lr": 0.01, "opt": "adamw"})
    run.start()
    run.log_metric("loss", 0.9, step=0)
    run.log_metric("loss", 0.7, step=1)
    run.log_metric("acc", 0.5, step=0)
    run.add_tag("sweep")
    run.finish()
    run_dir = run.save(str(tmp_path))

    assert os.path.isfile(os.path.join(run_dir, "metadata.json"))
    assert os.path.isfile(os.path.join(run_dir, "metrics.jsonl"))

    with open(os.path.join(run_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta["run_id"] == "run-5"
    assert meta["params"]["lr"] == 0.01
    assert meta["tags"] == ["sweep"]

    loaded = Run.from_dir(run_dir)
    assert loaded.run_id == "run-5"
    assert loaded.params == {"lr": 0.01, "opt": "adamw"}
    assert loaded.best("loss", mode="min") == 0.7
    assert loaded.best("acc", mode="max") == 0.5
    assert loaded.status == "completed"


def test_experiment_aggregation():
    exp = Experiment("sweep")
    r1 = exp.run({"lr": 0.1})
    r1.log_metric("acc", 0.5)
    r2 = exp.run({"lr": 0.01})
    r2.log_metric("acc", 0.8)

    assert len(exp) == 2
    best = exp.best_run("acc")
    assert best.run_id == r2.run_id


def test_experiment_tracker_roundtrip(tmp_path):
    tracker = ExperimentStore(str(tmp_path))
    exp = tracker.create_experiment("cnn_sweep")
    for i, lr in enumerate([0.1, 0.01, 0.001]):
        with exp.run({"lr": lr}) as run:
            run.log_metric("acc", 0.5 + 0.1 * i)
    tracker.save_experiment(exp)

    assert tracker.list_experiments() == ["cnn_sweep"]

    loaded = tracker.load_experiment("cnn_sweep")
    assert len(loaded) == 3
    assert loaded.best_run("acc").params["lr"] == 0.001

    # load_experiment function alias
    loaded2 = load_experiment(os.path.join(str(tmp_path), "cnn_sweep"))
    assert len(loaded2) == 3


def test_metrics_jsonl_format(tmp_path):
    run = Run("exp", run_id="run-6")
    run.log_metric("loss", 1.0, step=0)
    run.log_metric("loss", 0.5, step=1, split="val")
    run_dir = run.save(str(tmp_path))
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        lines = [l for l in f.read().strip().splitlines() if l]
    assert len(lines) == 2
    assert json.loads(lines[1])["split"] == "val"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
