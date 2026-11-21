"""Tests for experiment_tracker.py — experiment tracking."""

import os
import json
import tempfile
import pytest
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    Module,
    AdamW,
    ExperimentRun,
    LocalBackend,
    ExperimentTracker,
    CompositeTracker,
    LogTracker,
    Tracker,
)


def test_experiment_run_defaults():
    run = ExperimentRun(run_id="test-1")
    assert run.run_id == "test-1"
    assert run.status == "running"
    assert run.params == {}
    assert run.metrics == {}
    print("  test_experiment_run_defaults PASS")


def test_experiment_run_log_metric():
    run = ExperimentRun(run_id="test-2")
    run.log_metric("loss", 0.5, step=10)
    run.log_metric("loss", 0.4, step=20)
    assert "loss" in run.metrics
    assert len(run.metrics["loss"]) == 2
    assert run.metrics["loss"][0]["step"] == 10
    assert run.metrics["loss"][0]["value"] == 0.5
    print("  test_experiment_run_log_metric PASS")


def test_experiment_run_log_metrics_bulk():
    run = ExperimentRun(run_id="test-3")
    run.log_metrics({"loss": 0.5, "acc": 0.8}, step=1)
    assert "loss" in run.metrics
    assert "acc" in run.metrics
    print("  test_experiment_run_log_metrics_bulk PASS")


def test_experiment_run_params():
    run = ExperimentRun(run_id="test-4")
    run.log_param("lr", 0.001)
    run.log_params({"wd": 0.01, "batch_size": 32})
    assert run.params["lr"] == 0.001
    assert run.params["wd"] == 0.01
    print("  test_experiment_run_params PASS")


def test_experiment_run_artifacts():
    run = ExperimentRun(run_id="test-5")
    run.log_artifact("/tmp/checkpoint.pt")
    assert "/tmp/checkpoint.pt" in run.artifacts
    print("  test_experiment_run_artifacts PASS")


def test_local_backend_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalBackend(tmp)
        run = ExperimentRun(
            run_id="test-6",
            experiment_name="exp1",
            run_name="test",
            params={"lr": 0.001},
        )
        run.log_metric("loss", 0.5, step=1)
        backend.save(run)

        loaded = backend.load("test-6")
        assert loaded is not None
        assert loaded.run_id == "test-6"
        assert loaded.params["lr"] == 0.001
        assert len(loaded.metrics["loss"]) == 1
    print("  test_local_backend_save_load PASS")


def test_local_backend_list():
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalBackend(tmp)
        for i in range(3):
            run = ExperimentRun(
                run_id=f"run-{i}", experiment_name="exp1", run_name=f"run-{i}"
            )
            backend.save(run)
        runs = backend.list_runs("exp1")
        assert len(runs) == 3
        assert len(backend.list_runs("other")) == 0
    print("  test_local_backend_list PASS")


def test_local_backend_delete():
    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalBackend(tmp)
        backend.save(ExperimentRun(run_id="delete-me"))
        assert backend.load("delete-me") is not None
        backend.delete_run("delete-me")
        assert backend.load("delete-me") is None
    print("  test_local_backend_delete PASS")


def test_tracker_noop():
    tracker = Tracker()
    tracker.start_run()
    tracker.log_metric("x", 1.0)
    tracker.log_metrics({"a": 1.0})
    tracker.log_param("p", 1)
    tracker.log_params({"q": 2})
    tracker.log_artifact("f.txt")
    tracker.set_tag("t", "v")
    tracker.set_status("done")
    tracker.end_run()
    assert tracker.run is None
    print("  test_tracker_noop PASS")


def test_tracker_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = ExperimentTracker(storage_dir=tmp)
        tracker.start_run("test_experiment", params={"lr": 0.001})
        tracker.log_metrics({"loss": 0.5, "acc": 0.8}, step=1)
        tracker.log_metrics({"loss": 0.4, "acc": 0.85}, step=2)
        tracker.log_param("batch_size", 32)
        tracker.set_tag("env", "test")
        tracker.set_status("completed")
        tracker.end_run()

        runs = tracker.list_runs("test_experiment")
        assert len(runs) == 1
        assert runs[0].params["lr"] == 0.001
        assert len(runs[0].metrics["loss"]) == 2
    print("  test_tracker_smoke PASS")


def test_composite_tracker():
    with tempfile.TemporaryDirectory() as tmp:
        t1 = ExperimentTracker(storage_dir=tmp)
        t2 = LogTracker()
        comp = CompositeTracker([t1, t2])
        comp.start_run(experiment_name="composite_test", params={"lr": 0.01})
        comp.log_metrics({"loss": 0.5}, step=1)
        comp.end_run()

        runs = t1.list_runs("composite_test")
        assert len(runs) == 1
        assert runs[0].params["lr"] == 0.01
    print("  test_composite_tracker PASS")


def test_run_to_dict_roundtrip():
    run = ExperimentRun(run_id="rt", experiment_name="exp")
    run.log_metrics({"m": 1.0}, step=0)
    d = run.to_dict()
    restored = ExperimentRun.from_dict(d)
    assert restored.run_id == "rt"
    assert restored.metrics["m"][0]["value"] == 1.0
    print("  test_run_to_dict_roundtrip PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
