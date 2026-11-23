"""Tests for checkpoint_manager.py — CheckpointManager."""

import os
import sys
import tempfile

import numpy as np
import pytest

from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    Module,
    AdamW,
    TrainConfigV3,
    TrainerV3,
    CheckpointManager,
)
from SneppX_ALG.interface_bindings.amp import GradScaler


def _make_model():
    return Linear(4, 3)


def _make_trainer(ckpt_dir: str) -> TrainerV3:
    model = _make_model()
    cfg = TrainConfigV3(
        {
            "training": {
                "device": "cpu",
                "max_steps": 10,
                "log_every": 1000,
                "eval_every": 0,
                "save_every": 0,
                "checkpoint_dir": ckpt_dir,
            },
            "optimizer": {"name": "adamw", "lr": 0.001},
            "profiler": {"enabled": False},
            "checkpoint": {"keep_last_n": 3},
        }
    )
    return TrainerV3(model, cfg)


def test_checkpoint_manager_init():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp, keep_last=3, keep_best=2)
        assert os.path.isdir(tmp)
        s = mgr.summary()
        assert s["checkpoint_dir"] == tmp
        assert s["total_checkpoints"] == 0
        assert s["best_checkpoints"] == 0
    print("  test_checkpoint_manager_init PASS")


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        mgr = trainer.checkpoint_manager
        mgr.async_save = False
        original_w = trainer.model.weight.data.copy()

        mgr.save(trainer)
        trainer.model.weight.data = np.zeros_like(original_w)

        loaded = mgr.load_latest(trainer)
        assert loaded
        assert np.allclose(trainer.model.weight.data, original_w)
    print("  test_checkpoint_save_and_load PASS")


def test_checkpoint_scaler_state_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        trainer.scaler = GradScaler(
            init_scale=64.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=7
        )
        mgr = trainer.checkpoint_manager
        state = mgr._build_trainer_state(trainer)
        assert state["scaler_state"]["scale"] == 64.0

        # Mutate the live scaler, then restore from the captured state.
        trainer.scaler.update(2.0)
        trainer.scaler._growth_factor = 99.0
        trainer.scaler._backoff_factor = 99.0
        trainer.scaler._growth_interval = 99
        mgr._restore_trainer_state(trainer, state)

        assert trainer.scaler.scale == 64.0
        assert trainer.scaler._growth_factor == 3.0
        assert trainer.scaler._backoff_factor == 0.25
        assert trainer.scaler._growth_interval == 7
    print("  test_checkpoint_scaler_state_round_trip PASS")


def test_checkpoint_best_retention():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp, keep_last=10, keep_best=2, async_save=False)
        trainer = _make_trainer(tmp)

        for step, metric in [(0, 1.0), (1, 0.8), (2, 0.5), (3, 0.3)]:
            trainer._step = step
            mgr.save(trainer, metric=metric, is_best=True)

        best_files = mgr._best_checkpoints()
        assert len(best_files) <= 2
    print("  test_checkpoint_best_retention PASS")


def test_checkpoint_last_pruning():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp, keep_last=3, async_save=False)
        trainer = _make_trainer(tmp)

        for step in range(5):
            trainer._step = step
            mgr.save(trainer)

        ckpts = mgr._all_checkpoints()
        assert len(ckpts) <= 3
        basenames = {os.path.basename(p) for p in ckpts}
        assert "checkpoint_0.ckpt" not in basenames
        assert "checkpoint_1.ckpt" not in basenames
    print("  test_checkpoint_last_pruning PASS")


def test_checkpoint_rng_restore():
    with tempfile.TemporaryDirectory() as tmp:
        import random as _random

        trainer = _make_trainer(tmp)
        mgr = trainer.checkpoint_manager
        mgr.async_save = False

        _random.seed(42)
        saved = _random.getstate()
        seq_a = [_random.randint(0, 1000000) for _ in range(3)]

        _random.setstate(saved)
        mgr.save(trainer)

        _random.seed(99)
        _random.random()

        loaded = mgr.load_latest(trainer)
        assert loaded
        seq_b = [_random.randint(0, 1000000) for _ in range(3)]
        assert seq_a == seq_b
    print("  test_checkpoint_rng_restore PASS")


def test_checkpoint_binary_format():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(
            tmp, keep_last=3, save_format="binary", async_save=False, allow_unsafe=True
        )
        trainer = _make_trainer(tmp)
        original_w = trainer.model.weight.data.copy()

        mgr.save(trainer)
        trainer.model.weight.data = np.zeros_like(original_w)

        loaded = mgr.load_latest(trainer)
        assert loaded
        assert np.allclose(trainer.model.weight.data, original_w)
    print("  test_checkpoint_binary_format PASS")


def test_checkpoint_load_best():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(
            tmp, keep_last=5, keep_best=2, async_save=False, allow_unsafe=True
        )
        trainer = _make_trainer(tmp)
        orig = trainer.model.weight.data.copy()

        trainer._step = 0
        mgr.save(trainer, metric=1.0, is_best=False)
        trainer._step = 1
        mgr.save(trainer, metric=0.5, is_best=True)

        trainer.model.weight.data = np.zeros_like(orig)
        loaded = mgr.load_best(trainer)
        assert loaded
        assert np.allclose(trainer.model.weight.data, orig)
    print("  test_checkpoint_load_best PASS")


def test_pickle_checkpoint_requires_explicit_opt_in(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        mgr = trainer.checkpoint_manager
        mgr.async_save = False
        mgr.save(trainer)
        path = mgr._latest_path()

        strict = CheckpointManager(tmp, allow_unsafe=False)
        assert strict.load(trainer, path) is False
        assert "Refusing to deserialize" in capsys.readouterr().out


def test_binary_checkpoint_requires_explicit_opt_in(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        mgr = CheckpointManager(
            tmp, save_format="binary", async_save=False, allow_unsafe=True
        )
        mgr.save(trainer)
        path = mgr._latest_path()

        strict = CheckpointManager(tmp, save_format="binary", allow_unsafe=False)
        assert strict.load(trainer, path) is False
        assert "Refusing to deserialize" in capsys.readouterr().out


def test_explicit_opt_in_warns_and_loads():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(tmp)
        mgr = trainer.checkpoint_manager
        mgr.async_save = False
        original_w = trainer.model.weight.data.copy()
        mgr.save(trainer)
        trainer.model.weight.data = np.zeros_like(original_w)
        path = mgr._latest_path()

        permissive = CheckpointManager(tmp, allow_unsafe=True)
        with pytest.warns(RuntimeWarning):
            loaded = permissive.load(trainer, path)
        assert loaded
        assert np.allclose(trainer.model.weight.data, original_w)


if __name__ == "__main__":
    test_checkpoint_manager_init()
    test_checkpoint_save_and_load()
    test_checkpoint_best_retention()
    test_checkpoint_last_pruning()
    test_checkpoint_rng_restore()
    test_checkpoint_binary_format()
    test_checkpoint_load_best()
    print("\nAll checkpoint_manager tests passed!")
