"""Tests for Automatic Mixed Precision (AMP)."""

import numpy as np
from SneppX_ALG.interface_bindings.amp import autocast, GradScaler, is_autocast_enabled, get_autocast_dtype
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_autocast_disabled_by_default():
    assert is_autocast_enabled() is False


def test_autocast_context_manager():
    with autocast(enabled=True, dtype="float16"):
        assert is_autocast_enabled() is True
        assert get_autocast_dtype() == "float16"
    assert is_autocast_enabled() is False


def test_autocast_restores_prev_state():
    with autocast(enabled=True, dtype="bfloat16"):
        with autocast(enabled=False, dtype="float16"):
            assert is_autocast_enabled() is False
        assert is_autocast_enabled() is True
        assert get_autocast_dtype() == "bfloat16"


class DummyOptimizer:
    def __init__(self):
        self.step_called = False
    def step(self):
        self.step_called = True


def test_grad_scaler_defaults():
    scaler = GradScaler()
    assert scaler.scale == 2.0 ** 16
    assert scaler.is_enabled() is True


def test_grad_scaler_disabled():
    scaler = GradScaler(enabled=False)
    assert scaler.is_enabled() is False
    assert scaler.scale_value() == 1.0


def test_grad_scaler_scale_tensor():
    scaler = GradScaler()
    t = Tensor.ones((2, 3)) * 2.0
    scaled = scaler.scale_tensor(t)
    expected = 2.0 * scaler.scale
    assert np.allclose(scaled.data, expected)


def test_grad_scaler_scale_loss():
    scaler = GradScaler(init_scale=8.0)
    loss = Tensor.from_numpy(np.array([5.0]))
    scaled = scaler.scale_loss(loss)
    assert np.allclose(scaled.data, 40.0)


def test_grad_scaler_unscale():
    scaler = GradScaler(init_scale=8.0)
    t = Tensor.ones((3,)) * 16.0
    scaler.unscale_([t])
    assert np.allclose(t.data, 2.0)


def test_grad_scaler_step_overflow_backoff():
    scaler = GradScaler(init_scale=128.0, growth_interval=1)
    opt = DummyOptimizer()
    scaler.step(opt, [Tensor.from_numpy(np.array([1.0, 2.0]))])
    assert scaler.scale == 256.0
    scaler.step(opt, [Tensor.from_numpy(np.array([np.inf, 1.0]))])
    assert scaler.scale == 128.0


def test_grad_scaler_state_dict():
    scaler = GradScaler(init_scale=64.0)
    sd = scaler.state_dict()
    assert sd["scale"] == 64.0
    scaler2 = GradScaler()
    scaler2.load_state_dict(sd)
    assert scaler2.scale == 64.0


def test_grad_scaler_growth_interval():
    scaler = GradScaler(init_scale=16.0, growth_interval=3)
    opt = DummyOptimizer()
    for _ in range(2):
        scaler.step(opt, [Tensor.ones((2,))])
    assert scaler.scale == 16.0
    scaler.step(opt, [Tensor.ones((2,))])
    assert scaler.scale == 32.0


def test_grad_scaler_update():
    scaler = GradScaler(init_scale=16.0)
    scaler.update(100.0)
    assert scaler.scale == 100.0


def test_grad_scaler_disabled_step():
    scaler = GradScaler(enabled=False)
    opt = DummyOptimizer()
    result = scaler.step(opt, [Tensor.ones((2,))])
    assert result is True
    assert opt.step_called is True


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
