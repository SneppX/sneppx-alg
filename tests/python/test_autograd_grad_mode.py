"""Regression tests for gradient-mode context restoration."""

from SneppX_ALG.interface_bindings.autograd import (
    is_grad_enabled,
    set_grad_enabled,
)


def test_set_grad_enabled_restores_previous_mode():
    previous = is_grad_enabled()

    with set_grad_enabled(not previous):
        assert is_grad_enabled() is (not previous)

    assert is_grad_enabled() is previous


def test_set_grad_enabled_restores_nested_modes():
    previous = is_grad_enabled()

    with set_grad_enabled(False):
        assert is_grad_enabled() is False
        with set_grad_enabled(True):
            assert is_grad_enabled() is True
        assert is_grad_enabled() is False

    assert is_grad_enabled() is previous


def test_set_grad_enabled_sets_mode_when_used_without_with():
    previous = is_grad_enabled()
    manager = set_grad_enabled(not previous)
    assert is_grad_enabled() is (not previous)

    manager.__exit__(None, None, None)
    assert is_grad_enabled() is previous
