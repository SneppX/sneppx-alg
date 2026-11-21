"""Tests for distributed_wrapper.py — DistributedWrapper."""

import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Linear,
    Module,
    DistributedWrapper,
)


def _make_model():
    return Linear(4, 3)


def test_distributed_wrapper_forward():
    model = _make_model()
    wrapper = DistributedWrapper(model, device="cpu")
    x = Tensor.randn((2, 4))
    out = wrapper(x)
    assert out.shape == (2, 3)
    print("  test_distributed_wrapper_forward PASS")


def test_distributed_wrapper_parameters():
    model = _make_model()
    wrapper = DistributedWrapper(model)
    params = wrapper.parameters()
    assert len(params) == 2
    named = wrapper.named_parameters()
    names = [n for n, _ in named]
    assert "weight" in names
    print("  test_distributed_wrapper_parameters PASS")


def test_distributed_wrapper_state_dict():
    model = _make_model()
    wrapper = DistributedWrapper(model)
    sd = wrapper.state_dict()
    assert "weight" in sd
    wrapper.load_state_dict(sd)
    print("  test_distributed_wrapper_state_dict PASS")


def test_distributed_wrapper_sync_gradients():
    model = _make_model()
    for p in model.parameters():
        p.requires_grad = True
    wrapper = DistributedWrapper(model)
    x = Tensor.randn((2, 4))
    out = wrapper(x)
    loss = out.sum()
    loss.backward()
    if hasattr(wrapper, "sync_gradients"):
        wrapper.sync_gradients()
    for p in wrapper.parameters():
        assert p.grad is not None
    print("  test_distributed_wrapper_sync_gradients PASS")


def test_distributed_wrapper_train_eval():
    model = _make_model()
    wrapper = DistributedWrapper(model)
    wrapper.train()
    assert wrapper.module._training
    wrapper.eval()
    assert not wrapper.module._training
    print("  test_distributed_wrapper_train_eval PASS")


def test_distributed_wrapper_to_device():
    model = _make_model()
    wrapper = DistributedWrapper(model, device="cpu")
    wrapper.to("cpu")
    for p in wrapper.parameters():
        assert p.device == "cpu"
    print("  test_distributed_wrapper_to_device PASS")


def test_distributed_wrapper_nested_module():
    model = _make_model()
    wrapper = DistributedWrapper(model)
    assert isinstance(wrapper.module, Linear)
    print("  test_distributed_wrapper_nested_module PASS")


if __name__ == "__main__":
    test_distributed_wrapper_forward()
    test_distributed_wrapper_parameters()
    test_distributed_wrapper_state_dict()
    test_distributed_wrapper_sync_gradients()
    test_distributed_wrapper_train_eval()
    test_distributed_wrapper_to_device()
    test_distributed_wrapper_nested_module()
    print("\nAll distributed_wrapper tests passed!")
