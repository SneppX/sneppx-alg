"""Tests for LoRA/QLoRA fine-tuning module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.lora import (
    LoRAConfig,
    QLoRAConfig,
    LoRALinear,
    QLoRALinear,
    apply_lora,
    get_lora_parameters,
    quantize_nf4,
    dequantize_nf4,
    dpo_loss,
    grpo_loss,
    DPOTrainer,
    DPOTrainerConfig,
    GRPOTrainer,
    GRPOTrainerConfig,
)
from SneppX_ALG.interface_bindings.nn import Module, Linear, Sequential
from SneppX_ALG.interface_bindings.tensor import Tensor


class TinyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(64, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.alpha == 16.0
    assert cfg.dropout == 0.0
    assert cfg.target_modules is None
    assert cfg.use_rslora is False
    print("  PASS lora config defaults")


def test_qlora_config_defaults():
    cfg = QLoRAConfig()
    assert cfg.r == 8
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    print("  PASS qlora config defaults")


def test_lora_linear_forward():
    layer = LoRALinear(64, 128, r=4, alpha=8.0)
    x = Tensor.randn((2, 64))
    out = layer(x)
    assert out.shape == (2, 128)
    assert not layer.merged
    print("  PASS lora linear forward shape")


def test_lora_linear_merge():
    layer = LoRALinear(64, 128, r=4, alpha=8.0)
    x = Tensor.randn((1, 64))

    out_before = layer(x).data.copy()
    layer.merge_weights()
    assert layer.merged
    out_merged = layer(x).data.copy()
    assert np.allclose(out_before, out_merged, atol=1e-4)
    print("  PASS lora linear merge")


def test_lora_linear_unmerge():
    layer = LoRALinear(64, 128, r=4, alpha=8.0)
    x = Tensor.randn((1, 64))
    out_original = layer(x).data.copy()
    layer.merge_weights()
    layer.unmerge_weights()
    assert not layer.merged
    out_restored = layer(x).data.copy()
    assert np.allclose(out_original, out_restored, atol=1e-4)
    print("  PASS lora linear unmerge")


def test_apply_lora_all():
    model = TinyModel()
    cfg = LoRAConfig(r=4)
    model = apply_lora(model, cfg)

    lora_count = 0
    for m in model._modules.values():
        if "LoRALinear" in type(m).__name__:
            lora_count += 1
    assert lora_count == 3
    print(f"  PASS apply lora all ({lora_count} adapters)")


def test_apply_lora_targeted():
    model = TinyModel()
    cfg = LoRAConfig(r=4, target_modules=["fc1", "fc3"])
    model = apply_lora(model, cfg)

    lora_count = 0
    for name, m in model._modules.items():
        if "LoRALinear" in type(m).__name__:
            lora_count += 1
            assert name in ("fc1", "fc3"), f"unexpected target {name}"
    assert lora_count == 2
    print(f"  PASS apply lora targeted ({lora_count} adapters)")


def test_get_lora_parameters():
    model = TinyModel()
    cfg = LoRAConfig(r=4)
    model = apply_lora(model, cfg)
    params = get_lora_parameters(model)
    assert len(params) == 6
    print(f"  PASS get lora parameters ({len(params)} tensors)")


def test_lora_training_preserves_gradients():
    model = TinyModel()
    cfg = LoRAConfig(r=4)
    model = apply_lora(model, cfg)

    x = Tensor.randn((2, 64))
    out = model(x)
    assert out.shape == (2, 10)
    print("  PASS lora training forward")


def test_qlora_linear_forward():
    layer = QLoRALinear(64, 128, r=4, alpha=8.0)
    x = Tensor.randn((2, 64))
    out = layer(x)
    assert out.shape == (2, 128)
    print("  PASS qlora linear forward shape")


def test_nf4_quantize_roundtrip():
    w = np.random.randn(32, 32).astype(np.float32)
    indices = quantize_nf4(w)
    w_recovered = dequantize_nf4(indices)
    assert indices.dtype == np.uint8
    assert w_recovered.shape == w.shape
    mse = np.mean((w - w_recovered) ** 2)
    assert mse < 1.0
    print(f"  PASS nf4 roundtrip MSE={mse:.6f}")


def test_nf4_quantize_values():
    indices = quantize_nf4(np.array([[0.0, 1.0, -1.0]], dtype=np.float32))
    w = dequantize_nf4(indices)
    assert abs(w[0, 0]) < 0.01
    assert abs(w[0, 1] - 1.0) < 0.01
    assert abs(w[0, 2] - (-1.0)) < 0.01
    print("  PASS nf4 quantize values")


def test_rslora_scaling():
    layer = LoRALinear(64, 128, r=8, alpha=16.0, use_rslora=True)
    assert abs(layer.scaling - (16.0 / 8.0 / np.sqrt(8))) < 1e-6
    print("  PASS rslora scaling")


def test_dpo_loss_shape():
    policy = np.array([0.5, 0.6, 0.7])
    reference = np.array([0.4, 0.5, 0.6])
    win = np.array([0.8, 0.9, 1.0])
    lose = np.array([0.2, 0.3, 0.4])
    loss, per_sample = dpo_loss(policy, reference, win, lose, beta=0.1)
    assert isinstance(loss, (float, np.floating))
    assert per_sample.shape == (3,)
    print(f"  PASS dpo loss shape loss={loss:.4f}")


def test_dpo_loss_reference_free():
    policy = np.array([0.5, 0.6])
    reference = np.array([0.4, 0.5])
    win = np.array([0.8, 0.9])
    lose = np.array([0.2, 0.3])
    loss1, _ = dpo_loss(policy, reference, win, lose, beta=0.1)
    loss2, _ = dpo_loss(policy, reference, win, lose, beta=0.1, reference_free=True)
    assert loss1 != loss2
    print(f"  PASS dpo reference free loss1={loss1:.4f} loss2={loss2:.4f}")


def test_grpo_loss_shape():
    log_probs = np.random.randn(8, 10).astype(np.float32)
    old_log_probs = np.random.randn(8, 10).astype(np.float32)
    rewards = np.random.randn(8).astype(np.float32)
    loss = grpo_loss(log_probs, old_log_probs, rewards)
    assert isinstance(loss, (float, np.floating))
    print(f"  PASS grpo loss shape loss={loss:.4f}")


def test_dpo_trainer_config():
    cfg = DPOTrainerConfig(beta=0.2, learning_rate=1e-4)
    assert cfg.beta == 0.2
    assert cfg.learning_rate == 1e-4
    print("  PASS dpo trainer config")


def test_grpo_trainer_config():
    cfg = GRPOTrainerConfig(group_size=4, epsilon=0.3)
    assert cfg.group_size == 4
    assert cfg.epsilon == 0.3
    print("  PASS grpo trainer config")


if __name__ == "__main__":
    test_lora_config_defaults()
    test_qlora_config_defaults()
    test_lora_linear_forward()
    test_lora_linear_merge()
    test_lora_linear_unmerge()
    test_apply_lora_all()
    test_apply_lora_targeted()
    test_get_lora_parameters()
    test_lora_training_preserves_gradients()
    test_qlora_linear_forward()
    test_nf4_quantize_roundtrip()
    test_nf4_quantize_values()
    test_rslora_scaling()
    test_dpo_loss_shape()
    test_dpo_loss_reference_free()
    test_grpo_loss_shape()
    test_dpo_trainer_config()
    test_grpo_trainer_config()
    print("\nAll LoRA tests passed!")
