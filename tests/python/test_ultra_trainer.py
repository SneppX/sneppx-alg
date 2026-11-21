import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../bindings/python")
)
from SneppX_ALG.interface_bindings import tensor
from SneppX_ALG.interface_bindings.amp import GradScaler, autocast
from SneppX_ALG.interface_bindings.checkpoint import CheckpointCoordinator
from SneppX_ALG.interface_bindings.model_zoo import get_model_config
from SneppX_ALG.interface_bindings.nn import Linear, Module
from SneppX_ALG.interface_bindings.profiler import Timer
from SneppX_ALG.interface_bindings.quantization import quantize_int8_sym
from SneppX_ALG.interface_bindings.schedulers import LRScheduler
from SneppX_ALG.interface_bindings.train import SGD


# Simple test model
class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


# Minimal training loop
def test_basic_training():
    model = SimpleModel()
    optimizer = SGD(
        [p for m in [model.fc1, model.fc2] for p in m.parameters()], lr=0.01
    )

    # Create synthetic data
    data = []
    for i in range(10):
        inp = [float(i) / 10.0] * 10
        tgt = [float(i) / 10.0]
        data.append((inp, tgt))

    total_loss = 0.0
    for epoch in range(10):
        epoch_loss = 0.0
        for inp, tgt in data:
            inp_t = tensor.Tensor.from_numpy(
                np.array(inp, dtype=np.float32), dtype="float32"
            )
            tgt_t = tensor.Tensor.from_numpy(
                np.array(tgt, dtype=np.float32), dtype="float32"
            )
            out = model(inp_t)
            loss = ((out - tgt_t) ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.data.item())
        total_loss += epoch_loss
        print(f"  Epoch {epoch:2d}: loss = {epoch_loss:.4f}")

    avg_loss = total_loss / len(data) / 10
    assert avg_loss < 1.0, f"Loss too high: {avg_loss}"
    print(f"Basic training completed: avg loss {avg_loss:.4f}")


def test_autocast():
    with autocast(enabled=True, dtype="float16"), Timer("autocast_block"):
        a = tensor.Tensor.from_numpy(
            np.array([1.0, 2.0, 3.0]), dtype="float32"
        )
        b = tensor.Tensor.from_numpy(
            np.array([0.5, 1.0, 1.5]), dtype="float32"
        )
        c = a * b
        c.sum()
        print(f"  Autocast done, dtype = {a.dtype_name}")


def test_amp():
    scaler = GradScaler(init_scale=2.0**16, growth_factor=2.0)
    params = [
        tensor.Tensor(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dtype="float32",
            requires_grad=True,
        )
    ]
    loss = (params[0] * params[0]).sum()
    scaled_loss = scaler.scale_loss(loss)
    assert float(scaled_loss.data.item()) == float(loss.data.item()) * scaler.scale
    scaled_loss.backward()
    assert params[0].grad is not None
    scaler.unscale_([params[0].grad])
    np.testing.assert_allclose(params[0].grad.data, [2.0, 4.0, 6.0], rtol=1e-5, atol=1e-6)
    print(f"  AMPLoss (scaled): {float(scaled_loss.data.item())}")


def test_schedule():
    model = SimpleModel()
    optimizer = SGD([p for m in [model.fc1, model.fc2] for p in m.parameters()], lr=0.1)
    scheduler = LRScheduler(optimizer, last_epoch=-1)
    print(f"  Scheduler LR before step: {optimizer.lr}")
    scheduler.step()
    print(f"  Scheduler LR after step: {optimizer.lr}")


def test_checkpointing():
    ckpt_dir = tempfile.mkdtemp()
    try:
        coord = CheckpointCoordinator(
            ckpt_dir, world_size=1, rank=0, save_interval=1, async_save=False
        )
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        coord.save(data, step=42, metadata={"epoch": 1})
        time.sleep(0.1)
        loaded_data, meta = coord.load()
        assert np.allclose(loaded_data, data), "Checkpoint mismatch"
        print(f"  Checkpoint pass: metadata={meta}")
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_quantization():
    inp = tensor.Tensor.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0]), dtype="float32"
    )
    quant_tensor, scale = quantize_int8_sym(inp)
    print(f"  INT8 quantized: scale={scale:.4f}, dtype={quant_tensor.dtype_name}")


def test_model_zoo():
    config = get_model_config("llama3", "8B")
    assert config["hidden_size"] == 4096
    assert config["num_hidden_layers"] == 32
    assert config["num_attention_heads"] == 32
    print(f"  Model zoo config loaded: {config['family']}-{config['size']}")


def main():
    print("=== UltraTrainer Test Suite ===")
    print("\n1. Basic training loop")
    test_basic_training()
    print("\n2. Autocast")
    test_autocast()
    print("\n3. AMP GradScaler")
    test_amp()
    print("\n4. Scheduler")
    test_schedule()
    print("\n5. Checkpointing")
    test_checkpointing()
    print("\n6. Quantization")
    test_quantization()
    print("\n7. Model zoo configs")
    test_model_zoo()
    print("\n=== All UltraTrainer tests passed! ===")


if __name__ == "__main__":
    main()
