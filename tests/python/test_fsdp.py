"""Tests for Fully Sharded Data Parallel (FSDP)."""

import os
import tempfile
from pathlib import Path

from SneppX_ALG.interface_bindings.fsdp import (
    FullyShardedDataParallel,
    FSDPConfig,
    ShardingStrategy,
    MixedPrecision,
    shard_model,
    fsdp_save_checkpoint,
    fsdp_load_checkpoint,
    _shard_tensor,
    _gather_tensors,
)
from SneppX_ALG.interface_bindings.nn import Module, Linear, Sequential
from SneppX_ALG.interface_bindings.tensor import Tensor
import numpy as np


class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(64, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x


class SmallModel(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(4, 4, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_sharding_strategy_values():
    assert ShardingStrategy.NO_SHARD.value == "no_shard"
    assert ShardingStrategy.FULL_SHARD.value == "full_shard"


def test_mixed_precision_values():
    assert MixedPrecision.FP32.value == "fp32"
    assert MixedPrecision.FP16.value == "fp16"


def test_fsdp_config_defaults():
    cfg = FSDPConfig()
    assert cfg.sharding_strategy == ShardingStrategy.FULL_SHARD
    assert cfg.mixed_precision == MixedPrecision.FP32
    assert cfg.cpu_offload is False
    assert cfg.min_num_params == 1_000_000


def test_fsdp_config_custom():
    cfg = FSDPConfig(
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=MixedPrecision.BF16,
        cpu_offload=True,
        min_num_params=100,
    )
    assert cfg.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
    assert cfg.mixed_precision == MixedPrecision.BF16
    assert cfg.cpu_offload is True


def test_shard_tensor_world1():
    t = Tensor.ones((10, 10))
    out = _shard_tensor(t, rank=0, world_size=1)
    assert out.shape == t.shape
    assert np.allclose(out.data, t.data)


def test_shard_tensor_world2():
    t = Tensor.from_numpy(np.arange(8).astype(np.float32))
    s0 = _shard_tensor(t, rank=0, world_size=2)
    s1 = _shard_tensor(t, rank=1, world_size=2)
    assert s0.data.size + s1.data.size == t.data.size
    full = _gather_tensors([s0.data, s1.data], 2)
    assert np.allclose(full, t.data)


def test_shard_tensor_world4():
    t = Tensor.from_numpy(np.arange(20).astype(np.float32))
    shards = [_shard_tensor(t, r, 4) for r in range(4)]
    total = sum(s.data.size for s in shards)
    assert total == 20
    full = _gather_tensors([s.data for s in shards], 4)
    assert np.allclose(full, t.data)


def test_shard_tensor_uneven():
    t = Tensor.from_numpy(np.arange(7).astype(np.float32))
    shards = [_shard_tensor(t, r, 3) for r in range(3)]
    total = sum(s.data.size for s in shards)
    assert total == 7
    sizes = [s.data.size for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_full_shard_forward():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    x = Tensor.ones((2, 4))
    out = fsdp(x)
    assert out.shape == (2, 4)


def test_full_shard_forward_single_rank():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    try:
        model = SimpleModel()
        cfg = FSDPConfig(min_num_params=16)
        fsdp = shard_model(model, cfg)
        x = Tensor.ones((2, 64))
        out = fsdp(x)
        assert out.shape == (2, 10)
    finally:
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)


def test_no_shard_config():
    model = SimpleModel()
    cfg = FSDPConfig(sharding_strategy=ShardingStrategy.NO_SHARD, min_num_params=16)
    fsdp = shard_model(model, cfg)
    x = Tensor.ones((2, 64))
    out = fsdp(x)
    assert out.shape == (2, 10)


def test_named_parameters():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    params = fsdp.named_parameters()
    names = [n for n, _ in params]
    assert "fc.weight" in names


def test_parameters_count():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    params = fsdp.parameters()
    assert len(list(params)) == 1


def test_train_eval():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    fsdp.train()
    assert fsdp._module._training is True
    fsdp.eval()
    assert fsdp._module._training is False


def test_state_dict_roundtrip():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    sd = fsdp.state_dict()
    assert "fc.weight" in sd
    new_model = SmallModel()
    new_fsdp = shard_model(new_model, cfg)
    new_fsdp.load_state_dict(sd)
    x = Tensor.ones((1, 4))
    out1 = fsdp(x)
    out2 = new_fsdp(x)
    assert np.allclose(out1.data, out2.data)


def test_save_load_checkpoint():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    x = Tensor.ones((1, 4))
    fsdp(x)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        fsdp_save_checkpoint(fsdp, path)
        new_model = SmallModel()
        new_fsdp = shard_model(new_model, cfg)
        fsdp_load_checkpoint(new_fsdp, path)
        out1 = fsdp(Tensor.ones((1, 4)))
        out2 = new_fsdp(Tensor.ones((1, 4)))
        assert np.allclose(out1.data, out2.data)
    finally:
        Path(path).unlink(missing_ok=True)


def test_to_device():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    fsdp.to("cpu")
    for p in fsdp.parameters():
        assert p.device == "cpu"


def test_shard_model_creates_fsdp():
    model = SimpleModel()
    fsdp = shard_model(model)
    assert isinstance(fsdp, FullyShardedDataParallel)


def test_shard_model_respects_min_params():
    model = SimpleModel()
    cfg = FSDPConfig(min_num_params=10_000_000)
    fsdp = shard_model(model, cfg)
    assert len(fsdp._sharded_params) == 0


def test_shard_model_ignores_modules():
    model = SimpleModel()
    cfg = FSDPConfig(min_num_params=0, ignored_modules=["fc2"])
    fsdp = shard_model(model, cfg)
    shard_names = list(fsdp._sharded_params.keys())
    assert all("fc2" not in n for n in shard_names)


def test_all_gather_single_rank():
    model = SmallModel()
    cfg = FSDPConfig(min_num_params=0)
    fsdp = shard_model(model, cfg)
    gathered = fsdp.all_gather()
    assert len(gathered) == 1
    assert "fc.weight" in gathered


def test_module_property():
    model = SmallModel()
    fsdp = shard_model(model)
    assert fsdp.module is model


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
