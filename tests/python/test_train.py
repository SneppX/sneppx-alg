"""Tests for Trainer and training pipeline."""

import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Trainer,
    TrainConfig,
    MSELoss,
    CrossEntropyLoss,
    DataLoader,
    CppTensorDataset,
    _HAS_C_BACKEND,
)
from SneppX_ALG.interface_bindings.model import Model, ModelConfig


def test_train_config():
    config = TrainConfig()
    config.num_epochs = 10
    config.batch_size = 32
    config.learning_rate = 0.001
    assert config.num_epochs == 10
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    print("  test_train_config PASS")


def test_trainer_create():
    mcfg = ModelConfig()
    mcfg.input_dim = 8
    mcfg.output_dim = 8
    model = Model(mcfg)
    config = TrainConfig()
    config.learning_rate = 0.01
    config.batch_size = 4
    trainer = Trainer(model, config)
    assert trainer is not None
    print("  test_trainer_create PASS")


def test_trainer_train_eval():
    if not _HAS_C_BACKEND:
        print("  test_trainer_train_eval SKIP (no C backend)")
        return
    mcfg = ModelConfig()
    mcfg.input_dim = 8
    mcfg.output_dim = 8
    model = Model(mcfg)
    config = TrainConfig()
    config.learning_rate = 0.01
    config.batch_size = 4
    trainer = Trainer(model, config)
    x = Tensor.randn((1, 4, 8))
    y = Tensor.randn((1, 4, 8))
    loss = trainer.train_step(x, y)
    assert loss >= 0
    assert np.isfinite(loss)
    val_loss = trainer.evaluate(x, y)
    assert np.isfinite(val_loss)
    print(f"  test_trainer_train_eval PASS")


def test_trainer_save_load_checkpoints():
    import tempfile, os

    mcfg = ModelConfig()
    mcfg.input_dim = 8
    mcfg.output_dim = 8
    model = Model(mcfg)
    config = TrainConfig()
    config.learning_rate = 0.01
    config.batch_size = 4
    trainer = Trainer(model, config)
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        path = f.name
    try:
        trainer.save_checkpoint(path)
        assert os.path.exists(path)
        trainer.load_checkpoint(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  test_trainer_save_load_checkpoints PASS")


def test_mseloss():
    loss_fn = MSELoss()
    pred = Tensor.ones((3,))
    target = Tensor.zeros((3,))
    loss = loss_fn(pred, target)
    assert loss.shape == (1,)
    assert loss.data[0] > 0
    print("  test_mseloss PASS")


def test_cross_entropy_loss():
    loss_fn = CrossEntropyLoss()
    pred = Tensor.randn((2, 5))
    target = Tensor.from_numpy(np.eye(5)[[0, 1]])
    loss = loss_fn(pred, target)
    assert loss.shape == (1,)
    print("  test_cross_entropy_loss PASS")


def test_tensordataset():
    x = Tensor.ones((10, 4))
    y = Tensor.zeros((10,))
    ds = CppTensorDataset(x, y)
    assert len(ds) == 10
    xi, yi = ds[0]
    assert isinstance(xi, Tensor)
    assert isinstance(yi, Tensor)
    print("  test_tensordataset PASS")


def test_dataloader():
    x = Tensor.ones((20, 4))
    y = Tensor.zeros((20,))
    ds = CppTensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batches = list(loader)
    assert len(batches) == 5
    print(f"  test_dataloader: {len(batches)} batches PASS")


if __name__ == "__main__":
    test_train_config()
    test_trainer_create()
    test_trainer_train_eval()
    test_trainer_save_load_checkpoints()
    test_mseloss()
    test_cross_entropy_loss()
    test_tensordataset()
    test_dataloader()
    print("\nAll train tests passed!")
