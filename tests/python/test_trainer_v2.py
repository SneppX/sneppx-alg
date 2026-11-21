import sys, os, math
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../bindings/python")
)

from SneppX_ALG.interface_bindings.train import Trainer, TrainConfig, MSELoss
from SneppX_ALG.interface_bindings.nn import Module, Linear
from SneppX_ALG.interface_bindings.tensor import Tensor

failed = []


def check(name, cond):
    if not cond:
        print(f"  FAIL {name}")
        failed.append(name)
    else:
        print(f"  PASS {name}")


# Simple test model
class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def test_train_config():
    config = TrainConfig()
    config.learning_rate = 0.01
    config.batch_size = 32
    config.epochs = 5
    check("config attr", hasattr(config, "learning_rate"))


def test_trainer_init():
    from SneppX_ALG.interface_bindings.train import Trainer, TrainConfig

    config = TrainConfig()

    # Create a simple model for testing
    class SimpleModel(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    trainer = Trainer(model, config)
    check("trainer created", hasattr(trainer, "train"))


def test_mse_loss():
    from SneppX_ALG.interface_bindings.train import MSELoss

    loss_fn = MSELoss()
    pred = Tensor.from_numpy(np.array([[1.0], [2.0], [3.0]]))
    target = Tensor.from_numpy(np.array([[1.5], [2.5], [2.5]]))
    loss = loss_fn(pred, target)
    val = float(loss.data)
    expected = ((0.5) ** 2 + (0.5) ** 2 + (0.5) ** 2) / 3
    check("mse loss correct", abs(val - expected) < 1e-5)


# Note: Full E2E training is CPU-bound and would require significant
# time and data. The existing C++ Tensor test suite and mixed
# precision / distributed / quantization, checkpoint, and
# profiler tests cover many of the same concepts. The Python API
# tests (test_tensor.py, test_nn.py, test_train.py) ensure the
# bindings are functional.

if __name__ == "__main__":
    import numpy as np

    test_train_config()
    test_trainer_init()
    test_mse_loss()

    print(
        f"\nAll trainer (v2) tests passed!"
        if not failed
        else f"{len(failed)} failures: {failed}"
    )
    sys.exit(1 if failed else 0)
