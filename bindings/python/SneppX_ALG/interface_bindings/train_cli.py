"""CLI entry point for training with the v3 autograd trainer.

Usage:
    python -m SneppX_ALG.interface_bindings.train_cli --config config.yaml
    python -m SneppX_ALG.interface_bindings.train_cli --config config.yaml --eval-only
"""

import os
import sys
import argparse
import json

from .trainer_v3 import Trainer, TrainConfig


def build_model_from_config(model_config: dict):
    """Build a model from a config dict.

    Supports basic ``Linear`` models and can be extended for LLMs.
    """
    from .nn import Linear, Sequential, ReLU
    from .tensor import Tensor

    name = model_config.get("name", "linear").lower()
    if name == "linear":
        from .nn import Linear as Lin

        in_dim = model_config.get("in_features", 64)
        out_dim = model_config.get("out_features", 10)
        return Lin(in_dim, out_dim)
    elif name == "mlp":
        from .nn import Linear as Lin, ReLU

        dims = model_config.get("dims", [64, 128, 10])
        layers = []
        for i in range(len(dims) - 1):
            layers.append(Lin(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(ReLU())
        return Sequential(*layers)
    else:
        raise ValueError(f"Unknown model: {name}")


def build_dummy_data(config):
    """Build a synthetic data loader for testing."""
    from .tensor import Tensor
    import numpy as np

    batch_size = config.training.batch_size
    in_features = 64
    out_features = 10
    num_batches = 100

    class DummyLoader:
        def __init__(self):
            self._batches = [
                (
                    Tensor(np.random.randn(batch_size, in_features).astype(np.float32)),
                    Tensor(
                        np.random.randn(batch_size, out_features).astype(np.float32)
                    ),
                )
                for _ in range(num_batches)
            ]
            self._idx = 0

        def __iter__(self):
            self._idx = 0
            return self

        def __next__(self):
            if self._idx >= len(self._batches):
                raise StopIteration
            batch = self._batches[self._idx]
            self._idx += 1
            return batch

    return DummyLoader()


def main():
    parser = argparse.ArgumentParser(description="SneppX Training CLI")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training, run evaluation only"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--override",
        "-o",
        type=str,
        default=None,
        help="JSON string of config overrides",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Override checkpoint directory"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max training steps"
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    if args.override:
        overrides = json.loads(args.override)
        config.override(overrides)

    if args.checkpoint_dir:
        config._data["training"]["checkpoint_dir"] = args.checkpoint_dir

    resume_from = args.resume or config.training.resume_from

    model_config = config._data.get("model", {})
    model = build_model_from_config(model_config)

    trainer = Trainer(model, config)

    if resume_from:
        trainer.load(resume_from)

    train_loader = build_dummy_data(config)
    val_loader = build_dummy_data(config) if config.training.eval_every > 0 else None

    if args.eval_only:
        loss = trainer.evaluate(val_loader)
        print(f"Evaluation loss: {loss:.4f}")
    else:
        trainer.fit(
            train_loader,
            val_loader=val_loader,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
