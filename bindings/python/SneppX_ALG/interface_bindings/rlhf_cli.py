"""CLI launcher for RLHF training (DPO/GRPO)."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="SneppX RLHF Training")
    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "grpo"],
                        help="RLHF method")
    parser.add_argument("--model", type=str, default=None, help="Model config path")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/GRPO beta")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--output", type=str, default=None, help="Output path for trained model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .lora import LoRAConfig, DPOTrainerConfig, DPOTrainer, GRPOTrainerConfig, GRPOTrainer
    from .nn import Module, Linear

    model = Linear(16, 16)

    if args.method == "dpo":
        trainer_config = DPOTrainerConfig(
            learning_rate=args.learning_rate,
            beta=args.beta,
        )
        trainer = DPOTrainer(model=model, ref_model=Module(), config=trainer_config)

        import numpy as np
        for epoch in range(args.epochs):
            win = np.array([[1, 2, 3, 4, 5]])
            lose = np.array([[1, 2, 3, 4, 6]])
            loss = trainer.train_step(win, lose)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} — DPO loss: {loss:.4f}")

    elif args.method == "grpo":
        trainer_config = GRPOTrainerConfig(
            learning_rate=args.learning_rate,
            beta=args.beta,
            group_size=args.group_size,
        )
        trainer = GRPOTrainer(model=model, config=trainer_config)

        import numpy as np
        for epoch in range(args.epochs):
            prompts = np.array([[1, 2, 3]])
            reward_fn = lambda resp: np.array([1.0 if len(r) > 0 else 0.0 for r in resp])
            loss = trainer.train_step(prompts, reward_fn)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} — GRPO loss: {loss:.4f}")

    logging.info("Training complete")


if __name__ == "__main__":
    main()
