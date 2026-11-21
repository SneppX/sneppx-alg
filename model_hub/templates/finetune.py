#!/usr/bin/env python3
"""
Fine-tuning script template for SNEPPX models.

Usage:
    python finetune.py --model sneppx/llama-7b --dataset /path/to/data.jsonl --epochs 3 --lr 2e-5

This script works with any model from the SNEPPX Hub. It downloads the model,
optionally applies LoRA adapters, and fine-tunes using the built-in trainer.
"""

import argparse
import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model from the SNEPPX Hub")
    parser.add_argument("--model", required=True, help="Model name (e.g. sneppx/llama-7b)")
    parser.add_argument("--version", default="latest", help="Model version (default: latest)")
    parser.add_argument("--dataset", required=True, help="Path to training dataset (JSONL or CSV)")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0 = no LoRA)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-dataset", default=None, help="Path to eval dataset")
    parser.add_argument("--push-to-hub", action="store_true", help="Push fine-tuned model to hub")
    parser.add_argument("--hub-username", default=None, help="Hub username for upload")
    parser.add_argument("--hub-token", default=None, help="Hub API token")
    parser.add_argument("--use-flash-attn", action="store_true", help="Use Flash Attention (CUDA only)")
    parser.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="Mixed precision")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers")

    args = parser.parse_args()

    # Import sneppx after parsing args (allows partial installs)
    try:
        from SneppX_ALG import Hub, Tensor, AdamW, Trainer, LoRAConfig, apply_lora
        from SneppX_ALG.interface_bindings.dataset import load_dataset
    except ImportError as e:
        print("Error: SneppX_ALG not found. Install with: pip install -e bindings/python")
        sys.exit(1)

    # Download model from hub
    print(f"Loading model: {args.model}@{args.version}")
    model = Hub.load(args.model, version=args.version)

    if isinstance(model, str):
        # It's a directory path — load from config
        from SneppX_ALG import build_model_from_config
        with open(os.path.join(model, "config.json")) as f:
            config = json.load(f)
        model = build_model_from_config(config)

    # Optionally apply LoRA
    if args.lora_rank > 0:
        print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
        lora_cfg = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = apply_lora(model, lora_cfg)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # Training config
    from SneppX_ALG import TrainConfig, cosine_schedule_with_warmup

    total_steps = len(dataset) * args.epochs // (args.batch_size * args.gradient_accumulation)
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    train_cfg = TrainConfig(
        experiment_name=f"finetune_{args.model.split('/')[-1]}",
        experiment_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        max_seq_len=args.max_seq_len,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.logging_steps,
        save_interval=args.save_steps,
        num_workers=args.num_workers,
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = cosine_schedule_with_warmup(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_cfg,
    )

    # Optional eval
    eval_dataset = None
    if args.eval_dataset:
        eval_dataset = load_dataset(args.eval_dataset)

    # Train
    print(f"Starting training: {total_steps} total steps")
    start_time = time.time()
    trainer.train(dataset, eval_dataset=eval_dataset)

    # Save final model
    output_path = os.path.join(args.output_dir, "final")
    trainer.save(output_path)
    print(f"Model saved to: {output_path}")

    # Push to hub
    if args.push_to_hub:
        hub = Hub()
        if args.hub_token:
            hub.login(args.hub_token)

        # Create model card
        card = {
            "name": f"{args.hub_username}/{Path(args.model).name}-finetuned",
            "description": f"Fine-tuned from {args.model}",
            "version": "v1.0.0",
            "base_model": args.model,
            "fine_tuned": True,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "lora_rank": args.lora_rank,
            },
        }

        # Prepare upload
        upload_dir = os.path.join(args.output_dir, "hub_upload")
        os.makedirs(upload_dir, exist_ok=True)

        # Copy model files
        if os.path.isdir(output_path):
            for f in os.listdir(output_path):
                src = os.path.join(output_path, f)
                dst = os.path.join(upload_dir, f)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy2(src, dst)

        # Write model card
        with open(os.path.join(upload_dir, "model_card.json"), "w") as f:
            json.dump(card, f, indent=2)

        # Upload
        result = hub.upload(upload_dir)
        print(f"Uploaded to hub: {result}")

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"Final model: {output_path}")


if __name__ == "__main__":
    main()