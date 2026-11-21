"""CLI launcher for LM evaluation harness (MMLU, GSM8K, HumanEval)."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="SneppX LM Evaluation Harness")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mmlu", "gsm8k", "humaneval"],
                        help="Tasks to evaluate")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .eval_harness import EvalHarness, EvalConfig

    config = EvalConfig(
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=args.output,
        seed=args.seed,
    )
    harness = EvalHarness(config=config)
    harness.register_defaults()

    def model_fn(prompt: str, ctx: dict = None) -> str:
        return f"Eval response for: {prompt[:40]}..."

    results = harness.run(model_fn)
    print(harness.summary())

    if args.output:
        harness.export_json(args.output)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
