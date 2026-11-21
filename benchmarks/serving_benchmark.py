"""Benchmark: compare static batch_generate vs continuous batching overhead."""

import time
import numpy as np

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.nn import Module
from SneppX_ALG.interface_bindings.generation import batch_generate, GenerationConfig
from SneppX_ALG.interface_bindings.continuous_batching import (
    ContinuousBatchScheduler,
    continuous_generate_loop,
)


class DummyModel(Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = input_ids.data if hasattr(input_ids, "data") else np.asarray(input_ids)
        batch, seq = x.shape if x.ndim == 2 else (1, x.shape[0])
        return {"logits": Tensor.randn((batch, seq, self.vocab_size))}


def pad_to_max(sequences, pad_id=0):
    max_len = max(len(s) for s in sequences)
    padded = []
    for s in sequences:
        arr = np.array(s, dtype=np.int64)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=pad_id)
        padded.append(arr)
    return np.stack(padded)


def benchmark_static_batch(model, prompts, gen_config):
    """Benchmark using batch_generate (static batching)."""
    padded = pad_to_max(prompts)
    start = time.perf_counter()
    batch_generate(model, padded, generation_config=gen_config)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_continuous_batch(model, prompts, gen_config):
    """Benchmark the continuous batching scheduler overhead."""
    scheduler = ContinuousBatchScheduler()
    for p in prompts:
        scheduler.submit(
            prompt="", prompt_tokens=p,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
        )

    def model_fn(input_ids_list, positions):
        padded = pad_to_max(input_ids_list)
        result = batch_generate(model, padded, generation_config=gen_config)
        tokens_out = result["output_ids"]
        next_tokens = []
        for i, ids in enumerate(input_ids_list):
            row = tokens_out[i] if tokens_out.ndim > 1 else tokens_out
            next_tokens.append([int(row[len(ids)])] if len(row) > len(ids) else [0])
        return next_tokens, None

    def decode_fn(token_ids):
        return ",".join(str(t) for t in token_ids)

    start = time.perf_counter()
    for _ in continuous_generate_loop(
        model_fn, scheduler, decode_fn,
        max_steps=gen_config.max_new_tokens * len(prompts) * 2,
    ):
        pass
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    model = DummyModel()
    gen_config = GenerationConfig(max_new_tokens=4, do_sample=True, temperature=1.0)

    batch_sizes = [1, 2, 4, 8]
    rng = np.random.RandomState(42)

    print(f"{'Batch':>6} | {'Static (s)':>10} | {'Cont (s)':>10} | {'Speedup':>8} | {'Note':>20}")
    print("-" * 62)

    for n in batch_sizes:
        prompts = [rng.randint(0, 128, size=8).tolist() for _ in range(n)]

        t_static = benchmark_static_batch(model, prompts, gen_config)
        t_cont = benchmark_continuous_batch(model, prompts, gen_config)

        speedup = t_static / t_cont if t_cont > 0 else float("inf")
        note = "scheduler overhead dominates" if speedup < 1 else "cont batching wins"
        print(f"{n:>6} | {t_static:>10.4f} | {t_cont:>10.4f} | {speedup:>7.2f}x | {note:>20}")

    print()
    print("Note: Benchmarked with tiny dummy model. Continuous batching's real")
    print("advantage appears with large models where scheduler overhead is negligible.")


if __name__ == "__main__":
    main()
