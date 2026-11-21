"""Comprehensive benchmarking suite for SNEPPX operations."""

import time
import statistics
import platform
import sys
import os
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

try:
    from ..tensor import Tensor
    from ..advanced_ops import (
        conv2d,
        conv1d,
        max_pool2d,
        avg_pool2d,
        multi_head_attention,
        transformer_block,
        linear,
        layernorm,
        rmsnorm,
        gelu,
        silu,
        relu,
        softmax,
        batch_norm,
        layer_norm,
        group_norm,
        matmul,
        add,
        mul,
        matmul_batched,
        matmul_strided,
    )
except ImportError:
    # For standalone benchmarking
    pass


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # operations/sec
    memory_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    warmup_iters: int = 10
    measure_iters: int = 100
    min_time_s: float = 1.0
    max_iters: int = 10000
    verbose: bool = True


class BenchmarkTimer:
    """High-resolution timer with statistics."""

    def __init__(self):
        self.times: List[float] = []
        self._start: float = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)
        return elapsed

    def reset(self):
        self.times.clear()

    def stats(self) -> Tuple[float, float, float, float, float]:
        if not self.times:
            return 0, 0, 0, 0, 0
        arr = np.array(self.times)
        return (
            float(np.mean(arr) * 1000),  # mean ms
            float(np.median(arr) * 1000),  # median ms
            float(np.std(arr) * 1000),  # std ms
            float(np.min(arr) * 1000),  # min ms
            float(np.max(arr) * 1000),  # max ms
        )


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.timer = BenchmarkTimer()
        self._device_info = self._get_device_info()

    def _get_device_info(self) -> Dict[str, str]:
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu": platform.processor(),
        }
        try:
            import psutil

            info["memory_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
        except ImportError:
            pass
        return info

    def _warmup(self, fn: Callable, *args, **kwargs):
        for _ in range(self.config.warmup_iters):
            fn(*args, **kwargs)

    def _measure(self, fn: Callable, *args, **kwargs) -> List[float]:
        self.timer.reset()

        # Adaptive iteration count
        target_iters = self.config.measure_iters
        min_time = self.config.min_time_s

        # Quick estimate
        start = time.perf_counter()
        for _ in range(5):
            fn(*args, **kwargs)
        est_per_iter = (time.perf_counter() - start) / 5

        # Adjust iterations to meet min_time
        if est_per_iter > 0:
            target_iters = max(
                self.config.measure_iters, int(min_time / est_per_iter) + 1
            )
        else:
            target_iters = self.config.measure_iters

        target_iters = min(target_iters, self.config.max_iters)

        for _ in range(target_iters):
            self.timer.start()
            fn(*args, **kwargs)
            self.timer.stop()

        return self.timer.times

    def benchmark(
        self, name: str, fn: Callable, *args, memory_mb: float = 0, **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function."""
        if self.config.verbose:
            print(f"  Benchmarking {name}...", end=" ", flush=True)

        self._warmup(fn, *args, **kwargs)
        times = self._measure(fn, *args, **kwargs)

        mean_ms, median_ms, std_ms, min_ms, max_ms = self.timer.stats()
        total_time = sum(self.timer.times)
        throughput = len(times) / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time_s=total_time,
            mean_time_ms=mean_ms,
            median_time_ms=median_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            throughput=throughput,
            memory_mb=memory_mb,
            metadata={"device_info": self._device_info},
        )

        self.results.append(result)

        if self.config.verbose:
            print(f"done ({mean_ms:.2f} ms/iter, {throughput:.0f} ops/s)")

        return result

    def benchmark_matmul(
        self, shapes: List[Tuple[int, int, int]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark matrix multiplication."""
        if shapes is None:
            shapes = [
                (64, 64, 64),
                (128, 128, 128),
                (256, 256, 256),
                (512, 512, 512),
                (1024, 1024, 1024),
                (64, 2048, 512),
                (512, 64, 2048),  # attention shapes
            ]

        results = []
        for M, N, K in shapes:
            A = Tensor.from_numpy(np.random.randn(M, K).astype(np.float32))
            B = Tensor.from_numpy(np.random.randn(K, N).astype(np.float32))

            mem_mb = (M * K + K * N + M * N) * 4 / (1024 * 1024)

            result = self.benchmark(
                f"matmul_{M}x{N}x{K}", lambda: matmul(A, B), memory_mb=mem_mb
            )
            results.append(result)
        return results

    def benchmark_conv2d(self, configs: List[Dict] = None) -> List[BenchmarkResult]:
        """Benchmark 2D convolution."""
        if configs is None:
            configs = [
                {"N": 1, "C": 3, "H": 224, "W": 224, "OC": 64, "k": 7, "s": 2},
                {"N": 4, "C": 64, "H": 56, "W": 56, "OC": 128, "k": 3, "s": 1},
                {"N": 8, "C": 256, "H": 28, "W": 28, "OC": 512, "k": 3, "s": 1},
            ]

        results = []
        for cfg in configs:
            N, C, H, W, OC, k, s = (
                cfg[k] for k in ["N", "C", "H", "W", "OC", "k", "s"]
            )
            x = Tensor.from_numpy(np.random.randn(N, C, H, W).astype(np.float32))
            w = Tensor.from_numpy(np.random.randn(OC, C, k, k).astype(np.float32))

            mem_mb = (
                (N * C * H * W + OC * C * k * k + N * OC * (H // s) * (W // s))
                * 4
                / (1024 * 1024)
            )

            result = self.benchmark(
                f"conv2d_N{N}_C{C}_H{H}_W{W}_OC{OC}_k{k}_s{s}",
                lambda: conv2d(x, w, stride=(s, s), padding=(k // 2, k // 2)),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_attention(self, configs: List[Dict] = None) -> List[BenchmarkResult]:
        """Benchmark multi-head attention."""
        if configs is None:
            configs = [
                {"batch": 2, "seq": 128, "dim": 512, "heads": 8},
                {"batch": 4, "seq": 512, "dim": 768, "heads": 12},
                {"batch": 1, "seq": 2048, "dim": 1024, "heads": 16},
                {"batch": 8, "seq": 1024, "dim": 2048, "heads": 32},
            ]

        results = []
        for cfg in configs:
            B, L, D, H = cfg["batch"], cfg["seq"], cfg["dim"], cfg["heads"]
            q = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))
            k = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))
            v = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))

            mem_mb = (B * L * D * 4 + B * H * L * L * 4) / (1024 * 1024)

            result = self.benchmark(
                f"mha_B{B}_L{L}_D{D}_H{H}",
                lambda: multi_head_attention(q, k, v, num_heads=H),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_transformer_block(
        self, configs: List[Dict] = None
    ) -> List[BenchmarkResult]:
        """Benchmark transformer block."""
        if configs is None:
            configs = [
                {"batch": 2, "seq": 128, "dim": 512, "heads": 8, "ffn": 2048},
                {"batch": 4, "seq": 512, "dim": 768, "heads": 12, "ffn": 3072},
                {"batch": 1, "seq": 2048, "dim": 1024, "heads": 16, "ffn": 4096},
            ]

        results = []
        for cfg in configs:
            B, L, D, H, F = (
                cfg["batch"],
                cfg["seq"],
                cfg["dim"],
                cfg["heads"],
                cfg["ffn"],
            )
            x = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))

            mem_mb = B * L * D * 4 * 10 / (1024 * 1024)  # rough estimate

            # Mock weights
            class MockModule:
                def __init__(self):
                    self.w_qkv = Tensor.from_numpy(
                        np.random.randn(3 * D, D).astype(np.float32)
                    )
                    self.w_o = Tensor.from_numpy(
                        np.random.randn(D, D).astype(np.float32)
                    )
                    self.ln1_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
                    self.ln1_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
                    self.ln2_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
                    self.ln2_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
                    self.ff1_w = Tensor.from_numpy(
                        np.random.randn(D * 4, D).astype(np.float32)
                    )
                    self.ff1_b = Tensor.from_numpy(np.zeros(D * 4, dtype=np.float32))
                    self.ff2_w = Tensor.from_numpy(
                        np.random.randn(D, D * 4).astype(np.float32)
                    )
                    self.ff2_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
                    self.ln2_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
                    self.ln2_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))

            mod = MockModule()

            result = self.benchmark(
                f"transformer_B{B}_L{L}_D{D}_H{H}",
                lambda: transformer_block(x, mod),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_activations(self, sizes: List[int] = None) -> List[BenchmarkResult]:
        """Benchmark activation functions."""
        if sizes is None:
            sizes = [1024, 4096, 16384, 65536, 262144]

        acts = [
            ("relu", relu),
            ("gelu", gelu),
            ("silu", silu),
            ("sigmoid", sigmoid),
            ("tanh", tanh),
        ]

        results = []
        for name, fn in acts:
            for size in sizes:
                x = Tensor.from_numpy(np.random.randn(size).astype(np.float32))
                mem_mb = size * 4 / (1024 * 1024)
                result = self.benchmark(
                    f"{name}_{size}", lambda: fn(x), memory_mb=mem_mb
                )
                results.append(result)
        return results

    def benchmark_norms(
        self, shapes: List[Tuple[int, ...]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark normalization layers."""
        if shapes is None:
            shapes = [
                (32, 512),
                (64, 768),
                (32, 1024),
                (16, 2048),
                (4, 32, 56, 56),
                (8, 64, 28, 28),
            ]

        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            C = shape[1] if len(shape) > 1 else shape[0]
            w = Tensor.from_numpy(np.ones(C, dtype=np.float32))
            b = Tensor.from_numpy(np.zeros(C, dtype=np.float32))

            mem_mb = np.prod(shape) * 4 * 3 / (1024 * 1024)

            for name, fn in [
                ("layernorm", layernorm),
                ("rmsnorm", rmsnorm),
                ("batchnorm", batch_norm),
                ("groupnorm", group_norm),
            ]:
                if name == "batchnorm" and len(shape) != 4:
                    continue
                if name == "groupnorm" and len(shape) != 4:
                    continue

                result = self.benchmark(
                    f"{name}_{'x'.join(map(str, shape))}",
                    lambda: fn(
                        x,
                        Tensor.from_numpy(np.ones(C, dtype=np.float32)),
                        Tensor.from_numpy(np.zeros(C, dtype=np.float32)),
                    ),
                    memory_mb=mem_mb,
                )
                results.append(result)
        return results

    def run_full_suite(self) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark suite."""
        print(f"\n{'='*60}")
        print(f"SNEPPX Benchmark Suite")
        print(f"Platform: {self._device_info['platform']}")
        print(
            f"Python: {self._device_info['python']}, NumPy: {self._device_info['numpy']}"
        )
        print(f"{'='*60}\n")

        suite = {}

        print("Running MatMul benchmarks...")
        suite["matmul"] = self.benchmark_matmul()

        print("\nRunning Conv2D benchmarks...")
        suite["conv2d"] = self.benchmark_conv2d()

        print("\nRunning Attention benchmarks...")
        suite["attention"] = self.benchmark_attention()

        print("\nRunning Transformer Block benchmarks...")
        suite["transformer"] = self.benchmark_transformer_block()

        print("\nRunning Activation benchmarks...")
        suite["activations"] = self.benchmark_activations()

        print("\nRunning Normalization benchmarks...")
        suite["norms"] = self.benchmark_norms()

        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        self.print_summary()

        return suite

    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            return

        # Group by category
        categories = {}
        for r in self.results:
            cat = r.name.split("_")[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        for cat, results in sorted(categories.items()):
            print(f"\n{cat.upper()}:")
            for r in sorted(results, key=lambda x: x.mean_time_ms):
                print(
                    f"  {r.name:40s} {r.mean_time_ms:>8.2f} ms  "
                    f"({r.throughput:>10.0f} ops/s  {r.memory_mb:>6.1f} MB)"
                )


# Standalone benchmark functions for CLI
def run_benchmark_cli():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SNEPPX Benchmark Suite")
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "matmul",
            "conv",
            "attention",
            "transformer",
            "activations",
            "norms",
        ],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument("--iters", type=int, default=100, help="Measure iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    config = BenchmarkConfig(
        warmup_iters=args.warmup,
        measure_iters=args.iters,
    )

    suite = BenchmarkSuite(config)

    if args.suite == "all":
        results = suite.run_full_suite()
    elif args.suite == "matmul":
        results = {"matmul": suite.benchmark_matmul()}
    elif args.suite == "conv":
        results = {"conv2d": suite.benchmark_conv2d()}
    elif args.suite == "attention":
        results = {"attention": suite.benchmark_attention()}
    elif args.suite == "transformer":
        results = {"transformer": suite.benchmark_transformer_block()}
    elif args.suite == "activations":
        results = {"activations": suite.benchmark_activations()}
    elif args.suite == "norms":
        results = {"norms": suite.benchmark_norms()}
    else:
        results = suite.run_full_suite()

    if args.output:
        import json

        out = {}
        for cat, res in results.items():
            out[cat] = [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "mean_ms": r.mean_time_ms,
                    "median_ms": r.median_time_ms,
                    "std_ms": r.std_time_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                }
                for r in res
            ]
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    run_benchmark_cli()
