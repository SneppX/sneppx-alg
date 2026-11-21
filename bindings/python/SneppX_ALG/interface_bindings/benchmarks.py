"""Comprehensive benchmarking suite with microbenchmarks, macrobenchmarks, and profiling."""

import time
import statistics
import platform
import sys
import os
import json
import csv
from typing import Dict, List, Any, Callable, Optional, Tuple, Dict
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    warmup_iters: int = 10
    measure_iters: int = 100
    min_time_s: float = 1.0
    max_iters: int = 10000
    verbose: bool = True
    output_dir: Optional[str] = None
    save_json: bool = True
    save_csv: bool = True


@dataclass
class BenchmarkResult:
    name: str
    category: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    median_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_s: float
    memory_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkTimer:
    """High-resolution timer with statistical analysis."""

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
        arr = np.array(self.times) * 1000
        return (
            float(np.mean(arr)),
            float(np.median(arr)),
            float(np.std(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
        )

    def percentile(self, p: float) -> float:
        if not self.times:
            return 0
        return float(np.percentile(np.array(self.times) * 1000, p))


class MemoryTracker:
    """Track memory usage during benchmarks."""

    def __init__(self):
        self.process = psutil.Process() if HAS_PSUTIL else None
        self.peak_mb = 0.0
        self.samples: List[float] = []

    def start(self):
        if self.process:
            self.peak_mb = 0.0
            self.samples.clear()
            self._sample()

    def _sample(self):
        if self.process:
            mb = self.process.memory_info().rss / (1024 * 1024)
            self.samples.append(mb)
            if mb > self.peak_mb:
                self.peak_mb = mb

    def stop(self) -> float:
        self._sample()
        return self.peak_mb


class BenchmarkSuite:
    """Comprehensive benchmarking suite for SNEPPX operations."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.timer = BenchmarkTimer()
        self.memory = MemoryTracker() if HAS_PSUTIL else None
        self._device_info = self._get_device_info()

    def _get_device_info(self) -> Dict[str, str]:
        info = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpu": platform.processor(),
        }
        if HAS_PSUTIL:
            info["memory_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
            info["cpu_count"] = str(psutil.cpu_count())
        return info

    def _warmup(self, fn: Callable, *args, **kwargs):
        for _ in range(self.config.warmup_iters):
            fn(*args, **kwargs)

    def _measure(self, fn: Callable, *args, **kwargs) -> List[float]:
        self.timer.reset()
        if self.memory:
            self.memory.start()

        # Adaptive iteration count
        target_iters = self.config.measure_iters
        min_time = self.config.min_time_s

        # Quick estimate
        start = time.perf_counter()
        for _ in range(5):
            fn(*args, **kwargs)
        est_per_iter = (time.perf_counter() - start) / 5

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
            if self.memory:
                self.memory._sample()

        peak_mem = self.memory.stop() if self.memory else 0.0

        return self.timer.times, peak_mem

    def benchmark(
        self,
        name: str,
        category: str,
        fn: Callable,
        *args,
        memory_mb: float = 0,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a function."""
        if self.config.verbose:
            print(f"  Benchmarking {name}...", end=" ", flush=True)

        self._warmup(fn, *args, **kwargs)
        times, peak_mem = self._measure(fn, *args, **kwargs)

        mean_ms, median_ms, std_ms, min_ms, max_ms = self.timer.stats()
        total_time = sum(self.timer.times)
        throughput = len(times) / total_time if total_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=len(times),
            total_time_s=total_time,
            mean_time_ms=mean_ms,
            median_time_ms=median_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            throughput_ops_s=throughput,
            memory_mb=max(memory_mb, peak_mem),
            metadata={
                "device_info": self._device_info,
                "p99_ms": self.timer.percentile(99),
            },
        )

        self.results.append(result)

        if self.config.verbose:
            print(f"done ({mean_ms:.2f} ms/iter, {throughput:.0f} ops/s)")

        return result

    # ==================== Microbenchmarks ====================

    def benchmark_matmul(
        self, shapes: Optional[List[Tuple[int, int, int]]] = None
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
                (512, 64, 2048),
            ]

        from .advanced_ops import matmul

        results = []
        for M, N, K in shapes:
            A = Tensor.from_numpy(np.random.randn(M, K).astype(np.float32))
            B = Tensor.from_numpy(np.random.randn(K, N).astype(np.float32))
            mem_mb = (M * K + K * N + M * N) * 4 / (1024 * 1024)

            result = self.benchmark(
                f"matmul_{M}x{N}x{K}", "matmul", lambda: matmul(A, B), memory_mb=mem_mb
            )
            results.append(result)
        return results

    def benchmark_conv2d(
        self, configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark 2D convolution."""
        if configs is None:
            configs = [
                {"N": 1, "C": 3, "H": 224, "W": 224, "OC": 64, "k": 7, "s": 2},
                {"N": 4, "C": 64, "H": 56, "W": 56, "OC": 128, "k": 3, "s": 1},
                {"N": 8, "C": 256, "H": 28, "W": 28, "OC": 512, "k": 3, "s": 1},
            ]

        from .advanced_ops import conv2d

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
                "conv",
                lambda: conv2d(x, w, stride=(s, s), padding=(k // 2, k // 2)),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_attention(
        self, configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark multi-head attention."""
        if configs is None:
            configs = [
                {"batch": 2, "seq": 128, "dim": 512, "heads": 8},
                {"batch": 4, "seq": 512, "dim": 768, "heads": 12},
                {"batch": 1, "seq": 2048, "dim": 1024, "heads": 16},
                {"batch": 8, "seq": 1024, "dim": 2048, "heads": 32},
            ]

        from .advanced_ops import multi_head_attention

        results = []
        for cfg in configs:
            B, L, D, H = cfg["batch"], cfg["seq"], cfg["dim"], cfg["heads"]
            q = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))
            k = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))
            v = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))

            mem_mb = (B * L * D * 4 + B * H * L * L * 4) / (1024 * 1024)

            result = self.benchmark(
                f"mha_B{B}_L{L}_D{D}_H{H}",
                "attention",
                lambda: multi_head_attention(q, k, v, num_heads=H),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_transformer_block(
        self, configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark transformer block."""
        if configs is None:
            configs = [
                {"batch": 2, "seq": 128, "dim": 512, "heads": 8, "ffn": 2048},
                {"batch": 4, "seq": 512, "dim": 768, "heads": 12, "ffn": 3072},
                {"batch": 1, "seq": 2048, "dim": 1024, "heads": 16, "ffn": 4096},
            ]

        from .advanced_ops import transformer_block, layernorm, gelu, linear, dropout

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

            # Mock weights for transformer block
            class MockModule:
                def __init__(self):
                    self.ln1_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
                    self.ln1_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
                    self.ln2_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
                    self.ln2_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
                    self.attn = type(
                        "Attn",
                        (),
                        {
                            "qkv": Tensor.from_numpy(
                                np.random.randn(3 * D, D).astype(np.float32)
                            ),
                            "o": Tensor.from_numpy(
                                np.random.randn(D, D).astype(np.float32)
                            ),
                        },
                    )()
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

            module = MockModule()

            def transformer_forward(x):
                return transformer_block(x, module)

            mem_mb = B * L * D * 4 * 10 / (1024 * 1024)

            result = self.benchmark(
                f"transformer_B{B}_L{L}_D{D}_H{H}_FFN{F}",
                "transformer",
                lambda: transformer_forward(x),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_activation(self) -> List[BenchmarkResult]:
        """Benchmark activation functions."""
        from .advanced_ops import (
            relu,
            gelu,
            silu,
            relu6,
            hardswish,
            leaky_relu,
            elu,
            selu,
            softplus,
            mish,
        )

        shapes = [(1024,), (1024, 1024), (32, 32, 32, 32)]
        activations = [
            ("relu", relu),
            ("gelu", gelu),
            ("silu", silu),
            ("relu6", relu6),
            ("hardswish", hardswish),
            ("leaky_relu", leaky_relu),
            ("elu", elu),
            ("selu", selu),
            ("softplus", softplus),
            ("mish", mish),
        ]

        results = []
        for shape in shapes:
            for name, fn in activations:
                x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
                mem_mb = np.prod(shape) * 4 / (1024 * 1024)

                result = self.benchmark(
                    f"{name}_{'x'.join(map(str, shape))}",
                    "activation",
                    lambda: fn(x),
                    memory_mb=mem_mb,
                )
                results.append(result)
        return results

    def benchmark_norm(self) -> List[BenchmarkResult]:
        """Benchmark normalization layers."""
        from .advanced_ops import (
            layernorm,
            rmsnorm,
            batch_norm,
            group_norm,
            instance_norm,
        )

        shapes = [(32, 128), (32, 512), (16, 256, 32, 32)]
        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))

            # LayerNorm
            w = Tensor.from_numpy(np.ones(shape[-1], dtype=np.float32))
            b = Tensor.from_numpy(np.zeros(shape[-1], dtype=np.float32))

            mem_mb = np.prod(shape) * 4 / (1024 * 1024)

            result = self.benchmark(
                f"layernorm_{'x'.join(map(str, shape))}",
                "norm",
                lambda: layernorm(x, w, b),
                memory_mb=mem_mb,
            )
            results.append(result)

            # RMSNorm
            result = self.benchmark(
                f"rmsnorm_{'x'.join(map(str, shape))}",
                "norm",
                lambda: rmsnorm(x, w),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_reductions(self) -> List[BenchmarkResult]:
        """Benchmark reduction operations."""
        shapes = [(1024,), (1024, 1024), (32, 32, 32, 32)]
        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            mem_mb = np.prod(shape) * 4 / (1024 * 1024)

            for name, fn in [
                ("sum", sum),
                ("mean", mean),
                ("var", var),
                ("std", std),
                ("max", max),
                ("min", min),
                ("prod", prod),
                ("cumsum", cumsum),
                ("cumprod", cumprod),
            ]:
                result = self.benchmark(
                    f"{name}_{'x'.join(map(str, shape))}",
                    "reduction",
                    lambda: fn(x),
                    memory_mb=mem_mb,
                )
                results.append(result)
        return results

    def run_full_suite(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        print(f"Running SNEPPX Benchmark Suite")
        print(f"Device: {self._device_info}")
        print("=" * 60)

        self.benchmark_matmul()
        print()
        self.benchmark_conv2d()
        print()
        self.benchmark_attention()
        print()
        self.benchmark_transformer_block()
        print()
        self.benchmark_activation()
        print()
        self.benchmark_norm()
        print()
        self.benchmark_reductions()
        print()

        print("=" * 60)
        print(f"Total benchmarks: {len(self.results)}")

        if self.config.output_dir:
            self.save_results()

        return self.results

    def save_results(self):
        """Save results to JSON and CSV."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = output_dir / f"benchmark_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "device_info": self._device_info,
                    "timestamp": timestamp,
                    "results": [
                        {
                            "name": r.name,
                            "category": r.category,
                            "iterations": r.iterations,
                            "total_time_s": r.total_time_s,
                            "mean_time_ms": r.mean_time_ms,
                            "median_time_ms": r.median_time_ms,
                            "std_time_ms": r.std_time_ms,
                            "min_time_ms": r.min_time_ms,
                            "max_time_ms": r.max_time_ms,
                            "throughput_ops_s": r.throughput_ops_s,
                            "memory_mb": r.memory_mb,
                            "metadata": r.metadata,
                        }
                        for r in self.results
                    ],
                },
                f,
                indent=2,
            )

        # CSV
        csv_path = output_dir / f"benchmark_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "category",
                    "iterations",
                    "total_time_s",
                    "mean_ms",
                    "median_ms",
                    "std_ms",
                    "min_ms",
                    "max_ms",
                    "throughput_ops_s",
                    "memory_mb",
                    "p99_ms",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.name,
                        r.category,
                        r.iterations,
                        r.total_time_s,
                        r.mean_time_ms,
                        r.median_time_ms,
                        r.std_time_ms,
                        r.min_time_ms,
                        r.max_time_ms,
                        r.throughput_ops_s,
                        r.memory_mb,
                        r.metadata.get("p99_ms", 0),
                    ]
                )

        print(f"Results saved to {output_dir}")

    def print_summary(self):
        """Print benchmark summary table."""
        print(f"\n{'='*100}")
        print(
            f"{'Operation':<40} {'Category':<12} {'Mean (ms)':>10} {'Median (ms)':>12} {'Throughput (ops/s)':>20}"
        )
        print(f"{'-'*100}")

        for r in sorted(self.results, key=lambda x: (x.category, x.mean_time_ms)):
            print(
                f"{r.name:<40} {r.category:<12} {r.mean_time_ms:>10.2f} {r.median_time_ms:>12.2f} {r.throughput_ops_s:>20.0f}"
            )

        print(f"{'='*100}")


def run_benchmarks(output_dir: Optional[str] = None) -> List[BenchmarkResult]:
    """Run full benchmark suite and return results."""
    config = BenchmarkConfig(
        warmup_iters=10,
        measure_iters=50,
        min_time_s=0.5,
        verbose=True,
        output_dir=output_dir,
    )
    suite = BenchmarkSuite(config)
    return suite.run_full_suite()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SNEPPX Benchmark Suite")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=50, help="Measure iterations"
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        verbose=not args.quiet,
        output_dir=args.output,
        measure_iters=args.iterations,
    )

    suite = BenchmarkSuite(config)
    suite.run_full_suite()
    suite.print_summary()
