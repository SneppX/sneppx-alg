"""Benchmarking suite for SNEPPX operations."""

import time
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
import os

from .tensor import Tensor

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

    def _measure(self, fn: Callable, *args, **kwargs) -> Tuple[List[float], float]:
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

        def _matmul(a, b):
            return a @ b

        results = []
        for M, N, K in shapes:
            A = Tensor.from_numpy(np.random.randn(M, K).astype(np.float32))
            B = Tensor.from_numpy(np.random.randn(K, N).astype(np.float32))
            mem_mb = (M * K + K * N + M * N) * 4 / (1024 * 1024)

            result = self.benchmark(
                f"matmul_{M}x{N}x{K}", "matmul", lambda: _matmul(A, B), memory_mb=mem_mb
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
                f"attention_B{B}_L{L}_D{D}_H{H}",
                "attention",
                lambda: multi_head_attention(
                    q, k, v, num_heads=H, dropout_p=0.1, training=True
                ),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_rnn(
        self, configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark RNN variants (LSTM, GRU, vanilla RNN)."""
        if configs is None:
            configs = [
                {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "lstm"},
                {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "gru"},
                {"batch": 1, "seq": 32, "input": 64, "hidden": 128, "type": "rnn"},
            ]

        from .advanced_ops import rnn_cell, lstm_cell, gru_cell

        results = []
        for cfg in configs:
            B, S, I, H = cfg["batch"], cfg["seq"], cfg["input"], cfg["hidden"]
            x = Tensor.from_numpy(np.random.randn(B, S, I).astype(np.float32))
            h0 = Tensor.from_numpy(np.random.randn(B, H).astype(np.float32))
            c0 = Tensor.from_numpy(np.random.randn(B, H).astype(np.float32))

            if cfg["type"] == "lstm":
                fn = lambda: lstm_cell(
                    x[:, 0, :], h0, c0,
                    Tensor.from_numpy(np.random.randn(4 * H, I).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(4 * H, H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(4 * H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(4 * H).astype(np.float32)),
                )
            elif cfg["type"] == "gru":
                fn = lambda: gru_cell(
                    x[:, 0, :], h0,
                    Tensor.from_numpy(np.random.randn(3 * H, I).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(3 * H, H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(3 * H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(3 * H).astype(np.float32)),
                )
            else:
                fn = lambda: rnn_cell(
                    x[:, 0, :], h0,
                    Tensor.from_numpy(np.random.randn(H, I).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(H, H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(H).astype(np.float32)),
                    Tensor.from_numpy(np.random.randn(H).astype(np.float32)),
                )

            mem_mb = (B * S * I + B * H + I * H + H * H) * 4 / (1024 * 1024)

            result = self.benchmark(
                f"rnn_{cfg['type']}_B{B}_S{S}_I{I}_H{H}", "rnn", fn, memory_mb=mem_mb
            )
            results.append(result)
        return results

    def benchmark_transformer_block(
        self, configs: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark transformer encoder block."""
        if configs is None:
            configs = [
                {
                    "batch": 1,
                    "seq": 64,
                    "dim": 256,
                    "heads": 4,
                    "ff_mult": 4,
                    "dropout": 0.1,
                },
                {
                    "batch": 2,
                    "seq": 128,
                    "dim": 512,
                    "heads": 8,
                    "ff_mult": 4,
                    "dropout": 0.1,
                },
                {
                    "batch": 4,
                    "seq": 256,
                    "dim": 1024,
                    "heads": 16,
                    "ff_mult": 4,
                    "dropout": 0.1,
                },
            ]

        from .advanced_ops import transformer_block as _tf_block

        results = []
        for cfg in configs:
            B, L, D, H = cfg["batch"], cfg["seq"], cfg["dim"], cfg["heads"]
            ff_dim = D * cfg.get("ff_mult", 4)
            x = Tensor.from_numpy(np.random.randn(B, L, D).astype(np.float32))
            attn_w = Tensor.from_numpy(np.random.randn(3 * D, D).astype(np.float32))
            attn_b = Tensor.from_numpy(np.random.randn(3 * D).astype(np.float32))
            ff1_w = Tensor.from_numpy(np.random.randn(ff_dim, D).astype(np.float32))
            ff1_b = Tensor.from_numpy(np.random.randn(ff_dim).astype(np.float32))
            ff2_w = Tensor.from_numpy(np.random.randn(D, ff_dim).astype(np.float32))
            ff2_b = Tensor.from_numpy(np.random.randn(D).astype(np.float32))
            ln1_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
            ln1_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))
            ln2_w = Tensor.from_numpy(np.ones(D, dtype=np.float32))
            ln2_b = Tensor.from_numpy(np.zeros(D, dtype=np.float32))

            mem_mb = (
                B * L * D * 4 + B * L * L * 4 + B * L * ff_dim * 4
            ) / (1024 * 1024)

            result = self.benchmark(
                f"transformer_B{B}_L{L}_D{D}_H{H}",
                "transformer",
                lambda: _tf_block(
                    x, attn_w, attn_b, ff1_w, ff1_b, ff2_w, ff2_b,
                    ln1_w, ln1_b, ln2_w, ln2_b,
                    num_heads=H, dropout=cfg.get("dropout", 0.1),
                ),
                memory_mb=mem_mb,
            )
            results.append(result)
        return results

    def benchmark_activation(
        self, activations: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark activation functions."""
        if activations is None:
            activations = [
                "relu",
                "gelu",
                "silu",
                "tanh",
                "sigmoid",
                "relu6",
                "hardswish",
                "leaky_relu",
                "elu",
                "selu",
                "softplus",
                "softsign",
                "mish",
            ]

        from .advanced_ops import (
            relu,
            gelu,
            silu,
            tanh,
            sigmoid,
            relu6,
            hardswish,
            leaky_relu,
            elu,
            selu,
            softplus,
            softsign,
            mish,
        )

        act_fns = {
            "relu": lambda x: relu(x),
            "gelu": lambda x: gelu(x),
            "silu": lambda x: silu(x),
            "tanh": lambda x: tanh(x),
            "sigmoid": lambda x: sigmoid(x),
            "relu6": lambda x: relu6(x),
            "hardswish": lambda x: hardswish(x),
            "leaky_relu": lambda x: leaky_relu(x),
            "elu": lambda x: elu(x),
            "selu": lambda x: selu(x),
            "softplus": lambda x: softplus(x),
            "softsign": lambda x: softsign(x),
            "mish": lambda x: mish(x),
        }

        results = []
        for name in activations:
            x = Tensor.from_numpy(np.random.randn(1000, 1000).astype(np.float32))
            result = self.benchmark(
                f"activation_{name}", "activation", act_fns[name], memory_mb=4.0
            )
            results.append(result)
        return results

    def benchmark_reduction(
        self, shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark reduction operations."""
        if shapes is None:
            shapes = [(1000, 1000), (100, 10000), (10, 100000), (1000, 1000, 10)]

        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))

            for op_name, fn in [
                ("sum", lambda: x.sum()),
                ("mean", lambda: x.mean()),
                ("var", lambda: x.var()),
                ("std", lambda: x.std()),
                ("max", lambda: x.max()),
                ("min", lambda: x.min()),
                ("prod", lambda: x.prod()),
                ("norm", lambda: x.norm()),
            ]:
                mem_mb = np.prod(shape) * 4 / (1024 * 1024)
                result = self.benchmark(
                    f"{op_name}_{shape}", "reduction", fn, memory_mb=mem_mb
                )
                results.append(result)
        return results

    def benchmark_pointwise(
        self, shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark pointwise operations."""
        if shapes is None:
            shapes = [(1000, 1000), (100, 10000), (10, 100000)]

        from .advanced_ops import add, mul, div, pow

        results = []
        for shape in shapes:
            a = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            b = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            mem_mb = shape[0] * shape[1] * 4 * 3 / (1024 * 1024)

            for op_name, fn in [
                ("add", lambda: add(a, b)),
                ("mul", lambda: mul(a, b)),
                ("div", lambda: div(a, b)),
                ("pow", lambda: pow(a, b)),
            ]:
                result = self.benchmark(
                    f"{op_name}_{shape}", "pointwise", fn, memory_mb=mem_mb
                )
                results.append(result)
        return results

    def benchmark_norm(
        self, shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark normalization operations."""
        if shapes is None:
            shapes = [(1000, 1000), (100, 10000), (10, 100000)]

        from .advanced_ops import (
            layernorm,
            rmsnorm,
            batch_norm,
            group_norm,
            instance_norm,
            layer_norm,
        )

        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            normalized_shape = shape[-1] if len(shape) > 1 else shape[0]
            w = Tensor.from_numpy(np.ones(normalized_shape, dtype=np.float32))
            b = Tensor.from_numpy(np.zeros(normalized_shape, dtype=np.float32))
            mem_mb = np.prod(shape) * 4 * 4 / (1024 * 1024)

            for op_name, fn in [
                ("layernorm", lambda: layernorm(x, w, b)),
                ("rmsnorm", lambda: rmsnorm(x, w)),
                ("batch_norm", lambda: batch_norm(x, w, b)),
                ("group_norm", lambda: group_norm(x, 32, w, b)),
                ("instance_norm", lambda: instance_norm(x, w, b)),
                ("layer_norm", lambda: layer_norm(x, (normalized_shape,), w, b)),
            ]:
                result = self.benchmark(
                    f"{op_name}_{shape}", "normalization", fn, memory_mb=mem_mb
                )
                results.append(result)
        return results

    def benchmark_pooling(
        self, shapes: Optional[List[Tuple[int, ...]]] = None
    ) -> List[BenchmarkResult]:
        """Benchmark pooling operations."""
        if shapes is None:
            shapes = [(1, 64, 224, 224), (4, 128, 56, 56), (8, 256, 28, 28)]

        from .advanced_ops import (
            max_pool2d,
            avg_pool2d,
            adaptive_avg_pool2d,
            adaptive_max_pool2d,
        )

        results = []
        for shape in shapes:
            x = Tensor.from_numpy(np.random.randn(*shape).astype(np.float32))
            mem_mb = np.prod(shape) * 4 * 3 / (1024 * 1024)

            for op_name, fn in [
                ("max_pool2d", lambda: max_pool2d(x, (3, 3), (2, 2), (1, 1))),
                ("avg_pool2d", lambda: avg_pool2d(x, (3, 3), (2, 2), (1, 1))),
                ("adaptive_avg_pool", lambda: adaptive_avg_pool2d(x, (7, 7))),
                ("adaptive_max_pool", lambda: adaptive_max_pool2d(x, (7, 7))),
            ]:
                result = self.benchmark(
                    f"{op_name}_{shape}", "pooling", fn, memory_mb=mem_mb
                )
                results.append(result)
        return results

    def save_results(
        self, output_dir: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Save benchmark results to JSON and CSV."""
        if not self.results:
            return None, None

        output_dir = output_dir or self.config.output_dir or tempfile.gettempdir()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
        csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")

        # Save JSON
        if self.config.save_json:
            data = {
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
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

        # Save CSV
        if self.config.save_csv:
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
                        ]
                    )

        return json_path if self.config.save_json else None, (
            csv_path if self.config.save_csv else None
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all built-in benchmarks."""
        print("Running all benchmarks...")

        self.benchmark_matmul()
        self.benchmark_conv2d()
        self.benchmark_attention()
        self.benchmark_rnn()
        self.benchmark_transformer_block()
        self.benchmark_activation()
        self.benchmark_reduction()
        self.benchmark_pointwise()
        self.benchmark_norm()
        self.benchmark_pooling()

        print(f"\nCompleted {len(self.results)} benchmarks")
        return self.results


def run_benchmarks(
    config: Optional[BenchmarkConfig] = None,
    output_dir: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> Tuple[List[BenchmarkResult], Optional[str], Optional[str]]:
    """Convenience function to run benchmarks and save results."""
    suite = BenchmarkSuite(config)
    results = suite.run_all()
    json_path, csv_path = suite.save_results(output_dir)
    return results, json_path, csv_path


from typing import Optional, Dict, List, Any, Callable, Tuple, Union, List
from dataclasses import dataclass, field
import time
import numpy as np
import sys
import platform
from pathlib import Path
import tempfile
