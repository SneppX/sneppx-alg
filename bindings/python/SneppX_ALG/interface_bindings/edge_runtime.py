"""Mobile / edge runtime — device detection, backends, INT8 inference.

A lightweight inference runtime for resource-constrained targets: detects the
host CPU (architecture + ISA flags), executes compiled graphs through pluggable
backends, runs INT8-quantized matmuls with fp32 accumulation, pools scratch
buffers to avoid reallocation, and reports latency/flops. The NPU/ARM delegate
slot is an explicit extension point (``register_backend``).

Typical usage::

    compiled = GraphCompiler().compile(model)
    runtime = EdgeRuntime(compiled, EdgeConfig(quant_mode="int8"))
    y = runtime({"x": x})              # or runtime.forward({"x": x})
    ms = runtime.latency_ms({"x": x})
"""

import math
import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .graph_compiler import CompiledGraph, FusedNode, GraphNode, _eval_fused

__all__ = [
    "EdgeDevice",
    "EdgeConfig",
    "EdgeBackend",
    "CPUBackend",
    "EdgeRuntime",
    "detect_cpu",
    "register_backend",
    "quantize_graph_weights",
    "benchmark",
]


# --------------------------------------------------------------------------
# device detection
# --------------------------------------------------------------------------

_ARCH_ALIASES = {
    "amd64": "x86_64",
    "x64": "x86_64",
    "i386": "x86",
    "i686": "x86",
    "arm64": "aarch64",
    "armv7l": "armv7",
    "armv7": "armv7",
    "aarch64": "aarch64",
    "riscv64": "riscv64",
}

_ARCH_DEFAULT_ISA = {
    "x86_64": {"sse2", "sse3", "ssse3"},
    "x86": {"sse2", "sse3"},
    "aarch64": {"neon", "neon_fp16", "aes", "dotprod"},
    "armv7": {"neon"},
    "riscv64": {"rvv"},
    "unknown": set(),
}

_ARM_VECTOR_LENGTH = {
    "aarch64": 16,
    "armv7": 16,
    "riscv64": 16,
}


@dataclass
class EdgeDevice:
    """Detected host CPU capabilities for a mobile/edge target."""

    architecture: str
    isa: Set[str] = field(default_factory=set)
    cores: int = 1
    vendor: str = "unknown"
    vector_width_bytes: int = 16
    family: str = "cpu"

    @property
    def supports_int8(self) -> bool:
        return True

    @property
    def supports_fp16(self) -> bool:
        return "neon_fp16" in self.isa or "fp16" in self.isa

    def __repr__(self) -> str:
        return (f"EdgeDevice({self.architecture}, cores={self.cores}, "
                f"isa={sorted(self.isa)})")


def _read_cpuinfo_flags() -> Set[str]:
    """Best-effort ISA flags from /proc/cpuinfo (Linux/Android)."""
    path = "/proc/cpuinfo"
    flags: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    flags |= set(line.split(":", 1)[1].split())
                elif line.lower().startswith("features"):
                    flags |= set(line.split(":", 1)[1].split())
    except OSError:
        return flags
    return flags


def detect_cpu() -> EdgeDevice:
    """Detect the current host as an EdgeDevice."""
    raw = (platform.machine() or "").lower()
    arch = _ARCH_ALIASES.get(raw, raw or "unknown")
    isa = set(_ARCH_DEFAULT_ISA.get(arch, _ARCH_DEFAULT_ISA["unknown"]))
    flags = _read_cpuinfo_flags()
    if arch == "x86_64":
        if "avx512f" in flags:
            isa |= {"avx2", "avx512f"}
        elif "avx2" in flags:
            isa |= {"avx", "avx2"}
        elif "avx" in flags:
            isa |= {"avx"}
        elif "sse4_2" in flags:
            isa |= {"sse4_2"}
    elif arch == "aarch64":
        if "asimddp" in flags or "dotprod" in flags:
            isa |= {"dotprod"}
    try:
        cores = os.cpu_count() or 1
    except Exception:  # pragma: no cover
        cores = 1
    vw = _ARM_VECTOR_LENGTH.get(arch, 16)
    if arch == "x86_64" and "avx512f" in isa:
        vw = 64
    elif arch == "x86_64" and "avx2" in isa:
        vw = 32
    vendor = (platform.processor() or "unknown").split()[0] if platform.processor() else "unknown"
    return EdgeDevice(
        architecture=arch,
        isa=isa,
        cores=cores,
        vendor=vendor,
        vector_width_bytes=vw,
    )


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class EdgeConfig:
    """Runtime configuration for a mobile/edge target."""

    backend: str = "auto"          # "auto" | "cpu" | backend name
    threads: int = 1
    quant_mode: str = "none"       # "none" | "int8" | "int4"
    buffer_pool: bool = True       # reuse scratch buffers across calls
    fp32_fallback: bool = True
    max_batch: int = 512
    int8_accumulator: str = "fp32"  # fp32 | int32 accumulation
    quant_error_tol: float = 5e-2  # rel tolerance for quantized parity checks


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class EdgeBackend:
    """Base class for edge execution backends.

    A backend declares which node ops it can execute and how. Concrete
    backends are registered on EdgeRuntime via ``register_backend``.
    """

    name = "base"

    def supports(self, node: Any) -> bool:
        raise NotImplementedError

    def run(self, node: Any, inputs: List[np.ndarray],
            pool: "BufferPool", **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<EdgeBackend {self.name}>"


class BufferPool:
    """Reusable scratch-buffer pool for steady-state mobile inference.

    Buffers are recycled *between* forward calls (never within one), so node
    outputs are never corrupted mid-graph. Each forward starts a fresh epoch
    of fresh arrays that are returned to the free list at the end.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._free: Dict[Tuple, List[np.ndarray]] = {}
        self._outstanding: List[np.ndarray] = []

    def begin(self) -> None:
        self._outstanding = []

    def acquire(self, shape: Tuple[int, ...], dtype: Any) -> np.ndarray:
        if not self.enabled:
            return np.empty(shape, dtype=dtype)
        key = (shape, np.dtype(dtype).str)
        free = self._free.get(key)
        if free:
            buf = free.pop()
        else:
            buf = np.empty(shape, dtype=dtype)
        self._outstanding.append(buf)
        return buf

    def owns(self, buf: np.ndarray) -> bool:
        return any(b is buf for b in self._outstanding)

    def end(self) -> None:
        if not self.enabled:
            return
        for buf in self._outstanding:
            key = (buf.shape, buf.dtype.str)
            self._free.setdefault(key, []).append(buf)
        self._outstanding = []

    def release(self) -> None:
        self._free.clear()
        self._outstanding = []

    def __len__(self) -> int:
        if not self.enabled:
            return 0
        return sum(len(v) for v in self._free.values())


def int8_matmul(a: np.ndarray, b: np.ndarray, scale_b: float,
                accumulator: str = "fp32") -> np.ndarray:
    """INT8 symmetric matmul: A @ B with fp32 or int32 accumulation.

    ``b`` is the pre-quantized INT8 weight tensor (K, N); ``scale_b`` its
    per-tensor scale. Input activations are quantized per-call so the result
    matches ``A @ dequantize(B)`` within quantization tolerance.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.int8)
    peak = float(np.abs(a).max()) if a.size else 0.0
    scale_a = peak / 127.0 if peak > 0.0 else 1.0
    aq = np.round(a / scale_a).astype(np.int8) if peak > 0.0 else \
        np.zeros(a.shape, dtype=np.int8)
    if accumulator == "int32":
        acc = aq.astype(np.int32) @ b.astype(np.int32)
        return acc.astype(np.float32) * (scale_a * scale_b)
    acc = aq.astype(np.float32) @ b.astype(np.float32)
    return acc * (scale_a * scale_b)


class CPUBackend(EdgeBackend):
    """Reference numpy backend: executes every node, optionally INT8 matmuls."""

    name = "cpu"

    def __init__(self, quant_mode: str = "none",
                 quant_weights: Optional[Dict[int, Tuple[np.ndarray, float]]] = None,
                 int8_accumulator: str = "fp32"):
        self.quant_mode = quant_mode
        self.quant_weights = quant_weights or {}
        self.int8_accumulator = int8_accumulator

    def supports(self, node: Any) -> bool:
        return True

    def run(self, node: Any, inputs: List[np.ndarray],
            pool: BufferPool, **kwargs: Any) -> np.ndarray:
        if node.op == "matmul":
            a, b = inputs
            out_shape = a.shape[:-1] + b.shape[-1:]
            if id(node) in self.quant_weights:
                w, scale = self.quant_weights[id(node)]
                acc = int8_matmul(a, w, scale, self.int8_accumulator)
                out = pool.acquire(acc.shape, acc.dtype)
                out[:] = acc
                return out
            out = pool.acquire(out_shape, np.result_type(a, b))
            np.matmul(a, b, out=out)
            return out
        if node.op == "clip":
            return np.clip(inputs[0], node.params["min"], node.params["max"])
        return node.fn(*inputs)


# --------------------------------------------------------------------------
# graph weight quantization
# --------------------------------------------------------------------------

def _int8_weight_scale(w: np.ndarray) -> Tuple[np.ndarray, float]:
    peak = float(np.abs(w).max()) if w.size else 0.0
    if peak == 0.0:
        return np.zeros(w.shape, dtype=np.int8), 1.0
    scale = peak / 127.0
    return np.round(w / scale).astype(np.int8), scale


def quantize_graph_weights(compiled: CompiledGraph,
                           mode: str = "int8") -> Dict[int, Tuple[np.ndarray, float]]:
    """Quantize matmul weight tensors (const/param RHS) to INT8.

    Returns a mapping of node id -> (int8_array, scale) for every matmul whose
    second operand is a constant/parameter array. Other nodes are left float.
    """
    if mode not in ("int8",):
        raise ValueError(f"unsupported edge quant mode: {mode!r}")
    out: Dict[int, Tuple[np.ndarray, float]] = {}
    for node in compiled.nodes:
        if node.op != "matmul":
            continue
        rhs = node.inputs[1]
        if rhs.op not in ("const", "param"):
            continue
        val = np.asarray(rhs.params["value"], dtype=np.float32)
        if val.ndim < 2:
            continue
        w, scale = _int8_weight_scale(val)
        out[id(node)] = (w, scale)
    return out


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------

_BACKEND_REGISTRY: Dict[str, Callable[..., EdgeBackend]] = {
    "cpu": lambda **kw: CPUBackend(**kw),
}


def register_backend(name: str, factory: Callable[..., EdgeBackend]) -> None:
    """Register a custom edge backend factory under ``name``."""
    if not name or not callable(factory):
        raise ValueError("backend name and factory callable required")
    _BACKEND_REGISTRY[name] = factory


class EdgeRuntime:
    """Mobile/edge inference runtime over a compiled graph."""

    def __init__(self, compiled: CompiledGraph, config: Optional[EdgeConfig] = None,
                 device: Optional[EdgeDevice] = None, quant_weights: Optional[Dict] = None):
        self.compiled = compiled
        self.config = config or EdgeConfig()
        self.device = device or detect_cpu()
        self.pool = BufferPool(enabled=self.config.buffer_pool)
        name = self.config.backend
        if name == "auto":
            name = "cpu"
        if name not in _BACKEND_REGISTRY:
            raise ValueError(f"unknown edge backend {name!r}; "
                             f"registered: {sorted(_BACKEND_REGISTRY)}")
        self.quant_weights = quant_weights
        if self.quant_weights is None and self.config.quant_mode != "none":
            self.quant_weights = quantize_graph_weights(self.compiled, self.config.quant_mode)
        self.backend = _BACKEND_REGISTRY[name](
            quant_mode=self.config.quant_mode,
            quant_weights=self.quant_weights or {},
            int8_accumulator=self.config.int8_accumulator,
        )
        self._fallback = _BACKEND_REGISTRY["cpu"](
            quant_mode=self.config.quant_mode,
            quant_weights=self.quant_weights or {},
            int8_accumulator=self.config.int8_accumulator,
        )

    # ---- public API ------------------------------------------------------
    def __call__(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: Dict[str, np.ndarray],
                tile_size: Optional[int] = None) -> np.ndarray:
        """Execute the compiled graph through the configured backend."""
        values: Dict[int, np.ndarray] = {}
        bindings = {k: np.asarray(v) for k, v in inputs.items()}
        self.pool.begin()
        try:
            for node in self.compiled.nodes:
                if node.op == "input":
                    values[id(node)] = bindings.get(
                        node.name, bindings.get(id(node))
                    )
                elif node.op == "const":
                    values[id(node)] = node.params["value"]
                elif node.op == "param":
                    values[id(node)] = node.params["value"]
                elif isinstance(node, FusedNode) and tile_size is not None:
                    values[id(node)] = _eval_fused(node, values, tile_size)
                else:
                    if isinstance(node, FusedNode):
                        args = [values[id(i)] for i in node.external_inputs]
                    else:
                        args = [values[id(self.compiled._resolve(i))] for i in node.inputs]
                    backend = self.backend if self.backend.supports(node) else self._fallback
                    values[id(node)] = backend.run(node, args, self.pool)
            result = values[id(self.compiled.entry)]
            if self.pool.owns(result):
                result = np.array(result, copy=True)
            return result
        finally:
            self.pool.end()

    def predict(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.forward(inputs)

    def latency_ms(self, inputs: Dict[str, np.ndarray], iterations: int = 20,
                   warmup: int = 3) -> float:
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        for _ in range(warmup):
            self.forward(inputs)
        best = math.inf
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.forward(inputs)
            best = min(best, time.perf_counter() - t0)
        return best * 1e3

    def throughput(self, inputs: Dict[str, np.ndarray], iterations: int = 20) -> float:
        ms = self.latency_ms(inputs, iterations=iterations)
        return 1000.0 / ms if ms > 0 else math.inf

    def flops(self) -> int:
        total = 0
        for node in self.compiled.nodes:
            if node.op == "matmul":
                a = node.inputs[0]
                b = node.inputs[1]
                if a.op == "input" and b.op in ("const", "param"):
                    val = np.asarray(b.params["value"])
                    total += 2 * int(np.prod(val.shape))
                else:
                    total += 2 * 1024
            elif isinstance(node, FusedNode):
                total += 8 * len(node.members)
        return total

    def summary(self) -> Dict[str, Any]:
        quant = {k: v[1] for k, v in (self.quant_weights or {}).items()}
        return {
            "device": self.device.architecture,
            "isa": sorted(self.device.isa),
            "cores": self.device.cores,
            "backend": self.backend.name,
            "quant_mode": self.config.quant_mode,
            "quantized_matmul_layers": len(self.quant_weights or {}),
            "scratch_buffers": len(self.pool),
            "estimated_flops": self.flops(),
        }

    def close(self) -> None:
        self.pool.release()


def benchmark(runtime: EdgeRuntime, inputs: Dict[str, np.ndarray],
              iterations: int = 20) -> Dict[str, Any]:
    """Run a one-shot benchmark over the runtime."""
    return {
        "latency_ms": runtime.latency_ms(inputs, iterations=iterations),
        "throughput_ops": runtime.throughput(inputs, iterations=iterations),
        "flops": runtime.flops(),
        **runtime.summary(),
    }
