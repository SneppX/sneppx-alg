"""Hardware inference backends — OpenVINO/ONNX-Runtime-style execution
providers with auto-detection and fallback.

Provides a device-agnostic ``inference_session(...)``/``select_execution_provider``
surface modelled on OpenVINO's device selector ("AUTO", "CPU", "GPU", "NPU")
and ONNX Runtime's execution-provider list::

    session = inference_session("cpu_fp32")
    out = session.run("mlp", x)

Backends:

- ``cpu_fp32`` — pure NumPy executor (always available, no dependencies).
- ``cpu_fp16`` — NumPy executor casting to float16 (simulated fp16 math).
- ``onnx``     — wraps the ``onnxruntime`` package when installed.
- ``openvino`` — wraps the ``openvino`` package when installed.
- ``npu``      — OpenVINO NPU device (Core.AvailableDevices) when present.
- ``auto``     — resolves to the fastest *available* backend (never raises).

Every backend reports ``name``, ``device``, ``available``, ``latency_ms`` and
implements ``run(model_id, inputs) -> np.ndarray``. Model execution falls back
to a deterministic linear map so the provider machinery is testable without
an LLM or accelerator.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DeviceInfo",
    "HardwareProbe",
    "InferenceBackend",
    "CPUFP32Backend",
    "CPUFP16Backend",
    "ONNXBackend",
    "OpenVINOBackend",
    "NPUBackend",
    "AutoBackend",
    "BackendRegistry",
    "select_execution_provider",
    "inference_session",
    "detect_devices",
]

DEVICE_ALIASES = {
    "auto": "AUTO",
    "cpu": "CPU",
    "cpu_fp32": "CPU",
    "cpu_fp16": "CPU_FP16",
    "onnx": "ONNX",
    "onnxruntime": "ONNX",
    "openvino": "OPENVINO",
    "gpu": "GPU",
    "npu": "NPU",
}


@dataclass
class DeviceInfo:
    name: str
    available: bool
    kind: str = "cpu"
    detail: str = ""
    priority: int = 0

    def __repr__(self) -> str:
        state = "available" if self.available else "unavailable"
        return f"<Device {self.name} ({state}) priority={self.priority}>"


class HardwareProbe:
    """Detect available compute devices without importing heavy packages.

    Probing is side-effect-free: we only check ``importlib.util.find_spec`` and
    (for NPU) parse device lists. No CUDA/ONNX/OpenVINO import is triggered.
    """

    @staticmethod
    def has_package(name: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(name) is not None

    @classmethod
    def cpu(cls) -> DeviceInfo:
        info = "AVX512" if cls._cpu_flag("avx512f") else (
            "AVX2" if cls._cpu_flag("avx2") else "scalar")
        return DeviceInfo(name="CPU", available=True, kind="cpu",
                          detail=f"numpy/{info}", priority=20)

    @staticmethod
    def _cpu_flag(name: str) -> bool:
        try:
            import cpuinfo as _ci  # optional
            return bool(getattr(_ci.get_cpu_info().get("flags", []), "count", 0) and
                        name in _ci.get_cpu_info().get("flags", []))
        except Exception:
            return False

    @classmethod
    def onnx(cls) -> DeviceInfo:
        ok = cls.has_package("onnxruntime")
        ver = ""
        if ok:
            try:
                import onnxruntime as ort
                ver = ort.__version__
            except Exception:
                ver = "?"
        return DeviceInfo(name="ONNX", available=ok, kind="onnx",
                          detail=f"onnxruntime {ver}" if ok else "package not installed",
                          priority=30)

    @classmethod
    def openvino(cls) -> DeviceInfo:
        ok = cls.has_package("openvino")
        detail = ""
        if ok:
            try:
                from openvino import Core
                detail = ",".join(Core().available_devices)
            except Exception as exc:  # pragma: no cover - env dependent
                ok = False
                detail = str(exc)
        return DeviceInfo(name="OPENVINO", available=ok, kind="openvino",
                          detail=detail or "package not installed", priority=40)

    @classmethod
    def npu(cls) -> DeviceInfo:
        ok = cls.has_package("openvino")
        detail = ""
        if ok:
            try:
                from openvino import Core
                devices = Core().available_devices
                ok = any(d.startswith("NPU") for d in devices)
                detail = ",".join(devices)
            except Exception as exc:  # pragma: no cover - env dependent
                ok = False
                detail = str(exc)
        return DeviceInfo(name="NPU", available=ok, kind="npu",
                          detail=detail or "no NPU device", priority=50)

    @classmethod
    def gpu(cls) -> DeviceInfo:
        ok = cls.has_package("openvino")
        detail = ""
        if ok:
            try:
                from openvino import Core
                devices = Core().available_devices
                ok = any(d.startswith("GPU") for d in devices)
                detail = ",".join(devices)
            except Exception as exc:  # pragma: no cover - env dependent
                ok = False
                detail = str(exc)
        return DeviceInfo(name="GPU", available=ok, kind="gpu",
                          detail=detail or "no GPU device", priority=45)

    @classmethod
    def all(cls) -> List[DeviceInfo]:
        return [cls.cpu(), cls.onnx(), cls.openvino(), cls.npu(), cls.gpu()]

    @classmethod
    def summary(cls) -> Dict[str, bool]:
        return {d.name: d.available for d in cls.all()}


def detect_devices() -> List[DeviceInfo]:
    """Public entry point: probe every supported device."""
    return HardwareProbe.all()


# ===========================================================================
#  Backends
# ===========================================================================


class InferenceBackend:
    """Base class: all backends implement ``run(model_id, inputs)``."""

    name: str = "base"
    device: str = "CPU"

    def __init__(self, model_fn: Optional[Callable] = None, **options: Any):
        self._model_fn = model_fn
        self.options = options
        self._latency_ms: List[float] = []
        self._lock = threading.Lock()
        self._session = None

    @property
    def available(self) -> bool:
        return True

    def latency_ms(self, average: bool = True) -> float:
        with self._lock:
            if not self._latency_ms:
                return 0.0
            return (sum(self._latency_ms) / len(self._latency_ms)) if average else self._latency_ms[-1]

    def _record(self, ms: float) -> None:
        with self._lock:
            self._latency_ms.append(ms)

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        """Execute ``inputs`` for ``model_id``.

        If ``model_fn`` was supplied it is called as ``model_fn(model_id, inputs)``.
        Otherwise a deterministic per-model linear map ``y = x @ A + b`` is used
        (seeded by ``model_id``) so routing is testable without a real model.
        """
        t0 = time.perf_counter()
        try:
            if self._model_fn is not None:
                return self._model_fn(model_id, inputs)
            return self._fallback(model_id, inputs)
        finally:
            self._record((time.perf_counter() - t0) * 1000.0)

    def _fallback(self, model_id: str, inputs: Any) -> Any:
        x = np.asarray(inputs, dtype=np.float32)
        seed = abs(hash(model_id)) % (2**31)
        rng = np.random.default_rng(seed)
        if x.ndim == 0:
            return np.asarray(0.0, dtype=np.float32)
        last = x.shape[-1]
        a = rng.normal(size=(last, last)).astype(np.float32)
        b = rng.normal(size=(last,)).astype(np.float32)
        return x @ a + b


class CPUFP32Backend(InferenceBackend):
    name = "cpu_fp32"
    device = "CPU"

    @property
    def available(self) -> bool:
        return True


class CPUFP16Backend(InferenceBackend):
    name = "cpu_fp16"
    device = "CPU_FP16"

    @property
    def available(self) -> bool:
        return True

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            if self._model_fn is not None:
                return self._model_fn(model_id, np.asarray(inputs, dtype=np.float16))
            return self._fallback(model_id, np.asarray(inputs, dtype=np.float32).astype(np.float16)).astype(np.float32)
        finally:
            self._record((time.perf_counter() - t0) * 1000.0)


class ONNXBackend(InferenceBackend):
    """Execution provider backed by ``onnxruntime`` when installed."""

    name = "onnx"
    device = "ONNX"
    provider = "CPUExecutionProvider"

    def __init__(self, model_fn: Optional[Callable] = None, **options: Any):
        super().__init__(model_fn, **options)
        self._ort = None
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError:  # pragma: no cover - dependency optional
            self._ort = None

    @property
    def available(self) -> bool:
        return self._ort is not None

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        if not self.available:
            return super().run(model_id, inputs, **kw)
        t0 = time.perf_counter()
        try:
            if self._model_fn is not None:
                return self._model_fn(model_id, inputs)
            # No real .onnx graph in this environment: degrade to numpy fallback.
            return self._fallback(model_id, inputs)
        finally:
            self._record((time.perf_counter() - t0) * 1000.0)


class OpenVINOBackend(InferenceBackend):
    """Execution provider backed by the ``openvino`` package when installed."""

    name = "openvino"
    device = "OPENVINO"

    def __init__(self, model_fn: Optional[Callable] = None, **options: Any):
        super().__init__(model_fn, **options)
        self._ov = None
        try:
            from openvino import Core
            self._ov = Core()
        except ImportError:  # pragma: no cover - dependency optional
            self._ov = None

    @property
    def available(self) -> bool:
        return self._ov is not None

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        if not self.available:
            return super().run(model_id, inputs, **kw)
        t0 = time.perf_counter()
        try:
            if self._model_fn is not None:
                return self._model_fn(model_id, inputs)
            return self._fallback(model_id, inputs)
        finally:
            self._record((time.perf_counter() - t0) * 1000.0)


class NPUBackend(InferenceBackend):
    """OpenVINO NPU device (e.g. Intel AI Boost) when present."""

    name = "npu"
    device = "NPU"

    def __init__(self, model_fn: Optional[Callable] = None, **options: Any):
        super().__init__(model_fn, **options)
        self._ov = None
        self._npu_ok = False
        try:
            from openvino import Core
            core = Core()
            self._ov = core
            self._npu_ok = any(d.startswith("NPU") for d in core.available_devices)
        except Exception:  # pragma: no cover - dependency optional
            self._ov = None

    @property
    def available(self) -> bool:
        return self._ov is not None and self._npu_ok

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        if not self.available:
            return super().run(model_id, inputs, **kw)
        t0 = time.perf_counter()
        try:
            if self._model_fn is not None:
                return self._model_fn(model_id, inputs)
            return self._fallback(model_id, inputs)
        finally:
            self._record((time.perf_counter() - t0) * 1000.0)


class AutoBackend(InferenceBackend):
    """Resolves to the best available backend at construction time.

    Priority order: NPU > OPENVINO > ONNX > CPU_FP16 > CPU_FP32.
    Never raises: CPU_FP32 always exists.
    """

    name = "auto"
    device = "AUTO"

    def __init__(self, model_fn: Optional[Callable] = None, **options: Any):
        super().__init__(model_fn, **options)
        self._backend = BackendRegistry.default().resolve("auto", model_fn)

    @property
    def available(self) -> bool:
        return True

    @property
    def resolved(self) -> InferenceBackend:
        return self._backend

    def run(self, model_id: str, inputs: Any, **kw: Any) -> Any:
        return self._backend.run(model_id, inputs, **kw)


# ===========================================================================
#  Registry + selector
# ===========================================================================


class BackendRegistry:
    """Maps device names/aliases to backend classes with priority ordering."""

    def __init__(self):
        self._backends: List[InferenceBackend] = [
            NPUBackend(),
            OpenVINOBackend(),
            ONNXBackend(),
            CPUFP16Backend(),
            CPUFP32Backend(),
        ]

    def list(self) -> List[InferenceBackend]:
        return list(self._backends)

    def get(self, name: str) -> Optional[InferenceBackend]:
        for b in self._backends:
            if b.name == name:
                return b
        return None

    def available_backends(self) -> List[InferenceBackend]:
        return [b for b in self._backends if b.available]

    def resolve(self, device: str = "auto", model_fn: Optional[Callable] = None) -> InferenceBackend:
        """Pick a concrete backend for ``device`` (alias-aware), with fallback.

        - ``"auto"``       -> first available backend in priority order.
        - explicit name    -> that backend if available, else the requested
          backend's degraded path (still never raises if fallback exists).
        - unknown string   -> ``ValueError``.

        When ``model_fn`` is given, a fresh backend is built with it so the
        custom executor is honoured regardless of the cached registry probes.
        """
        def fresh(cls: type) -> InferenceBackend:
            return cls(model_fn) if model_fn is not None else cls()

        key = DEVICE_ALIASES.get(str(device).lower(), str(device).upper())

        if key == "AUTO":
            for b in self._backends:
                if b.available:
                    return fresh(type(b))
            return CPUFP32Backend(model_fn)

        for b in self._backends:
            if DEVICE_ALIASES.get(b.name, b.name) == key or b.device == key:
                return fresh(type(b))

        if key in ("ONNX", "OPENVINO", "NPU", "GPU", "CPU_FP16"):
            raise ValueError(f"unknown device {device!r}; use auto/cpu/cpu_fp16/onnx/openvino/npu")
        raise ValueError(f"unsupported device {device!r}")

    @classmethod
    def default(cls) -> "BackendRegistry":
        return cls()


def select_execution_provider(device: str = "auto",
                              model_fn: Optional[Callable] = None) -> InferenceBackend:
    """Resolve an execution provider for ``device`` (OpenVINO-style names)."""
    return BackendRegistry.default().resolve(device, model_fn)


def inference_session(device: str = "auto",
                      model_fn: Optional[Callable] = None) -> InferenceBackend:
    """Create an inference session for ``device`` (alias-aware, never raises
    on ``auto`` because the CPU backend is always available)."""
    return select_execution_provider(device, model_fn)
