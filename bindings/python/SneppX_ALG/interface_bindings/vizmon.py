"""Visualization & Monitoring (vizmon) Python bindings.

Wraps the C `sneppx_vizmon` library with a Pythonic interface for
real-time training dashboards.
"""

import ctypes
import numpy as np
from typing import Optional, List, Dict, Any

from .c_loader import load_library, find_load


# --- Load library ---
_VIZMON_LIB, _HAS_C = load_library("sneppx_vizmon")

if _HAS_C and _VIZMON_LIB:
    # Define C signatures
    _VIZMON_LIB.SNEPPX_vizmon_create.argtypes = []
    _VIZMON_LIB.SNEPPX_vizmon_create.restype = ctypes.c_void_p

    _VIZMON_LIB.SNEPPX_vizmon_destroy.argtypes = [ctypes.c_void_p]
    _VIZMON_LIB.SNEPPX_vizmon_destroy.restype = None

    # Scalars
    _VIZMON_LIB.SNEPPX_vizmon_push_scalar.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_scalar.restype = ctypes.c_int

    # Graph
    _VIZMON_LIB.SNEPPX_vizmon_add_node.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    _VIZMON_LIB.SNEPPX_vizmon_add_node.restype = ctypes.c_int

    # Embeddings
    _VIZMON_LIB.SNEPPX_vizmon_push_embedding.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_embedding.restype = ctypes.c_int

    _VIZMON_LIB.SNEPPX_vizmon_project_pca.argtypes = [
        ctypes.c_void_p, ctypes.c_int
    ]
    _VIZMON_LIB.SNEPPX_vizmon_project_pca.restype = ctypes.c_int

    # Samples
    _VIZMON_LIB.SNEPPX_vizmon_push_sample.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.c_double
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_sample.restype = ctypes.c_int

    # Histograms
    _VIZMON_LIB.SNEPPX_vizmon_push_histogram.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_histogram.restype = ctypes.c_int

    # Timeline
    _VIZMON_LIB.SNEPPX_vizmon_push_timeline.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_size_t
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_timeline.restype = ctypes.c_int

    # Sweeps
    _VIZMON_LIB.SNEPPX_vizmon_push_sweep.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    _VIZMON_LIB.SNEPPX_vizmon_push_sweep.restype = ctypes.c_int

    # Snapshot JSON
    _VIZMON_LIB.SNEPPX_vizmon_snapshot_json.argtypes = [ctypes.c_void_p]
    _VIZMON_LIB.SNEPPX_vizmon_snapshot_json.restype = ctypes.c_void_p

    _VIZMON_LIB.SNEPPX_vizmon_free.argtypes = [ctypes.c_void_p]
    _VIZMON_LIB.SNEPPX_vizmon_free.restype = None

    # Server control
    _VIZMON_LIB.SNEPPX_vizmon_start.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _VIZMON_LIB.SNEPPX_vizmon_start.restype = ctypes.c_int

    _VIZMON_LIB.SNEPPX_vizmon_stop.argtypes = [ctypes.c_void_p]
    _VIZMON_LIB.SNEPPX_vizmon_stop.restype = None

    # Frontend / Export HTML
    _VIZMON_LIB.SNEPPX_vizmon_frontend_html.argtypes = []
    _VIZMON_LIB.SNEPPX_vizmon_frontend_html.restype = ctypes.c_void_p

    _VIZMON_LIB.SNEPPX_vizmon_export_html.argtypes = [ctypes.c_void_p]
    _VIZMON_LIB.SNEPPX_vizmon_export_html.restype = ctypes.c_void_p


# --- Pythonic wrapper ---

class VizMon:
    """High-level Python wrapper for the SNEPPX vizmon control plane.

    Provides thread-safe metric ingestion and a self-contained HTTP+WS dashboard.
    """

    SAMPLE_KIND_IMG = 0  # PNG
    SAMPLE_KIND_AUD = 1  # WAV

    def __init__(self):
        if not _HAS_C:
            raise RuntimeError("C vizmon library not found. Build with SNEPPX_BUILD_VIZMON=ON.")
        self._handle = _VIZMON_LIB.SNEPPX_vizmon_create()
        if not self._handle:
            raise RuntimeError("Failed to create vizmon engine")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        if self._handle:
            _VIZMON_LIB.SNEPPX_vizmon_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---- Scalars ----
    def push_scalar(self, tag: str, value: float, ts: Optional[float] = None):
        """Push a scalar value (loss, accuracy, LR, etc.).

        Args:
            tag: Metric name (e.g. "loss", "accuracy").
            value: Metric value.
            ts: Optional timestamp (monotonic seconds). Defaults to time.time().
        """
        import time
        if ts is None:
            ts = time.time()
        r = _VIZMON_LIB.SNEPPX_vizmon_push_scalar(
            self._handle, tag.encode(), value, ts
        )
        if r != 0:
            raise RuntimeError(f"push_scalar failed for tag '{tag}'")

    # ---- Graph ----
    def add_node(self, name: str, in_nodes: List[int], out_node: int,
                 shape: List[int]):
        """Register a graph node (operation + tensor shape)."""
        in_arr = (ctypes.c_int * len(in_nodes))(*in_nodes)
        shape_arr = (ctypes.c_double * len(shape))(*[float(s) for s in shape])
        r = _VIZMON_LIB.SNEPPX_vizmon_add_node(
            self._handle, name.encode(), in_arr, len(in_nodes),
            out_node, shape_arr, len(shape)
        )
        if r != 0:
            raise RuntimeError(f"add_node failed for '{name}'")

    # ---- Embeddings ----
    def push_embedding(self, tag: str, data: np.ndarray):
        """Push high-dimensional embedding data (n_samples, n_dim).

        Automatically copies to contiguous C-order float64.
        """
        if data.ndim != 2:
            raise ValueError("embedding data must be 2-D (n_samples, n_dim)")
        arr = np.ascontiguousarray(data, dtype=np.float64)
        r = _VIZMON_LIB.SNEPPX_vizmon_push_embedding(
            self._handle, tag.encode(),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            arr.shape[0], arr.shape[1]
        )
        if r != 0:
            raise RuntimeError(f"push_embedding failed for '{tag}'")

    def project_pca(self, max_components: int = 2):
        """Project all stored embeddings to 2-D via PCA."""
        r = _VIZMON_LIB.SNEPPX_vizmon_project_pca(self._handle, max_components)
        if r != 0:
            raise RuntimeError("project_pca failed")

    # ---- Samples ----
    def push_image(self, tag: str, png_bytes: bytes, ts: Optional[float] = None):
        """Push a PNG image sample."""
        import time
        if ts is None:
            ts = time.time()
        arr = (ctypes.c_ubyte * len(png_bytes)).from_buffer_copy(png_bytes)
        r = _VIZMON_LIB.SNEPPX_vizmon_push_sample(
            self._handle, tag.encode(), self.SAMPLE_KIND_IMG,
            arr, len(png_bytes), ts
        )
        if r != 0:
            raise RuntimeError(f"push_image failed for '{tag}'")

    def push_audio(self, tag: str, wav_bytes: bytes, ts: Optional[float] = None):
        """Push a WAV audio sample."""
        import time
        if ts is None:
            ts = time.time()
        arr = (ctypes.c_ubyte * len(wav_bytes)).from_buffer_copy(wav_bytes)
        r = _VIZMON_LIB.SNEPPX_vizmon_push_sample(
            self._handle, tag.encode(), self.SAMPLE_KIND_AUD,
            arr, len(wav_bytes), ts
        )
        if r != 0:
            raise RuntimeError(f"push_audio failed for '{tag}'")

    # ---- Histograms ----
    def push_histogram(self, tag: str, values: np.ndarray,
                       n_bins: int = 64, ts: Optional[float] = None):
        """Push weight/bias/gradient histogram."""
        import time
        if ts is None:
            ts = time.time()
        arr = np.ascontiguousarray(values, dtype=np.float64).flatten()
        r = _VIZMON_LIB.SNEPPX_vizmon_push_histogram(
            self._handle, tag.encode(),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            arr.shape[0], n_bins, ts
        )
        if r != 0:
            raise RuntimeError(f"push_histogram failed for '{tag}'")

    # ---- Timeline ----
    def push_timeline(self, name: str, start_ms: float,
                      dur_ms: float, mem_bytes: int):
        """Push a kernel profiling event."""
        r = _VIZMON_LIB.SNEPPX_vizmon_push_timeline(
            self._handle, name.encode(), start_ms, dur_ms, mem_bytes
        )
        if r != 0:
            raise RuntimeError(f"push_timeline failed for '{name}'")

    # ---- Sweeps ----
    def push_sweep(self, config_id: str, final_loss: float,
                   final_acc: float, duration_s: float):
        """Record a hyperparameter sweep result."""
        r = _VIZMON_LIB.SNEPPX_vizmon_push_sweep(
            self._handle, config_id.encode(),
            final_loss, final_acc, duration_s
        )
        if r != 0:
            raise RuntimeError(f"push_sweep failed for '{config_id}'")

    # ---- Snapshot / Export ----
    def snapshot_json(self) -> Dict[str, Any]:
        """Get current dashboard state as a JSON-decoded dict."""
        import json
        ptr = _VIZMON_LIB.SNEPPX_vizmon_snapshot_json(self._handle)
        if not ptr:
            return {}
        try:
            data = ctypes.cast(ptr, ctypes.c_char_p).value
            if data is None:
                return {}
            return json.loads(data.decode())
        finally:
            _VIZMON_LIB.SNEPPX_vizmon_free(ptr)

    def export_html(self) -> str:
        """Generate a standalone HTML file embedding current snapshot."""
        ptr = _VIZMON_LIB.SNEPPX_vizmon_export_html(self._handle)
        if not ptr:
            return ""
        try:
            return ctypes.cast(ptr, ctypes.c_char_p).value.decode()
        finally:
            _VIZMON_LIB.SNEPPX_vizmon_free(ptr)

    # ---- Server ----
    def start_server(self, port: int = 8300):
        """Start the HTTP+WebSocket dashboard server in background thread."""
        r = _VIZMON_LIB.SNEPPX_vizmon_start(self._handle, port)
        if r != 0:
            raise RuntimeError(f"Failed to start server on port {port}")

    def stop_server(self):
        """Stop the dashboard server."""
        _VIZMON_LIB.SNEPPX_vizmon_stop(self._handle)

    @staticmethod
    def get_frontend_html() -> str:
        """Get the embedded Vue 3 + Chart.js dashboard HTML."""
        if not _HAS_C:
            return ""
        ptr = _VIZMON_LIB.SNEPPX_vizmon_frontend_html()
        if not ptr:
            return ""
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode()


# --- Convenience functions ---

def start_dashboard(port: int = 8300) -> VizMon:
    """Create a VizMon instance and start its HTTP+WS server.

    Returns the VizMon instance; call .close() to shut down.
    """
    vm = VizMon()
    vm.start_server(port)
    return vm


def create_vizmon() -> VizMon:
    """Create a VizMon instance (server not started)."""
    return VizMon()


__all__ = [
    "VizMon",
    "start_dashboard",
    "create_vizmon",
    "_HAS_C",
]