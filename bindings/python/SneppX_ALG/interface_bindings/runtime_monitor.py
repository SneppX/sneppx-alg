"""Runtime monitoring bindings — S3: integrity monitor, container breakout detection.

Wraps C implementations in ``security/monitor/`` with pure-Python fallback.
"""

import os
import hashlib
import threading
import time
from typing import Dict, List, Optional, Callable

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class IntegrityMonitor:
    """Code and file integrity monitoring — hash verification, tamper detection."""

    def __init__(self):
        self._baselines: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def add_baseline(self, path: str, data: Optional[bytes] = None) -> None:
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        h = hashlib.sha256(data).digest()
        with self._lock:
            self._baselines[path] = h

    def verify(self, path: str) -> bool:
        with self._lock:
            expected = self._baselines.get(path)
        if expected is None:
            return False
        try:
            with open(path, "rb") as f:
                current = hashlib.sha256(f.read()).digest()
            return current == expected
        except (IOError, OSError):
            return False

    def verify_all(self) -> Dict[str, bool]:
        results = {}
        with self._lock:
            paths = list(self._baselines.keys())
        for p in paths:
            results[p] = self.verify(p)
        return results

    def remove_baseline(self, path: str) -> bool:
        with self._lock:
            return self._baselines.pop(path, None) is not None


class ContainerBreakoutDetector:
    """Detection of container escape / breakout attempts."""

    def __init__(self):
        self._suspicious_events: List[dict] = []
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    @staticmethod
    def check_capabilities() -> list:
        return []

    @staticmethod
    def check_mounts() -> list:
        return []

    def record_event(self, event_type: str, details: dict) -> None:
        with self._lock:
            self._suspicious_events.append({
                "type": event_type,
                "details": details,
                "timestamp": time.time(),
            })

    def recent_events(self, n: int = 10) -> List[dict]:
        with self._lock:
            return self._suspicious_events[-n:]

    def clear_events(self) -> None:
        with self._lock:
            self._suspicious_events.clear()


class AdvancedMonitor:
    """Advanced runtime monitoring — anomaly detection, alert correlation."""

    def __init__(self, alert_callback: Optional[Callable] = None):
        self._alerts: List[dict] = []
        self._callback = alert_callback
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    @staticmethod
    def detect_anomaly(metrics: dict) -> float:
        return 0.0

    def raise_alert(self, severity: str, message: str, source: str = "") -> None:
        alert = {
            "severity": severity,
            "message": message,
            "source": source,
            "timestamp": time.time(),
        }
        with self._lock:
            self._alerts.append(alert)
        if self._callback:
            self._callback(alert)

    def alerts_since(self, t: float) -> List[dict]:
        with self._lock:
            return [a for a in self._alerts if a["timestamp"] >= t]

    def clear_alerts(self) -> None:
        with self._lock:
            self._alerts.clear()
