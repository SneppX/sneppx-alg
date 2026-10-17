"""An in-process metric registry: counters, gauges, histograms (pure Python)."""

import numpy as np


class MetricRegistry:
    """Collect counters, gauges and histogram observations."""

    def __init__(self):
        self.counters: dict = {}
        self.gauges: dict = {}
        self.histograms: dict = {}

    def counter(self, name: str, amount: float = 1.0) -> float:
        self.counters[name] = self.counters.get(name, 0.0) + amount
        return self.counters[name]

    def gauge(self, name: str, value: float) -> float:
        self.gauges[name] = value
        return value

    def histogram(self, name: str, value: float) -> None:
        self.histograms.setdefault(name, []).append(float(value))

    def histogram_summary(self, name: str) -> dict:
        values = self.histograms.get(name, [])
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    def summary(self) -> dict:
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: self.histogram_summary(name) for name in self.histograms
            },
        }

    def clear(self) -> None:
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
