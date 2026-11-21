"""Tests for runtime_monitor.py — S3 runtime monitoring."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import IntegrityMonitor, ContainerBreakoutDetector, AdvancedMonitor


def test_integrity_baseline():
    import tempfile
    im = IntegrityMonitor()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data")
        fname = f.name
    try:
        im.add_baseline(fname)
        assert im.verify(fname)
    finally:
        os.unlink(fname)


def test_integrity_tampered():
    import tempfile
    im = IntegrityMonitor()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"original")
        fname = f.name
    try:
        im.add_baseline(fname)
        with open(fname, "wb") as f:
            f.write(b"tampered")
        assert not im.verify(fname)
    finally:
        os.unlink(fname)


def test_integrity_missing():
    im = IntegrityMonitor()
    assert not im.verify("/nonexistent")


def test_integrity_remove():
    im = IntegrityMonitor()
    im.add_baseline("/tmp/x", b"x")
    assert im.remove_baseline("/tmp/x")
    assert not im.remove_baseline("/tmp/x")


def test_container_breakout_record():
    c = ContainerBreakoutDetector()
    c.record_event("sensitive_mount", {"path": "/proc"})
    events = c.recent_events()
    assert len(events) == 1
    assert events[0]["type"] == "sensitive_mount"


def test_container_breakout_clear():
    c = ContainerBreakoutDetector()
    c.record_event("test", {})
    c.clear_events()
    assert len(c.recent_events()) == 0


def test_advanced_monitor_alert():
    alerts = []
    def cb(a):
        alerts.append(a)
    m = AdvancedMonitor(alert_callback=cb)
    m.raise_alert("high", "test alert", "sensor1")
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "high"
    assert len(m.alerts_since(0)) == 1


def test_advanced_monitor_clear():
    m = AdvancedMonitor()
    m.raise_alert("low", "test", "")
    m.clear_alerts()
    assert len(m.alerts_since(0)) == 0


def test_advanced_monitor_detect_anomaly():
    score = AdvancedMonitor.detect_anomaly({"cpu": 95.0, "mem": 80.0})
    assert isinstance(score, float)


if __name__ == "__main__":
    test_integrity_baseline()
    test_integrity_tampered()
    test_integrity_missing()
    test_integrity_remove()
    test_container_breakout_record()
    test_container_breakout_clear()
    test_advanced_monitor_alert()
    test_advanced_monitor_clear()
    test_advanced_monitor_detect_anomaly()
    print("All runtime_monitor tests passed.")
