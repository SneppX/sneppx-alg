"""Tests for hardware inference backends: probing, routing, fallback, latency."""

import numpy as np
from SneppX_ALG.interface_bindings.hw_backends import (
    DeviceInfo,
    HardwareProbe,
    InferenceBackend,
    CPUFP32Backend,
    CPUFP16Backend,
    ONNXBackend,
    OpenVINOBackend,
    NPUBackend,
    AutoBackend,
    BackendRegistry,
    select_execution_provider,
    inference_session,
    detect_devices,
)

RNG_IN = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32)


def test_detect_devices():
    devs = detect_devices()
    names = {d.name for d in devs}
    assert names == {"CPU", "ONNX", "OPENVINO", "NPU", "GPU"}
    cpu = [d for d in devs if d.name == "CPU"][0]
    assert cpu.available and cpu.kind == "cpu" and cpu.priority == 20
    for d in devs:
        assert isinstance(d, DeviceInfo)
        assert d.priority >= 0
    summary = HardwareProbe.summary()
    assert summary["CPU"] is True
    assert set(summary) == names
    print("  test_detect_devices PASS")


def test_fallback_deterministic():
    b1 = CPUFP32Backend()
    b2 = CPUFP32Backend()
    out1 = b1.run("model-a", RNG_IN)
    out2 = b2.run("model-a", RNG_IN)
    assert np.array_equal(out1, out2)
    out3 = b1.run("model-b", RNG_IN)
    assert not np.array_equal(out1, out3)
    print("  test_fallback_deterministic PASS")


def test_cpu_fp32_backend():
    b = CPUFP32Backend()
    assert b.name == "cpu_fp32" and b.device == "CPU"
    assert b.available
    out = b.run("m", RNG_IN)
    assert out.shape == RNG_IN.shape and out.dtype == np.float32
    assert b.latency_ms() > 0.0
    assert b.latency_ms(average=False) > 0.0
    print("  test_cpu_fp32_backend PASS")


def test_cpu_fp16_backend():
    b = CPUFP16Backend()
    assert b.name == "cpu_fp16" and b.device == "CPU_FP16"
    out = b.run("m", RNG_IN)
    assert out.shape == RNG_IN.shape and out.dtype == np.float32
    ref = CPUFP32Backend().run("m", RNG_IN)
    assert np.allclose(out, ref, rtol=0.05, atol=0.05)
    print("  test_cpu_fp16_backend PASS")


def test_model_fn_honored():
    seen = {}

    def fn(model_id, inputs):
        seen["id"] = model_id
        seen["inputs"] = np.asarray(inputs)
        return np.asarray(7.0, dtype=np.float32)

    b = CPUFP16Backend(model_fn=fn)
    out = b.run("custom", RNG_IN)
    assert out == 7.0
    assert seen["id"] == "custom"
    assert seen["inputs"].dtype == np.float16
    assert b.latency_ms() > 0.0
    print("  test_model_fn_honored PASS")


def test_registry_resolve():
    reg = BackendRegistry.default()
    assert isinstance(reg.resolve("cpu"), CPUFP32Backend)
    assert isinstance(reg.resolve("cpu_fp16"), CPUFP16Backend)
    assert isinstance(reg.resolve("onnx"), ONNXBackend)
    assert isinstance(reg.resolve("openvino"), OpenVINOBackend)
    assert isinstance(reg.resolve("npu"), NPUBackend)
    auto = reg.resolve("auto")
    assert auto.available
    try:
        reg.resolve("quantum")
        raise AssertionError("expected ValueError for unknown device")
    except ValueError:
        pass
    assert reg.get("cpu_fp32") is not None
    assert reg.get("nope") is None
    print("  test_registry_resolve PASS")


def test_auto_backend_and_session():
    b = AutoBackend()
    assert b.available
    assert b.resolved.available
    assert isinstance(b.run("m", RNG_IN), np.ndarray)
    s = inference_session("auto")
    assert s.available and s.run("m", RNG_IN).shape == RNG_IN.shape
    s2 = select_execution_provider("cpu_fp16")
    assert isinstance(s2, CPUFP16Backend)
    print("  test_auto_backend_and_session PASS")


def test_optional_backends_degrade():
    for cls in (ONNXBackend, OpenVINOBackend, NPUBackend):
        b = cls()
        out = b.run("m", RNG_IN)
        assert out.shape == RNG_IN.shape
        assert b.latency_ms() > 0.0
    print("  test_optional_backends_degrade PASS")


def test_latency_and_usage():
    b = CPUFP32Backend()
    for _ in range(5):
        b.run("m", RNG_IN)
    avg = b.latency_ms()
    last = b.latency_ms(average=False)
    assert avg > 0.0 and last > 0.0
    assert b.latency_ms() == avg
    assert avg <= max(b.latency_ms(average=False), avg)
    print("  test_latency_and_usage PASS")


if __name__ == "__main__":
    test_detect_devices()
    test_fallback_deterministic()
    test_cpu_fp32_backend()
    test_cpu_fp16_backend()
    test_model_fn_honored()
    test_registry_resolve()
    test_auto_backend_and_session()
    test_optional_backends_degrade()
    test_latency_and_usage()
    print("ALL hw_backends TESTS PASS")
