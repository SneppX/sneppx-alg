"""Tests for pen_testing.py — S9 penetration testing."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import NetworkFuzzer, SelfAuditor


def test_fuzzer_fuzz():
    nf = NetworkFuzzer(seed=42)
    variants = nf.fuzz(b"GET / HTTP/1.1\r\n", n=10)
    assert len(variants) == 10
    assert all(isinstance(v, bytes) for v in variants)


def test_fuzzer_fuzz_protocol():
    nf = NetworkFuzzer()
    variants = nf.fuzz_protocol(b"HELLO")
    assert len(variants) >= 5
    assert all(isinstance(v, bytes) for v in variants)


def test_fuzzer_mutation_log():
    nf = NetworkFuzzer()
    nf.fuzz(b"data")
    nf.fuzz_protocol(b"data")
    assert len(nf.mutation_log) == 2


def test_self_auditor_register():
    sa = SelfAuditor()
    sa.register_check("disk_space", lambda: True, "high")
    summary = sa.summary()
    assert summary["total"] == 0  # not yet run


def test_self_auditor_run():
    sa = SelfAuditor()
    sa.register_check("check1", lambda: True)
    result = sa.run_check("check1")
    assert result is not None
    assert result["passed"]


def test_self_auditor_run_missing():
    sa = SelfAuditor()
    assert sa.run_check("missing") is None


def test_self_auditor_run_failing():
    sa = SelfAuditor()
    sa.register_check("failing", lambda: False)
    result = sa.run_check("failing")
    assert not result["passed"]


def test_self_auditor_run_all():
    sa = SelfAuditor()
    sa.register_check("a", lambda: True)
    sa.register_check("b", lambda: False)
    results = sa.run_all()
    assert results["a"]["passed"]
    assert not results["b"]["passed"]


def test_self_auditor_summary():
    sa = SelfAuditor()
    sa.register_check("p1", lambda: True)
    sa.register_check("p2", lambda: True)
    sa.register_check("f1", lambda: False)
    sa.run_all()
    summary = sa.summary()
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["compliance_pct"] == 66.66666666666666


def test_self_auditor_clear():
    sa = SelfAuditor()
    sa.register_check("c", lambda: True)
    sa.run_check("c")
    sa.clear()
    assert sa.summary()["total"] == 0


if __name__ == "__main__":
    test_fuzzer_fuzz()
    test_fuzzer_fuzz_protocol()
    test_fuzzer_mutation_log()
    test_self_auditor_register()
    test_self_auditor_run()
    test_self_auditor_run_missing()
    test_self_auditor_run_failing()
    test_self_auditor_run_all()
    test_self_auditor_summary()
    test_self_auditor_clear()
    print("All pen_testing tests passed.")
