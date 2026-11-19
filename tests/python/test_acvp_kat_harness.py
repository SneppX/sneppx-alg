"""Regression tests for ACVP runner failure and skip accounting."""

import pytest

from SneppX_ALG.interface_bindings import acvp_kat as kat


def test_interop_report_distinguishes_errors_mismatches_and_skips():
    vectors = [
        {"tcId": 101, "testPassed": True},
        {"tcId": 102, "testPassed": False},
        {"tcId": 103, "testPassed": True},
    ]

    def runner(_crypto, vector):
        if vector["tcId"] == 101:
            raise RuntimeError("backend failed")
        if vector["tcId"] == 102:
            return False
        return None

    report = kat.interop_report(None, vectors, runner)

    assert report["total"] == 2
    assert report["agreed"] == 1
    assert report["skipped"] == 1
    assert report["mismatches"] == [101]
    assert report["errors"] == [{"tcId": 101, "error": "RuntimeError"}]
    assert report["agreement"] == 0.5


def test_mldsa_keygen_backend_exception_is_not_a_skip():
    cleared = []

    class Crypto:
        def random_set_seed(self, _seed):
            pass

        def random_clear_seed(self):
            cleared.append(True)

        def dilithium_keygen(self, _variant):
            raise RuntimeError("native keygen failed")

    vector = {"parameterSet": "ML-DSA-44", "seed": "00"}
    with pytest.raises(RuntimeError, match="native keygen failed"):
        kat.run_mldsa_keygen(Crypto(), vector)
    assert cleared == [True]


def test_mlkem_keygen_backend_exception_is_not_a_skip():
    cleared = []

    class Crypto:
        def random_set_seed(self, _seed):
            pass

        def random_clear_seed(self):
            cleared.append(True)

        def kyber_keygen(self, _variant):
            raise RuntimeError("native keygen failed")

    vector = {"parameterSet": "ML-KEM-512", "d": "00"}
    with pytest.raises(RuntimeError, match="native keygen failed"):
        kat.run_mlkem_keygen(Crypto(), vector)
    assert cleared == [True]


def test_mlkem_roundtrip_keygen_exception_is_not_a_skip():
    cleared = []

    class Crypto:
        def random_set_seed(self, _seed):
            pass

        def random_clear_seed(self):
            cleared.append(True)

        def kyber_keygen(self, _variant):
            raise RuntimeError("native keygen failed")

    vector = {"parameterSet": "ML-KEM-512", "d": "00"}
    with pytest.raises(RuntimeError, match="native keygen failed"):
        kat.run_mlkem_encaps_decaps_roundtrip(Crypto(), vector)
    assert cleared == [True]
