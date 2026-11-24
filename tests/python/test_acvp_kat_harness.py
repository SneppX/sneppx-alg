"""Regression tests for ACVP runner failure and skip accounting."""

import pytest

from SneppX_ALG.interface_bindings import acvp_kat as kat


def test_load_kat_rejects_path_traversal():
    for name in ("../secrets.json", "subdir/vectors.json", "..", "/etc/passwd", "C:\\x\\y.json"):
        with pytest.raises(ValueError, match="invalid vector set name"):
            kat.load_kat(name)


def test_load_kat_rejects_non_json_and_accepts_known_vectors():
    with pytest.raises(ValueError, match="invalid vector set name"):
        kat.load_kat("sha3_256_kat.txt")
    doc = kat.load_kat("sha3_256_kat.json")
    assert doc.get("algorithm") == "SHA3-256"
    assert doc.get("vectors")


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


def test_sha3_kat_runner_compares_expected_digest():
    vector = {"tcId": 7, "len": 8, "msg": "A5", "md": "010203"}

    class Crypto:
        def __init__(self, digest):
            self.digest = digest

        def sha3_256(self, _message):
            return self.digest

    assert kat.run_sha3_256_kat(Crypto(bytes.fromhex("010203")), vector)
    assert not kat.run_sha3_256_kat(Crypto(bytes.fromhex("030201")), vector)


def test_sha3_kat_runner_rejects_missing_expected_digest():
    class Crypto:
        def sha3_256(self, _message):
            return b"\x00"

    with pytest.raises(ValueError, match="expected md digest"):
        kat.run_sha3_256_kat(Crypto(), {"msg": "00", "len": 8})
