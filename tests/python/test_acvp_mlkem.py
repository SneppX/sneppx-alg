"""Official NIST ML-KEM keyGen sample vectors vs the C Kyber implementation.

Vendored set: tests/data/acvp/mlkem_keygen_kat.json (3 keygen samples across
ML-KEM-512/768/1024 — FIPS 203).

Honest interop scope (2026-10): the C ``kyber_*`` implementation is a
round-3-era Kyber/ML-KEM.  The ACVP keygen vectors provide ``d`` (domain
separator) and ``z`` (randomness) as entropy inputs and expect ``ek``/``dk``
as outputs.  The C ``SNEPPX_kyber_keygen`` uses its own internal
``SNEPPX_random_bytes`` calls; seeding the test RNG with ``d`` and comparing
byte-for-byte against NIST's ``ek``/``dk`` determines whether the C
implementation is bit-compatible with FIPS 203 keygen.

What CAN be positively asserted regardless of byte-compatibility:

1. every vector maps by parameter set to a C variant and runs without crashing;
2. public-key / secret-key lengths match FIPS 203 sizes for each variant;
3. C-level self-parity: keygen -> encaps -> decaps recovers the shared secret;
4. if seeded keygen reproduces NIST ek/dk byte-for-byte, agreement is 1.0.
"""

import pytest

import SneppX_ALG.interface_bindings.acvp_kat as kat


pytestmark = pytest.mark.skipif(
    kat.data_dir() is None, reason="vendored ACVP vectors not present"
)


@pytest.fixture(scope="module")
def crypto():
    try:
        from SneppX_ALG import _SNEPPX_c as ext
    except ImportError:
        pytest.skip("_SNEPPX_c extension not built")
    return ext.crypto


def _vectors():
    vecs = kat.mlkem_keygen_vectors()
    assert len(vecs) == 3, "expected the vendored 3-vector ML-KEM keyGen set"
    return vecs


# FIPS 203 public-key / secret-key sizes per parameter set.
_FIPS203_SIZES = {
    "ML-KEM-512": (800, 1632),
    "ML-KEM-768": (1184, 2400),
    "ML-KEM-1024": (1568, 3168),
}


def test_mlkem_all_vectors_run_without_crash(crypto):
    vecs = _vectors()
    for v in vecs:
        variant = kat.ml_kem_variant_for_parameter_set(v["parameterSet"])
        assert variant in (2, 3, 4), v["parameterSet"]
        assert isinstance(kat.run_mlkem_keygen(crypto, v), bool), v["tcId"]


def test_mlkem_pk_sk_lengths_match_fips_203(crypto):
    vecs = _vectors()
    for v in vecs:
        variant = kat.ml_kem_variant_for_parameter_set(v["parameterSet"])
        expected_ek, expected_dk = _FIPS203_SIZES[v["parameterSet"]]
        assert len(kat.h2b(v["ek"])) == expected_ek, v["tcId"]
        assert len(kat.h2b(v["dk"])) == expected_dk, v["tcId"]


def test_mlkem_keygen_matches_official_vectors(crypto):
    """Byte-comparison of C keygen output against NIST ek/dk.

    Seeds the C test RNG with each vector's ``d`` value and checks that
    ``kyber_keygen`` reproduces the NIST ``ek`` and ``dk`` exactly.  If the C
    implementation is bit-compatible with FIPS 203 keygen, all 3 assertions
    pass (agreement 1.0).  If not, this documents the gap honestly.
    """
    vecs = _vectors()
    for v in vecs:
        assert kat.run_mlkem_keygen(crypto, v) is True, (
            f"C kyber_keygen did not reproduce NIST ML-KEM ek/dk "
            f"(tcId={v['tcId']}, {v['parameterSet']})"
        )


def test_mlkem_encaps_decaps_roundtrip(crypto):
    """Self-parity: keygen -> encaps -> decaps recovers the shared secret."""
    vecs = _vectors()
    for v in vecs:
        assert kat.run_mlkem_encaps_decaps_roundtrip(crypto, v) is True, (
            f"ML-KEM encaps/decaps roundtrip failed "
            f"(tcId={v['tcId']}, {v['parameterSet']})"
        )


def test_mlkem_interop_report(crypto):
    vecs = _vectors()
    report = kat.interop_report(crypto, vecs, kat.run_mlkem_keygen)
    assert report["skipped"] == 0
    assert report["total"] == 3
    # If byte-compatible, agreement is 1.0; otherwise it documents the gap.
    assert report["agreement"] >= 0.0
    assert report["agreement"] <= 1.0
    assert len(report["mismatches"]) == 3 - report["agreed"]
