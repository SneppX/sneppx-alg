"""Official NIST ML-DSA sigVer sample vectors vs the C Dilithium verifier.

Vendored set: tests/data/acvp/mldsa_sigver_kat.json (6 valid + 6 invalid
sigVer samples across ML-DSA-44/65/87 — FIPS 204).

Honest interop scope (2026-10): the C ``dilithium_*`` implementation is a
round-3-era simplified Dilithium (zeroed ``tr`` slot, no ``key`` domain
separation) which is structurally adjacent to but NOT byte-compatible with
finalized FIPS 204.  NIST's *valid* signatures therefore cannot verify
verbatim.  What CAN be positively asserted here:

1. every vector maps by pk length to a C variant and runs without crashing;
2. zero false accepts — all NIST-``testPassed=false`` vectors are rejected;
3. C-level self-parity: C keygen -> sign -> verify accepts, tampering is
   rejected (variants 2 and 3);
4. public-key lengths match the FIPS 204 sizes for each variant.
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
    vecs = kat.mldsa_sigver_vectors()
    assert len(vecs) == 18, "expected the vendored 18-vector ML-DSA sigVer set"
    return vecs


# FIPS 204 public-key sizes per defined variant id (2/3/5 = ML-DSA-44/65/87).
_FIPS204_PK_SIZES = {2: 1312, 3: 1952, 5: 2592}


def test_mldsa_all_vectors_run_without_crash(crypto):
    vecs = _vectors()
    for v in vecs:
        variant = kat.ml_dsa_variant_for_pk(kat.h2b(v["pk"]))
        assert variant in _FIPS204_PK_SIZES, v["parameterSet"]
        assert isinstance(kat.run_mldsa_sigver(crypto, v), bool), v["tcId"]


def test_mldsa_pk_lengths_match_fips_204(crypto):
    vecs = _vectors()
    for v in vecs:
        variant = kat.ml_dsa_variant_for_pk(kat.h2b(v["pk"]))
        assert len(kat.h2b(v["pk"])) == _FIPS204_PK_SIZES[variant]


def test_mldsa_no_false_accepts_on_invalid_nist_vectors(crypto):
    vecs = _vectors()
    invalid = [v for v in vecs if not bool(v["testPassed"])]
    assert len(invalid) == 9
    for v in invalid:
        assert kat.run_mldsa_sigver(crypto, v) is False, (
            f"C verifier accepted an invalid NIST ML-DSA sig "
            f"(tcId={v['tcId']}, {v['parameterSet']})"
        )


def test_mldsa_interop_report_shape_and_honest_agreement(crypto):
    vecs = _vectors()
    report = kat.interop_report(crypto, vecs, kat.run_mldsa_sigver)
    assert report["skipped"] == 0
    assert report["total"] == 18
    # 9 invalid correctly rejected; 9 valid rejected only due to the
    # round-3-vs-FIPS-204 structural mismatch documented in the docstring.
    assert report["mismatches"] == [v["tcId"] for v in vecs if v["testPassed"]]
    assert report["agreement"] == pytest.approx(0.5)


def test_mldsa_roundtrip_and_tamper_rejection(crypto):
    for variant in (2, 3):
        pk, sk = crypto.dilithium_keygen(variant)
        msg = b"sneppx mldsa self-parity message"
        sig = crypto.dilithium_sign(sk, msg, variant)
        assert crypto.dilithium_verify(sig, msg, pk, variant) == 0, variant
        flipped = sig[:-1] + bytes([sig[-1] ^ 0x01])
        assert crypto.dilithium_verify(flipped, msg, pk, variant) == -1
