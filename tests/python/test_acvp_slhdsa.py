"""Official NIST SLH-DSA sigVer sample vectors vs the C SPHINCS+ verifier.

Vendored set: tests/data/acvp/slhdsa_sigver_kat.json (12 valid + 12 invalid
sigVer samples covering all 12 SLH-DSA parameter sets — FIPS 205).

Honest interop scope (2026-10): the C ``sphincs_*`` implementation is a
round-3-era SPHINCS+ with a fixed single parameter set (sig = 14592 bytes,
pk = 32 e sk = 64 bytes), which is structurally adjacent to but NOT
byte-compatible with finalized FIPS 205 (SLH-DSA-128f valid signatures are
17088 bytes, 128s are 7856, ...).  Therefore NIST's *valid* signatures cannot
verify verbatim; what CAN be positively asserted, and what these tests check:

1. every vector runs through the C verifier without crashing;
2. zero false accepts — all 12 NIST-``testPassed=false`` (invalid/tampered)
   vectors are rejected by the C verifier;
3. C-level self-parity: C sign -> C verify accepts, tampering is rejected;
4. seeded-deterministic key generation.
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
    vecs = kat.slhdsa_sigver_vectors()
    assert len(vecs) == 24, "expected the vendored 24-vector SLH-DSA sigVer set"
    return vecs


def _all_param_sets(vecs):
    return {v["parameterSet"] for v in vecs}


def test_slhdsa_no_false_accepts_on_invalid_nist_vectors(crypto):
    vecs = _vectors()
    invalid = [v for v in vecs if not bool(v["testPassed"])]
    assert len(invalid) == 12
    for v in invalid:
        assert kat.run_slhdsa_sigver(crypto, v, 0) is False, (
            f"C verifier accepted an invalid NIST SLH-DSA sig (tcId={v['tcId']}, "
            f"{v['parameterSet']})"
        )


def test_slhdsa_all_vectors_run_without_crash(crypto):
    vecs = _vectors()
    for v in vecs:
        assert isinstance(kat.run_slhdsa_sigver(crypto, v, 0), bool), v["tcId"]


def test_slhdsa_interop_report_shape_and_honest_agreement(crypto):
    vecs = _vectors()
    report = kat.interop_report(
        crypto, vecs, lambda c, v: kat.run_slhdsa_sigver(c, v, 0)
    )
    assert report["skipped"] == 0
    assert report["total"] == 24
    # 12 invalid correctly rejected; 12 valid rejected only due to the
    # parameter-set mismatch documented in the module docstring.
    assert report["mismatches"] == [v["tcId"] for v in vecs if v["testPassed"]]
    assert report["agreement"] == pytest.approx(0.5)
    # every parameter set of FIPS 205 is represented
    assert len(_all_param_sets(vecs)) == 12


def test_slhdsa_c_roundtrip_and_tamper_rejection(crypto):
    msg = b"sneppx slhdsa self-parity message"
    pk, sk = crypto.sphincs_keygen(0)
    sig = crypto.sphincs_sign(sk, msg, 0)
    assert crypto.sphincs_verify(sig, msg, pk, 0) is True
    assert kat.b2h(pk).upper() != kat.b2h(sk).upper()
    tampered = sig[:-1] + bytes([sig[-1] ^ 0x01])
    assert crypto.sphincs_verify(tampered, msg, pk, 0) is False


def test_slhdsa_seeded_deterministic_keygen(crypto):
    crypto.random_set_seed(bytes(32))
    p1, s1 = crypto.sphincs_keygen(0)
    crypto.random_set_seed(bytes(32))
    p2, s2 = crypto.sphincs_keygen(0)
    assert p1 == p2 and s1 == s2
