"""NIST official SHA3-256 KAT tests (FIPS 202, via ACVP sample vectors).

Vendored vector set: tests/data/acvp/sha3_256_kat.json (byte-aligned subset).
Source: usnistgov/ACVP-Server gen-val/json-files/SHA3-256-2.0 (NIST, public domain).

ACVP ``msg`` values are *bit* strings (``len`` counts bits, final nibble
zero-padded). Only vectors with ``len % 8 == 0`` are vendored because the C
``SNEPPX_sha3_256`` API is byte-oriented; the SHA approved function pads after
the last message *bit*, so a partial-final-byte vector changes the padding and
cannot be produced from any byte string. (The excluded non-bit-aligned vectors
match the Keccak Team ShortMsgKAT bit-level digests — bit-level sponge
absorption requires a bit-length API.)

``acvp_msg_bytes`` performs the bit-string decoding; for byte-aligned vectors
it is exactly ``bytes.fromhex``.
"""

import hashlib

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


def test_sha3_256_official_known_answers(crypto):
    vectors = kat.sha3_256_vectors()
    assert vectors, "no SHA3-256 vectors loaded"
    assert all(v["len"] % 8 == 0 for v in vectors), (
        "vendored set must be byte-aligned only"
    )
    for v in vectors:
        msg = kat.acvp_msg_bytes(v["msg"], v["len"])
        assert len(msg) == v["len"] // 8
        got = kat.run_sha3_256(crypto, msg)
        assert kat.b2h(got).upper() == v["md"].upper(), (
            f"tcId={v['tcId']} msgLen(bits)={v['len']}"
        )


def test_sha3_256_c_matches_hashlib_on_official_vectors(crypto):
    for v in kat.sha3_256_vectors():
        msg = kat.acvp_msg_bytes(v["msg"], v["len"])
        assert (
            hashlib.sha3_256(msg).hexdigest().upper()
            == kat.b2h(kat.run_sha3_256(crypto, msg)).upper()
        ), f"tcId={v['tcId']}"


def test_sha3_256_empty_message_fips202(crypto):
    # FIPS 202 / ACVP: hash of the empty string.
    got = kat.run_sha3_256(crypto, b"")
    assert (
        kat.b2h(got).upper()
        == "A7FFC6F8BF1ED76651C14756A061D662F580FF4DE43B49FA82D80A4B80F8434A"
    )


def test_sha3_256_interop_report_byte_aligned(crypto):
    # KAT-style agreement report over the vendored official vectors.
    vectors = kat.sha3_256_vectors()

    def runner(_crypto, v):
        msg = kat.acvp_msg_bytes(v["msg"], v["len"])
        got = kat.b2h(kat.run_sha3_256(crypto, msg)).upper()
        return got == v["md"].upper()

    report = kat.interop_report(
        crypto,
        vectors,
        runner,
    )
    assert report["mismatches"] == []
    assert report["agreed"] == len(vectors)
    assert report["agreement"] == 1.0
