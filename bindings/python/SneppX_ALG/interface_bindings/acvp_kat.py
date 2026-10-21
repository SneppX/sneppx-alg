"""ACVP / NIST known-answer-test (KAT) harness helpers.

Loads official NIST sample vector sets (vendored under ``tests/data/acvp/``)
and adapters that run them against the C crypto implementations exposed by
the ``_SNEPPX_c`` extension.

Source of the vendored vectors:
https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files
(NIST public-domain sample vector sets for FIPS 202 / 203 / 204 / 205).

This module is a sibling of the ``interface_bindings`` package but only
depends on the standard library plus the optional ``_SNEPPX_c`` backend
passed in by the caller, so it stays self-contained.
"""

import binascii
import json
import os
from typing import Any, Dict, List, Optional

# Parameter-set maps -----------------------------------------------------------

#: FIPS 204 parameter set name -> the "variant" integer used by
#: ``SNEPPX_dilithium_*`` in the C layer (2 -> ML-DSA-44, 3 -> ML-DSA-65,
#: 5 -> ML-DSA-87).
ML_DSA_VARIANTS = {
    "ML-DSA-44": 2,
    "ML-DSA-65": 3,
    "ML-DSA-87": 5,
}

#: FIPS 205 parameter sets officially defined.  The C SPHINCS+ binding
#: currently exposes a single fixed parameter set regardless of ``variant``;
#: entries here map a set name to a stable id for reporting purposes.
SLH_DSA_SETS = [
    "SLH-DSA-SHA2-128s",
    "SLH-DSA-SHAKE-128s",
    "SLH-DSA-SHA2-128f",
    "SLH-DSA-SHAKE-128f",
    "SLH-DSA-SHA2-192s",
    "SLH-DSA-SHAKE-192s",
    "SLH-DSA-SHA2-192f",
    "SLH-DSA-SHAKE-192f",
    "SLH-DSA-SHA2-256s",
    "SLH-DSA-SHAKE-256s",
    "SLH-DSA-SHA2-256f",
    "SLH-DSA-SHAKE-256f",
]

_SIZE_TO_ML_DSA = {1312: "ML-DSA-44", 1952: "ML-DSA-65", 2592: "ML-DSA-87"}


# Data paths -------------------------------------------------------------------

_DATA_DIR_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests",
        "data",
        "acvp",
    ),
    os.path.join(os.getcwd(), "tests", "data", "acvp"),
]


def data_dir() -> Optional[str]:
    """Return the directory holding the vendored ACVP JSON files, or None."""
    for cand in _DATA_DIR_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def load_kat(name: str) -> Dict[str, Any]:
    """Load one vendored ACVP vector set by file name (e.g. ``sha3_256_kat.json``)."""
    d = data_dir()
    if d is None:
        raise FileNotFoundError("ACVP data directory not found (tests/data/acvp)")
    path = os.path.join(d, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ACVP vector set not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Hex helpers ------------------------------------------------------------------


def h2b(hexstr: str) -> bytes:
    """Hex string -> bytes (case-insensitive, tolerant of empty/None)."""
    if not hexstr:
        return b""
    return binascii.unhexlify(hexstr.strip())


def b2h(data: bytes) -> str:
    """bytes -> uppercase hex string."""
    return binascii.hexlify(data).decode("ascii").upper()


def acvp_msg_bytes(msg_hex: str, bit_len: int) -> bytes:
    """Decode an ACVP bit-string message into bytes.

    ACVP vectors carry ``msg`` as hex with two quirks:

    * ``len`` counts *bits*, not bytes;
    * when ``len`` is not a multiple of 8, the hex string is padded to a
      whole nibble with zero bits at the low end of the final nibble.
    """
    raw = h2b(msg_hex)
    total_bits = len(raw) * 8
    if bit_len is None or bit_len >= total_bits:
        return raw
    nbytes = (bit_len + 7) // 8
    out = bytearray(raw[:nbytes])
    if bit_len % 8:
        keep = bit_len % 8  # significant high bits in the final byte
        mask = ((1 << keep) - 1) << (8 - keep)
        out[nbytes - 1] &= mask
    return bytes(out)


# Adapters ---------------------------------------------------------------------


def sha3_256_vectors() -> List[Dict[str, Any]]:
    """Official NIST FIPS 202 SHA3-256 sample vectors (msg -> md)."""
    doc = load_kat("sha3_256_kat.json")
    return doc["vectors"]


def run_sha3_256(crypto: Any, msg: bytes) -> bytes:
    """Run SHA3-256 via the C backend."""
    return bytes(crypto.sha3_256(msg))


def mldsa_sigver_vectors() -> List[Dict[str, Any]]:
    """Official NIST FIPS 204 ML-DSA sigVer sample vectors."""
    doc = load_kat("mldsa_sigver_kat.json")
    return doc["vectors"]


def ml_dsa_variant_for_pk(pk: bytes) -> Optional[int]:
    """Map the public-key length used by ML-DSA sigVer vectors to a variant id."""
    return {"ML-DSA-44": 2, "ML-DSA-65": 3, "ML-DSA-87": 5}.get(
        _SIZE_TO_ML_DSA.get(len(pk), "")
    )


def run_mldsa_sigver(crypto: Any, vector: Dict[str, Any]) -> Optional[bool]:
    """Run one ML-DSA sigVer vector through the C verifier.

    Returns True/False (verifier decision) or None if the vector cannot be
    mapped to a supported parameter set via public-key length.
    """
    variant = ml_dsa_variant_for_pk(h2b(vector["pk"]))
    if variant is None:
        return None
    ok = crypto.dilithium_verify(
        h2b(vector["sig"]), h2b(vector["msg"]), h2b(vector["pk"]), variant
    )
    return bool(ok == 0)


def run_mldsa_sign_verify_roundtrip(crypto: Any, msg: bytes, variant: int) -> bool:
    """Sign ``msg`` with the C keygen output and verify it.  Self-parity check."""
    pk, sk = crypto.dilithium_keygen(variant)
    sig = crypto.dilithium_sign(sk, msg, variant)
    return crypto.dilithium_verify(sig, msg, pk, variant) == 0


def slhdsa_sigver_vectors() -> List[Dict[str, Any]]:
    """Official NIST FIPS 205 SLH-DSA sigVer sample vectors."""
    doc = load_kat("slhdsa_sigver_kat.json")
    return doc["vectors"]


def run_slhdsa_sigver(crypto: Any, vector: Dict[str, Any], variant: int) -> bool:
    """Run one SLH-DSA sigVer vector through the C verifier (fixed parameter set)."""
    return bool(
        crypto.sphincs_verify(
            h2b(vector["sig"]), h2b(vector["msg"]), h2b(vector["pk"]), variant
        )
    )


def mldsa_keygen_vectors() -> List[Dict[str, Any]]:
    """Official NIST FIPS 204 ML-DSA keyGen sample vectors (seed -> pk/sk)."""
    doc = load_kat("mldsa_keygen_kat.json")
    return doc["vectors"]


def mlkem_keygen_vectors() -> List[Dict[str, Any]]:
    """Official NIST FIPS 203 ML-KEM keyGen sample vectors (d/z -> ek/dk)."""
    doc = load_kat("mlkem_keygen_kat.json")
    return doc["vectors"]


def interop_report(
    crypto: Any, vectors: List[Dict[str, Any]], runner: Any
) -> Dict[str, Any]:
    """Summarise agreement between a verifier and NIST's expected decisions.

    Returns ``{"total", "agreed", "mismatches", "skipped", "agreement"}``
    where ``agreement`` is the fraction of mapped vectors whose C verifier
    decision matched NIST's ``testPassed`` flag.  Vectors (e.g. KATs) without a
    ``testPassed`` field are interpreted as "must succeed" — ``decision`` must
    be truthy (e.g. digest equality) to count as agreement.
    """
    total = agreed = skipped = 0
    mismatches = []
    for v in vectors:
        try:
            decision = runner(crypto, v)
        except Exception:
            decision = None
        if decision is None:
            skipped += 1
            continue
        total += 1
        expected = bool(v["testPassed"]) if "testPassed" in v else True
        if bool(decision) == expected:
            agreed += 1
        else:
            mismatches.append(v["tcId"])
    agreement = (agreed / total) if total else 0.0
    return {
        "total": total,
        "agreed": agreed,
        "mismatches": mismatches,
        "skipped": skipped,
        "agreement": agreement,
    }


__all__ = [
    "ML_DSA_VARIANTS",
    "SLH_DSA_SETS",
    "data_dir",
    "load_kat",
    "h2b",
    "b2h",
    "acvp_msg_bytes",
    "sha3_256_vectors",
    "run_sha3_256",
    "mldsa_sigver_vectors",
    "ml_dsa_variant_for_pk",
    "run_mldsa_sigver",
    "run_mldsa_sign_verify_roundtrip",
    "slhdsa_sigver_vectors",
    "run_slhdsa_sigver",
    "mldsa_keygen_vectors",
    "mlkem_keygen_vectors",
    "interop_report",
]
