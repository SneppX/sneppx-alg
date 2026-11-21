"""Known-answer tests (KAT) for Dilithium and SPHINCS+.

These vectors were generated deterministically from a fixed RNG seed and are
stored alongside the test as ``data/kat_vectors.json``. The test re-derives
the keypairs, signatures and public keys from the same seed and asserts byte
equality, which catches any silent regression in the post-quantum signature
implementations (parameter handling, rejection sampling, hint packing,
decomposition, etc.).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import crypto_sign as _cs
from SneppX_ALG.interface_bindings import Dilithium, SphincsPlus

import pytest

try:
    from SneppX_ALG import _SNEPPX_c as _ext
    _C = getattr(_ext, "crypto", None)
except ImportError:
    _C = None

pytestmark = pytest.mark.skipif(
    _C is None or not hasattr(_C, "random_set_seed"),
    reason="requires _SNEPPX_c built with SNEPPX_ENABLE_TEST_RNG=ON",
)
_DATA = os.path.join(os.path.dirname(__file__), "data", "kat_vectors.json")


def _load():
    with open(_DATA) as f:
        return json.load(f)


def test_kat_vectors():
    data = _load()
    seed = bytes.fromhex(data["seed"])
    msg = bytes.fromhex(data["msg"])
    vectors = data["vectors"]
    try:
        for name, vec in vectors.items():
            _C.random_set_seed(seed)
            if name.startswith("dilithium_"):
                variant = int(name.split("_")[1])
                d = Dilithium(variant)
                pk, sk = d.keypair()
                sig = d.sign(sk, msg)
            elif name == "sphincs":
                sp = SphincsPlus()
                pk, sk = sp.keypair()
                sig = sp.sign(sk, msg)
            else:
                raise AssertionError("unknown vector %r" % name)

            assert len(pk) == len(bytes.fromhex(vec["pk"])), name
            assert pk.hex() == vec["pk"], "%s pk mismatch" % name
            assert sk.hex() == vec["sk"], "%s sk mismatch" % name
            assert sig.hex() == vec["sig"], "%s sig mismatch" % name

            # The generated signature must also verify.
            if name.startswith("dilithium_"):
                assert d.verify(pk, msg, sig), "%s verify failed" % name
            else:
                assert sp.verify(pk, msg, sig), "%s verify failed" % name
    finally:
        _C.random_clear_seed()
