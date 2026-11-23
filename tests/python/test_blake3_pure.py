"""NIST-independent BLAKE3 known-answer tests (official BLAKE3 vectors)."""

import hashlib

import pytest

from SneppX_ALG.interface_bindings._blake3_pure import (
    blake3_derive_key,
    blake3_hash,
    blake3_keyed_hash,
)

KEY = b"whats the Elvish word for friend"
CONTEXT = "BLAKE3 2019-12-27 16:29:52 test vectors context"


def _input(length: int) -> bytes:
    return bytes(i % 251 for i in range(length))


@pytest.mark.parametrize(
    "length,h,keyed,derived",
    [
        (
            0,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26",
            "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d",
        ),
        (
            1,
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
            "6d7878dfff2f485635d39013278ae14f1454b8c0a3a2d34bc1ab38228a80c95b",
            "b3e2e340a117a499c6cf2398a19ee0d29cca2bb7404c73063382693bf66cb06c",
        ),
        (
            64,
            "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
            "ba8ced36f327700d213f120b1a207a3b8c04330528586f414d09f2f7d9ccb7e6",
            "a5c4a7053fa86b64746d4bb688d06ad1f02a18fce9afd3e818fefaa7126bf73e",
        ),
        (
            1024,
            "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
            "75c46f6f3d9eb4f55ecaaee480db732e6c2105546f1e675003687c31719c7ba4",
            "7356cd7720d5b66b6d0697eb3177d9f8d73a4a5c5e968896eb6a689684302706",
        ),
        (
            1025,
            "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
            "357dc55de0c7e382c900fd6e320acc04146be01db6a8ce7210b7189bd664ea69",
            "effaa245f065fbf82ac186839a249707c3bddf6d3fdda22d1b95a3c970379bcb",
        ),
        (
            4096,
            "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969",
            "befc660aea2f1718884cd8deb9902811d332f4fc4a38cf7c7300d597a081bfc0",
            "1e0d7f3db8c414c97c6307cbda6cd27ac3b030949da8e23be1a1a924ad2f25b9",
        ),
    ],
)
def test_official_vectors(length, h, keyed, derived):
    data = _input(length)
    assert blake3_hash(data).hex() == h
    assert blake3_keyed_hash(KEY, data).hex() == keyed
    assert blake3_derive_key(CONTEXT, data).hex() == derived


def test_xof_extends_first_output_block():
    expected = (
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        "e00f03e7b69af26b7faaf09fcd333050338ddfe085b8cc869ca98b206c08243a"
    )
    assert blake3_hash(b"", 64).hex() == expected
    assert blake3_hash(b"", 5) == bytes.fromhex(expected)[:5]


def test_invalid_key_length():
    with pytest.raises(ValueError):
        blake3_keyed_hash(b"short", b"data")


def test_blake3_api_never_downgrades_to_sha256(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "blake3", None)
    from SneppX_ALG.interface_bindings.crypto_hash import Blake3

    assert (
        Blake3().hash(b"abc").hex()
        == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
    )
    assert Blake3().hash(b"abc") != hashlib.sha256(b"abc").digest()
    assert Blake3().derive_key(CONTEXT, b"data", 64) == blake3_derive_key(
        CONTEXT, b"data", 64
    )
    assert Blake3(key=KEY).hash(b"abc") != Blake3(key=b"\x00" * 32).hash(b"abc")
