"""Pure-Python BLAKE3 (RFC-ish draft-irtf-cfrg-blake3) fallback.

Ported from the official BLAKE3 reference implementation so ``Blake3`` is
never silently downgraded to SHA-256 when the optional ``blake3`` wheel is
not installed. Supports unkeyed hashing, keyed hashing, key derivation, and
extendable output.
"""

import struct
from typing import List, Sequence

OUT_LEN = 32
BLOCK_LEN = 64
CHUNK_LEN = 1024

CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4
DERIVE_KEY_CONTEXT = 1 << 5
DERIVE_KEY_MATERIAL = 1 << 6

_MASK32 = 0xFFFFFFFF
_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)


def _rotr32(value: int, count: int) -> int:
    return ((value >> count) | (value << (32 - count))) & _MASK32


def _g(state: List[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    state[a] = (state[a] + state[b] + mx) & _MASK32
    state[d] = _rotr32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotr32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + my) & _MASK32
    state[d] = _rotr32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotr32(state[b] ^ state[c], 7)


def _round(state: List[int], m: Sequence[int]) -> None:
    _g(state, 0, 4, 8, 12, m[0], m[1])
    _g(state, 1, 5, 9, 13, m[2], m[3])
    _g(state, 2, 6, 10, 14, m[4], m[5])
    _g(state, 3, 7, 11, 15, m[6], m[7])
    _g(state, 0, 5, 10, 15, m[8], m[9])
    _g(state, 1, 6, 11, 12, m[10], m[11])
    _g(state, 2, 7, 8, 13, m[12], m[13])
    _g(state, 3, 4, 9, 14, m[14], m[15])


def _compress(
    chaining_value: Sequence[int],
    block_words: Sequence[int],
    counter: int,
    block_len: int,
    flags: int,
) -> List[int]:
    state = list(chaining_value) + [
        _IV[0],
        _IV[1],
        _IV[2],
        _IV[3],
        counter & _MASK32,
        (counter >> 32) & _MASK32,
        block_len,
        flags,
    ]
    block = list(block_words)
    for _ in range(7):
        _round(state, block)
        block = [block[i] for i in _MSG_PERMUTATION]
    for i in range(8):
        state[i] ^= state[i + 8]
        state[i + 8] ^= chaining_value[i]
    return state


def _words_from_bytes(data: bytes) -> List[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def _words_to_bytes(words: Sequence[int]) -> bytes:
    return b"".join(word.to_bytes(4, "little") for word in words)


class _Output:
    """State just before choosing a chaining value or root output."""

    def __init__(
        self,
        input_cv: Sequence[int],
        block_words: Sequence[int],
        counter: int,
        block_len: int,
        flags: int,
    ):
        self.input_cv = tuple(input_cv)
        self.block_words = tuple(block_words)
        self.counter = counter
        self.block_len = block_len
        self.flags = flags

    def chaining_value(self) -> List[int]:
        return _compress(
            self.input_cv,
            self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        )[:8]

    def root_output_bytes(self, length: int) -> bytes:
        out = bytearray()
        output_block_counter = 0
        while len(out) < length:
            words = _compress(
                self.input_cv,
                self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | ROOT,
            )
            out.extend(_words_to_bytes(words))
            output_block_counter += 1
        return bytes(out[:length])


def _chunk_output(
    chunk: bytes, chunk_counter: int, key_words: Sequence[int], flags: int
) -> _Output:
    blocks = [chunk[i : i + BLOCK_LEN] for i in range(0, len(chunk), BLOCK_LEN)] or [
        b""
    ]
    cv = list(key_words)
    output = None
    for index, block in enumerate(blocks):
        block_words = _words_from_bytes(block.ljust(BLOCK_LEN, b"\0"))
        block_flags = flags
        if index == 0:
            block_flags |= CHUNK_START
        if index == len(blocks) - 1:
            block_flags |= CHUNK_END
        compressed = _compress(cv, block_words, chunk_counter, len(block), block_flags)
        if index == len(blocks) - 1:
            output = _Output(cv, block_words, chunk_counter, len(block), block_flags)
        else:
            cv = compressed[:8]
    assert output is not None
    return output


def _hash(
    data: bytes,
    length: int,
    key_words: Sequence[int],
    flags: int,
) -> bytes:
    if length < 0:
        raise ValueError("length must be non-negative")
    chunks = [data[i : i + CHUNK_LEN] for i in range(0, len(data), CHUNK_LEN)] or [b""]
    outputs = [
        _chunk_output(chunk, index, key_words, flags)
        for index, chunk in enumerate(chunks)
    ]
    while len(outputs) > 1:
        reduced = []
        for i in range(0, len(outputs) - 1, 2):
            left_cv = outputs[i].chaining_value()
            right_cv = outputs[i + 1].chaining_value()
            reduced.append(
                _Output(
                    key_words,
                    left_cv + right_cv,
                    0,
                    BLOCK_LEN,
                    PARENT | flags,
                )
            )
        if len(outputs) % 2:
            reduced.append(outputs[-1])
        outputs = reduced
    return outputs[0].root_output_bytes(length)


def blake3_hash(data: bytes, length: int = OUT_LEN) -> bytes:
    """Unkeyed BLAKE3 hash, with extendable output."""
    return _hash(bytes(data), length, _IV, 0)


def blake3_keyed_hash(key: bytes, data: bytes, length: int = OUT_LEN) -> bytes:
    """BLAKE3 keyed hash (key must be exactly 32 bytes)."""
    if len(key) != 32:
        raise ValueError("BLAKE3 key must be 32 bytes")
    return _hash(bytes(data), length, _words_from_bytes(key), KEYED_HASH)


def blake3_derive_key(
    context: str, key_material: bytes, length: int = OUT_LEN
) -> bytes:
    """BLAKE3 key derivation using a hardcoded, application-specific context."""
    context_key = _hash(context.encode("utf-8"), 32, _IV, DERIVE_KEY_CONTEXT)
    return _hash(
        bytes(key_material), length, _words_from_bytes(context_key), DERIVE_KEY_MATERIAL
    )


__all__ = ["blake3_hash", "blake3_keyed_hash", "blake3_derive_key"]
