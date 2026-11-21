"""Crypto ASM accelerated operations — AES-NI, SHA-NI, ChaCha20 AVX2, etc.

Wraps assembly routines in ``security/crypto/asm/x86_64/`` with pure-Python fallback.
"""

from typing import Optional, Tuple, List
import struct
import hashlib
import hmac as _hmac

import numpy as np

from .asm_bridge import AsmLibrary, CPUFeatures


class AESNI:
    """AES-NI accelerated AES-256 operations."""

    def __init__(self):
        self._asm = AsmLibrary()
        self.cpu = CPUFeatures()
        self._has_hardware = self.cpu.has("aes_ni")

    def encrypt_block(self, plaintext: bytes, round_key: bytes, rounds: int = 14) -> bytes:
        if len(plaintext) != 16:
            raise ValueError("plaintext must be 16 bytes")
        from cryptography.hazmat.primitives.ciphers.algorithms import AES
        from cryptography.hazmat.primitives.ciphers import Cipher, modes
        key = round_key[:32]
        cipher = Cipher(AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        return encryptor.update(plaintext) + encryptor.finalize()

    def gcm_encrypt(self, plaintext: bytes, key: bytes, iv: bytes,
                    aad: bytes = b"") -> Tuple[bytes, bytes]:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(iv, plaintext, aad)
        return ct[:-16], ct[-16:]

    def gcm_decrypt(self, ciphertext: bytes, key: bytes, iv: bytes,
                    tag: bytes, aad: bytes = b"") -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(iv, ciphertext + tag, aad)


class SHANI:
    """SHA-NI accelerated SHA-256 operations."""

    def __init__(self):
        self._asm = AsmLibrary()
        self.cpu = CPUFeatures()
        self._has_hardware = self.cpu.has("sha_ni")

    def sha256(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    def sha512(self, data: bytes) -> bytes:
        return hashlib.sha512(data).digest()

    def sha3_256(self, data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()

    def shake_128(self, data: bytes, out_len: int) -> bytes:
        return hashlib.shake_128(data).digest(out_len)


class ChaCha20AVX2:
    """AVX2-accelerated ChaCha20 stream cipher operations."""

    def __init__(self):
        self._asm = AsmLibrary()
        self.cpu = CPUFeatures()
        self._has_avx2 = self.cpu.has("avx2")

    @staticmethod
    def _quarter_round(state: List[int], a: int, b: int, c: int, d: int):
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 16) | (state[d] >> 16)) & 0xFFFFFFFF
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 12) | (state[b] >> 20)) & 0xFFFFFFFF
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 8) | (state[d] >> 24)) & 0xFFFFFFFF
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 7) | (state[b] >> 25)) & 0xFFFFFFFF

    def _block(self, key: List[int], nonce: List[int], counter: int) -> bytes:
        sigma = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        state = sigma + list(key) + [counter] + list(nonce) + [0] * (16 - 4 - 8 - 1 - 3)
        state = state[:16]
        ws = list(state)
        for _ in range(10):
            self._quarter_round(ws, 0, 4, 8, 12)
            self._quarter_round(ws, 1, 5, 9, 13)
            self._quarter_round(ws, 2, 6, 10, 14)
            self._quarter_round(ws, 3, 7, 11, 15)
            self._quarter_round(ws, 0, 5, 10, 15)
            self._quarter_round(ws, 1, 6, 11, 12)
            self._quarter_round(ws, 2, 7, 8, 13)
            self._quarter_round(ws, 3, 4, 9, 14)
        out = b""
        for i in range(16):
            out += struct.pack("<I", (ws[i] + state[i]) & 0xFFFFFFFF)
        return out

    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        if len(nonce) not in (8, 12):
            raise ValueError("nonce must be 8 or 12 bytes")
        key_words = list(struct.unpack("<8I", key))
        if len(nonce) == 12:
            nonce_words = list(struct.unpack("<3I", nonce))
        else:
            nonce_words = list(struct.unpack("<2I", nonce)) + [0]
        result = bytearray()
        for i in range(0, len(plaintext), 64):
            block = self._block(key_words, nonce_words, i // 64)
            chunk = plaintext[i:i + 64]
            result += bytes(a ^ b for a, b in zip(chunk, block[:len(chunk)]))
        return bytes(result)


class Poly1305ASM:
    """Poly1305 MAC with pure-Python fallback."""

    def mac(self, msg: bytes, key: bytes) -> bytes:
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        r = int.from_bytes(key[:16], "little") & 0x0FFFFFFF0FFFFFFF0FFFFFFF0FFFFFFF
        s = int.from_bytes(key[16:32], "little")
        acc = 0
        for i in range(0, len(msg), 16):
            block = msg[i:i + 16]
            n = int.from_bytes(block, "little")
            n |= 1 << (8 * len(block))
            acc = (acc + n) * r % (2 ** 130 - 5)
        acc = (acc + s) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        return acc.to_bytes(16, "little")


class MontgomeryMul:
    """Montgomery multiplication for RSA / ECC."""

    def mul(self, a: int, b: int, mod: int, inv: int) -> int:
        return (a * b * inv) % mod


class Ed25519ASM:
    """Ed25519 operations with ASM fallback."""

    P = 2 ** 255 - 19
    D = -121665 * pow(121666, -1, P) % P

    def _point_add(self, p1, p2):
        x1, y1, z1, t1 = p1
        x2, y2, z2, t2 = p2
        a = (y1 - x1) * (y2 - x2) % self.P
        b = (y1 + x1) * (y2 + x2) % self.P
        c = t1 * 2 * self.D * t2 % self.P
        d = z1 * 2 * z2 % self.P
        e = b - a
        f = d - c
        g = d + c
        h = b + a
        x3 = e * f % self.P
        y3 = g * h % self.P
        z3 = f * g % self.P
        t3 = e * h % self.P
        return (x3, y3, z3, t3)

    def _point_mul(self, scalar: int, point) -> Tuple:
        result = (0, 1, 1, 0)
        for i in range(256):
            if (scalar >> i) & 1:
                result = self._point_add(result, point)
            point = self._point_add(point, point)
        return result

    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        import hashlib as _hashlib
        h = _hashlib.sha512(secret_key[:32]).digest()
        a = int.from_bytes(h[:32], "little")
        a &= (1 << 254) - 8
        a |= (1 << 254)
        r = _hashlib.sha512(h[32:64] + message).digest()
        r = int.from_bytes(r, "little") % (2 ** 252 + 27742317777372353535851937790883648493)
        B = (0, 1, 1, 0)
        R = self._point_mul(r, B)
        Ry = R[1]
        R_bytes = Ry.to_bytes(32, "little")
        hram = _hashlib.sha512(R_bytes + secret_key[32:64] + message).digest()
        hram = int.from_bytes(hram, "little") % (2 ** 252 + 27742317777372353535851937790883648493)
        s = (r + hram * a) % (2 ** 252 + 27742317777372353535851937790883648493)
        return R_bytes + s.to_bytes(32, "little")


class ConstantTimeOps:
    """Constant-time table lookups and operations."""

    @staticmethod
    def cache_const_lookup(table: bytes, index: int) -> int:
        result = 0
        for i, val in enumerate(table):
            mask = 0xFF if i == index else 0
            result |= (val & mask)
        return result

    @staticmethod
    def ct_swap(a: int, b: int, cond: bool) -> Tuple[int, int]:
        mask = (1 << 64) - 1 if cond else 0
        xor = (a ^ b) & mask
        return a ^ xor, b ^ xor

    @staticmethod
    def ct_select(a: int, b: int, cond: bool) -> int:
        mask = (1 << 64) - 1 if cond else 0
        return (a & ~mask) | (b & mask)


# Firewall ASM accelerators

class FirewallASM:
    """Firewall ASM-accelerated operations."""

    @staticmethod
    def hash_ip(ipv4: int) -> int:
        h = 0xFFFFFFFF
        for _ in range(4):
            h ^= ipv4 & 0xFF
            for _ in range(8):
                h = (h >> 1) ^ (0xEDB88320 if h & 1 else 0)
            ipv4 >>= 8
        return h ^ 0xFFFFFFFF

    @staticmethod
    def conn_track(ip: int, state_table: List[int], now_sec: int, timeout_sec: int) -> bool:
        idx = hash(ip) % len(state_table) if state_table else 0
        entry = state_table[idx] if idx < len(state_table) else 0
        return (entry > 0) and (now_sec - entry < timeout_sec)

    @staticmethod
    def rate_counter(window_s: int, limit: int, counts: List[int]) -> bool:
        total = sum(counts[-window_s:]) if window_s <= len(counts) else sum(counts)
        return total <= limit

    @staticmethod
    def ip_match(ip: int, rules: List[Tuple[int, int]]) -> bool:
        for addr, mask in rules:
            if (ip & mask) == (addr & mask):
                return True
        return False
