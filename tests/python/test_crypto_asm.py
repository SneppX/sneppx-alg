"""Tests for crypto_asm.py — ASM-accelerated crypto operations."""

import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.crypto_asm import (
    AESNI, SHANI, ChaCha20AVX2, Poly1305ASM, MontgomeryMul,
    Ed25519ASM, ConstantTimeOps, FirewallASM
)


def test_aes_encrypt_block():
    aes = AESNI()
    pt = b"\x00" * 16
    rk = b"\x00" * 32
    ct = aes.encrypt_block(pt, rk)
    assert len(ct) == 16


def test_aes_gcm_roundtrip():
    aes = AESNI()
    key = b"\x01" * 32
    iv = b"\x02" * 12
    pt = b"hello aes gcm"
    ct, tag = aes.gcm_encrypt(pt, key, iv)
    assert len(ct) == len(pt)
    assert len(tag) == 16
    dec = aes.gcm_decrypt(ct, key, iv, tag)
    assert dec == pt


def test_sha256():
    sha = SHANI()
    h = sha.sha256(b"test data")
    assert len(h) == 32
    assert h == hashlib.sha256(b"test data").digest()


def test_sha512():
    sha = SHANI()
    h = sha.sha512(b"test data")
    assert len(h) == 64


def test_sha3_256():
    sha = SHANI()
    h = sha.sha3_256(b"test data")
    assert len(h) == 32


def test_chacha20_encrypt():
    chacha = ChaCha20AVX2()
    key = b"\x00" * 32
    nonce = b"\x00" * 12
    pt = b"hello chacha20!"
    ct = chacha.encrypt(key, nonce, pt)
    assert len(ct) == len(pt)
    assert ct != pt
    # decrypt = encrypt with same key/nonce
    pt2 = chacha.encrypt(key, nonce, ct)
    assert pt2 == pt


def test_poly1305_mac():
    poly = Poly1305ASM()
    key = b"\x00" * 32
    msg = b"hello poly1305"
    tag = poly.mac(msg, key)
    assert len(tag) == 16


def test_montgomery_mul():
    mm = MontgomeryMul()
    a, b, mod = 7, 8, 13
    inv = pow(-mod, -1, 2 ** 64) & ((1 << 64) - 1)
    result = mm.mul(a, b, mod, inv)
    assert result == (a * b * inv) % mod


def test_ed25519_sign():
    ed = Ed25519ASM()
    sk = b"\x00" * 64
    sig = ed.sign(b"test message", sk)
    assert len(sig) == 64


def test_ct_cache_const_lookup():
    table = bytes(range(16))
    result = ConstantTimeOps.cache_const_lookup(table, 5)
    assert result == 5


def test_ct_swap():
    a, b = ConstantTimeOps.ct_swap(10, 20, True)
    assert a == 20 and b == 10
    a2, b2 = ConstantTimeOps.ct_swap(10, 20, False)
    assert a2 == 10 and b2 == 20


def test_ct_select():
    assert ConstantTimeOps.ct_select(10, 20, True) == 20
    assert ConstantTimeOps.ct_select(10, 20, False) == 10


def test_firewall_hash_ip():
    h = FirewallASM.hash_ip(0x01020304)
    assert isinstance(h, int)
    assert h != 0


def test_firewall_conn_track():
    result = FirewallASM.conn_track(0x01020304, [100, 200, 0], 150, 60)
    assert isinstance(result, bool)


def test_firewall_rate_counter():
    result = FirewallASM.rate_counter(10, 100, [10] * 10)
    assert result is True
    result2 = FirewallASM.rate_counter(10, 50, [10] * 10)
    assert result2 is False


def test_firewall_ip_match():
    rules = [(0x0A000000, 0xFF000000)]  # 10.0.0.0/8
    assert FirewallASM.ip_match(0x0A000001, rules) is True
    assert FirewallASM.ip_match(0x0B000001, rules) is False


if __name__ == "__main__":
    import hashlib
    test_aes_encrypt_block()
    test_aes_gcm_roundtrip()
    test_sha256()
    test_sha512()
    test_sha3_256()
    test_chacha20_encrypt()
    test_poly1305_mac()
    test_montgomery_mul()
    test_ed25519_sign()
    test_ct_cache_const_lookup()
    test_ct_swap()
    test_ct_select()
    test_firewall_hash_ip()
    test_firewall_conn_track()
    test_firewall_rate_counter()
    test_firewall_ip_match()
    print("All crypto_asm tests passed.")
