"""Tests for crypto_util.py — BigNum, DRBG, Random, ConstantTime, EntropyPool."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import BigNum, DRBG, Random, ConstantTime, EntropyPool


def test_bignum_from_int():
    bn = BigNum(42)
    assert int(bn) == 42


def test_bignum_from_bytes():
    bn = BigNum.from_bytes(b"\x2a", "little")
    assert int(bn) == 42


def test_bignum_to_bytes():
    bn = BigNum(0xDEAD)
    data = bn.to_bytes(byteorder="big")
    assert data == b"\xde\xad"


def test_bignum_to_bytes_fixed():
    bn = BigNum(5)
    data = bn.to_bytes(4, "little")
    assert data == b"\x05\x00\x00\x00"


def test_bignum_add():
    a = BigNum(10)
    b = BigNum(20)
    assert int(a.add(b)) == 30


def test_bignum_sub():
    a = BigNum(20)
    b = BigNum(5)
    assert int(a.sub(b)) == 15


def test_bignum_mul():
    a = BigNum(6)
    b = BigNum(7)
    assert int(a.mul(b)) == 42


def test_bignum_divmod():
    a = BigNum(42)
    q, r = a.divmod(BigNum(10))
    assert int(q) == 4
    assert int(r) == 2


def test_bignum_mod():
    a = BigNum(42)
    assert int(a.mod(BigNum(10))) == 2


def test_bignum_pow_mod():
    a = BigNum(2)
    e = BigNum(10)
    m = BigNum(1000)
    assert int(a.pow_mod(e, m)) == pow(2, 10, 1000)


def test_bignum_inv_mod():
    a = BigNum(3)
    m = BigNum(7)
    inv = a.inv_mod(m)
    assert inv is not None
    assert int(a.mul(inv).mod(m)) == 1


def test_bignum_zero_inv():
    a = BigNum(0)
    assert a.inv_mod(BigNum(7)) is None


def test_drbg_generate():
    drbg = DRBG(b"entropy_input_16b")
    out = drbg.generate(32)
    assert len(out) == 32


def test_drbg_deterministic():
    drbg1 = DRBG(b"same_seed_12345")
    drbg2 = DRBG(b"same_seed_12345")
    assert drbg1.generate(16) == drbg2.generate(16)


def test_drbg_varying_length():
    drbg = DRBG(b"seed")
    for length in [1, 8, 16, 32, 64]:
        out = drbg.generate(length)
        assert len(out) == length


def test_random_bytes():
    r = Random.bytes(16)
    assert len(r) == 16
    assert isinstance(r, bytes)


def test_random_int():
    r = Random.int(128)
    assert 0 <= r < 2**128


def test_random_range():
    r = Random.range(10, 20)
    assert 10 <= r < 20


def test_random_range_single():
    r = Random.range(5, 5)
    assert r == 5


def test_constanttime_compare_equal():
    assert ConstantTime.compare(b"abc", b"abc")


def test_constanttime_compare_not_equal():
    assert not ConstantTime.compare(b"abc", b"abd")


def test_constanttime_compare_diff_length():
    assert not ConstantTime.compare(b"abc", b"abcd")


def test_constanttime_compare_empty():
    assert ConstantTime.compare(b"", b"")


def test_constanttime_select():
    true_val = b"true"
    false_val = b"false"
    result = ConstantTime.select(1, true_val, false_val)
    assert result == b"true"


def test_entropypool_add():
    ep = EntropyPool()
    ep.add_entropy(b"some_random_data")
    assert ep._pool is not None


def test_entropypool_collect():
    ent = EntropyPool.collect_system_entropy()
    assert len(ent) == 32


def test_entropypool_reseed():
    ep = EntropyPool()
    ep.add_entropy(b"data")
    drbg = DRBG(b"initial")
    ep.reseed(drbg)


if __name__ == "__main__":
    test_bignum_from_int()
    test_bignum_from_bytes()
    test_bignum_to_bytes()
    test_bignum_to_bytes_fixed()
    test_bignum_add()
    test_bignum_sub()
    test_bignum_mul()
    test_bignum_divmod()
    test_bignum_mod()
    test_bignum_pow_mod()
    test_bignum_inv_mod()
    test_bignum_zero_inv()
    test_drbg_generate()
    test_drbg_deterministic()
    test_drbg_varying_length()
    test_random_bytes()
    test_random_int()
    test_random_range()
    test_random_range_single()
    test_constanttime_compare_equal()
    test_constanttime_compare_not_equal()
    test_constanttime_compare_diff_length()
    test_constanttime_compare_empty()
    test_constanttime_select()
    test_entropypool_add()
    test_entropypool_collect()
    test_entropypool_reseed()
    print("All crypto_util tests passed.")
