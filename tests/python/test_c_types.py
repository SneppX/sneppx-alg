"""Tests for sneppx_serve.c_types (sneppx-alg interface_bindings.c_types)."""

import ctypes

from SneppX_ALG.interface_bindings.c_types import (
    DataBuffer,
    make_struct,
    c_string,
    buffer_from_databuf,
)


def test_databuffer_create():
    db = DataBuffer(size=64)
    assert db is not None
    assert db.size == 64


def test_databuffer_zero_size():
    db = DataBuffer(size=0)
    assert db is not None
    assert db.size == 0


def test_c_string():
    s = c_string("hello")
    assert isinstance(s, bytes)
    assert s == b"hello"


def test_c_string_empty():
    s = c_string("")
    assert isinstance(s, bytes)
    assert s == b""


def test_make_struct():
    St = make_struct("Point", [("x", ctypes.c_int), ("y", ctypes.c_int)])
    assert St is not None
    p = St(x=3, y=4)
    assert p.x == 3
    assert p.y == 4


def test_buffer_from_databuf_none():
    db = DataBuffer(size=16)
    # data is None by default, so buffer_from_databuf returns None
    buf = buffer_from_databuf(db)
    assert buf is None


def test_buffer_from_databuf_with_data():
    arr = (ctypes.c_uint8 * 8)(*[1, 2, 3, 4, 5, 6, 7, 8])
    db = DataBuffer(data=ctypes.cast(arr, ctypes.c_void_p).value, size=8)
    buf = buffer_from_databuf(db)
    assert buf is not None
    assert len(buf) == 8
