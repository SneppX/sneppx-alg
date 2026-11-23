"""Hardening regressions for the Python socket transport."""

import inspect
import socket

import numpy as np

from SneppX_ALG.interface_bindings.net_bindings import SocketComm


def _pair(max_message_bytes=1024 * 1024):
    a, b = socket.socketpair()
    left, right = (
        SocketComm(max_message_bytes=max_message_bytes),
        SocketComm(max_message_bytes=max_message_bytes),
    )
    for comm, sock in ((left, a), (right, b)):
        comm._sock = sock
        comm.set_timeout(2.0)
    return left, right


def test_bind_defaults_to_loopback_and_validates_port():
    assert inspect.signature(SocketComm.bind).parameters["host"].default == "127.0.0.1"
    comm = SocketComm()
    comm.create()
    try:
        assert comm.bind(0, host="not-an-ip") == -1
        assert comm.bind(70000) == -1
    finally:
        comm.close()


def test_message_roundtrip_and_size_cap():
    left, right = _pair(max_message_bytes=64)
    try:
        assert left.send(b"sx") == 2
        assert right.recv() == b"sx"
        assert left.send(b"x" * 65) == -1
    finally:
        left.close()
        right.close()


def test_tensor_roundtrip_and_header_validation():
    left, right = _pair()
    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    try:
        assert left.send_tensor(payload) == payload.nbytes + 8
        np.testing.assert_array_equal(right.recv_tensor(), payload)
    finally:
        left.close()
        right.close()

    tiny_left, tiny_right = _pair(max_message_bytes=32)
    try:
        assert tiny_left.send_tensor(np.zeros((4, 4), dtype=np.float32)) == -1
    finally:
        tiny_left.close()
        tiny_right.close()


def test_optional_hmac_rejects_wrong_key():
    left, right = _pair()
    key = b"0123456789abcdef"
    left.set_auth_key(key)
    right.set_auth_key(key)
    try:
        assert left.send(b"authenticated") == len(b"authenticated")
        assert right.recv() == b"authenticated"
        right.set_auth_key(b"fedcba9876543210")
        assert left.send(b"tampered") == len(b"tampered")
        assert right.recv() is None
    finally:
        left.close()
        right.close()
