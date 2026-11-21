"""Tests for s4_network.py — S4 network security."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import DDoSMitigation, DDoSConfig, IdentityManager, TransportSecurity


def test_ddos_allow():
    cfg = DDoSConfig(max_connections_per_ip=3, max_connections_total=10,
                     token_bucket_capacity=1000, max_requests_per_minute=9999)
    d = DDoSMitigation(cfg)
    for _ in range(3):
        ok, _, _ = d.check_request("1.2.3.4")
        assert ok
    ok, _, _ = d.check_request("1.2.3.4")
    assert not ok


def test_ddos_different_ips():
    cfg = DDoSConfig(max_connections_per_ip=2, max_connections_total=10,
                     token_bucket_capacity=1000, max_requests_per_minute=9999)
    d = DDoSMitigation(cfg)
    ok, _, _ = d.check_request("1.1.1.1")
    assert ok
    ok, _, _ = d.check_request("2.2.2.2")
    assert ok
    ok, _, _ = d.check_request("1.1.1.1")
    assert ok
    ok, _, _ = d.check_request("1.1.1.1")
    assert not ok
    ok, _, _ = d.check_request("2.2.2.2")
    assert ok


def test_ddos_reset_ip():
    cfg = DDoSConfig(max_connections_per_ip=1, max_connections_total=10,
                     token_bucket_capacity=1000, max_requests_per_minute=9999)
    d = DDoSMitigation(cfg)
    d.check_request("1.2.3.4")
    ok, _, _ = d.check_request("1.2.3.4")
    assert not ok
    d.release_connection("1.2.3.4")
    ok, _, _ = d.check_request("1.2.3.4")
    assert ok


def test_ddos_reset_all():
    cfg = DDoSConfig(max_connections_per_ip=1, max_connections_total=10,
                     token_bucket_capacity=1000, max_requests_per_minute=9999)
    d = DDoSMitigation(cfg)
    d.check_request("1.2.3.4")
    d.unban_ip("1.2.3.4")
    d.release_connection("1.2.3.4")
    ok, _, _ = d.check_request("1.2.3.4")
    assert ok


def test_ddos_syn_flood():
    cfg = DDoSConfig(syn_flood_threshold=5, syn_flood_window_seconds=60,
                     token_bucket_capacity=1000, max_requests_per_minute=9999)
    d = DDoSMitigation(cfg)
    # First 4 SYN packets should be allowed (< threshold)
    for _ in range(4):
        ok, _, _ = d.check_request("1.2.3.4", is_syn=True)
        assert ok
    # 5th SYN should trigger flood detection (>= threshold)
    ok, action, _ = d.check_request("1.2.3.4", is_syn=True)
    assert not ok


def test_identity_register():
    im = IdentityManager()
    im.register("device1", b"\x01" * 32, {"location": "dc1"})
    entry = im.lookup("device1")
    assert entry is not None
    assert entry["public_key"] == b"\x01" * 32


def test_identity_lookup_missing():
    im = IdentityManager()
    assert im.lookup("nonexistent") is None


def test_identity_revoke():
    im = IdentityManager()
    im.register("d1", b"\x01" * 32)
    assert im.revoke("d1")
    assert not im.revoke("d1")


def test_transport_cipher_select():
    ts = TransportSecurity()
    ciphers = ["TLS_AES_128_GCM_SHA256", "TLS_CHACHA20_POLY1305_SHA256"]
    selected = ts.select_cipher(ciphers)
    assert selected in ts.supported_ciphers()


def test_transport_min_version():
    ts = TransportSecurity("1.2")
    assert ts.min_version == "1.2"
    ts.min_version = "1.3"
    assert ts.min_version == "1.3"


if __name__ == "__main__":
    test_ddos_allow()
    test_ddos_different_ips()
    test_ddos_reset_ip()
    test_ddos_reset_all()
    test_ddos_syn_flood()
    test_identity_register()
    test_identity_lookup_missing()
    test_identity_revoke()
    test_transport_cipher_select()
    test_transport_min_version()
    print("All s4_network tests passed.")
