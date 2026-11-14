"""TLS transport policy validation and certificate-pin regression tests."""

import hashlib
import ssl

import pytest

from SneppX_ALG.interface_bindings.firewall_transport import FirewallTransport


def test_legacy_integer_two_maps_to_tls_12():
    firewall = FirewallTransport(min_version=2)
    assert firewall.min_version == ssl.TLSVersion.TLSv1_2


def test_tls_13_string_alias_is_supported():
    firewall = FirewallTransport(min_version="TLSv1.3")
    assert firewall.min_version == ssl.TLSVersion.TLSv1_3


def test_tls_11_and_older_are_rejected():
    with pytest.raises(ValueError):
        FirewallTransport(min_version=1)
    with pytest.raises(ValueError):
        FirewallTransport(min_version="TLSv1.1")


def test_unknown_tls_version_is_rejected():
    with pytest.raises(ValueError):
        FirewallTransport(min_version="TLSv9.9")


def test_certificate_pins_are_normalized_and_verified():
    certificate = b"test certificate bytes"
    fingerprint = hashlib.sha256(certificate).hexdigest()
    formatted = ":".join(fingerprint[index:index + 2] for index in range(0, 64, 2))
    firewall = FirewallTransport(
        tls_enabled=True,
        pinned_cert_fingerprints=[formatted.upper()],
    )
    assert firewall.verify_peer_certificate(certificate).allowed
    denied = firewall.verify_peer_certificate(b"different certificate")
    assert not denied.allowed
    assert denied.status_code == 403


def test_client_certificate_requirement_needs_tls():
    with pytest.raises(ValueError):
        FirewallTransport(require_client_cert=True)


def test_certificate_pinning_needs_tls():
    with pytest.raises(ValueError):
        FirewallTransport(pinned_cert_fingerprints=["a" * 64])


def test_malformed_fingerprint_is_rejected():
    with pytest.raises(ValueError):
        FirewallTransport(tls_enabled=True, pinned_cert_fingerprints=["not-a-fingerprint"])


def test_missing_peer_certificate_fails_when_client_cert_required():
    firewall = FirewallTransport(tls_enabled=True, require_client_cert=True)
    result = firewall.verify_peer_certificate(b"")
    assert not result.allowed
    assert result.status_code == 403
