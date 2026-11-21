import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.firewall_transport import FirewallTransport


class TestFirewallTransport:
    def test_allows_when_tls_disabled(self):
        fw = FirewallTransport(tls_enabled=False)
        result = fw.check_request()
        assert result.allowed

    def test_fails_when_tls_required_and_no_ssl(self):
        fw = FirewallTransport(tls_enabled=True)
        try:
            fw._build_context()
        except FileNotFoundError:
            pass
        result = fw.check_request(ssl_object=None)
        assert not result.allowed
        assert result.status_code == 426

    def test_pinned_fingerprint_empty(self):
        fw = FirewallTransport()
        result = fw.verify_peer_certificate(b"")
        assert result.allowed
