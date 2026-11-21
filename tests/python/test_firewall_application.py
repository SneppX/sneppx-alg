import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.firewall_application import FirewallApplication


class TestFirewallApplication:
    def test_allows_valid_request(self):
        fw = FirewallApplication()
        result = fw.check_request("GET", "/v1/health", {"host": "localhost"})
        assert result.allowed

    def test_blocks_invalid_method(self):
        fw = FirewallApplication(allowed_methods={"GET"})
        result = fw.check_request("DELETE", "/v1/health", {"host": "localhost"})
        assert not result.allowed
        assert result.status_code == 405

    def test_blocks_path_traversal(self):
        fw = FirewallApplication()
        result = fw.check_request("GET", "/v1/../etc/passwd", {"host": "localhost"})
        assert not result.allowed
        assert result.status_code == 400

    def test_blocks_large_body(self):
        fw = FirewallApplication(max_body_size=100)
        result = fw.check_request("POST", "/v1/generate", {"host": "localhost", "content-type": "application/json"}, body=b"x" * 101)
        assert not result.allowed
        assert result.status_code == 413

    def test_blocks_sql_injection(self):
        fw = FirewallApplication(enable_injection_filter=True)
        result = fw.check_request("POST", "/v1/generate", {"host": "localhost", "content-type": "application/json"}, body=b"SELECT * FROM users")
        assert not result.allowed

    def test_blocks_script_injection(self):
        fw = FirewallApplication(enable_injection_filter=True)
        result = fw.check_request("POST", "/v1/generate", {"host": "localhost", "content-type": "application/json"}, body=b"<script>alert(1)</script>")
        assert not result.allowed

    def test_normalize_path_double_slash(self):
        fw = FirewallApplication()
        result = fw.check_request("GET", "/v1//health", {"host": "localhost"})
        assert result.allowed

    def test_content_type_blocked(self):
        fw = FirewallApplication(allowed_content_types={"application/json"})
        result = fw.check_request("POST", "/v1/generate", {"host": "localhost", "content-type": "text/html"}, body=b"{}")
        assert not result.allowed

    def test_suspicious_header(self):
        fw = FirewallApplication()
        result = fw.check_request("GET", "/v1/health", {"host": "localhost", "x-forwarded-for": "evil"})
        assert not result.allowed

    def test_release_concurrent(self):
        fw = FirewallApplication(max_concurrent_per_ip=1)
        result1 = fw.check_request("GET", "/", {"host": "localhost"}, client_ip="10.0.0.1")
        assert result1.allowed
        result2 = fw.check_request("GET", "/", {"host": "localhost"}, client_ip="10.0.0.1")
        assert not result2.allowed
        fw.release_concurrent("10.0.0.1")
        result3 = fw.check_request("GET", "/", {"host": "localhost"}, client_ip="10.0.0.1")
        assert result3.allowed
