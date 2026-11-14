"""Security regression tests for application-layer firewall validation."""

from SneppX_ALG.interface_bindings.firewall_application import (
    ConcurrentLimiter,
    FirewallApplication,
)


def test_blocks_double_encoded_path_traversal():
    firewall = FirewallApplication()
    result = firewall.check_request("GET", "/v1/%252e%252e/etc/passwd", {})
    assert not result.allowed
    assert result.status_code == 400


def test_blocks_encoded_null_and_backslash_paths():
    firewall = FirewallApplication()
    assert not firewall.check_request("GET", "/v1/%00admin", {}).allowed
    assert not firewall.check_request("GET", "/v1/%5c..%5csecret", {}).allowed


def test_normalizes_repeated_slashes_without_rejecting_safe_path():
    firewall = FirewallApplication()
    assert firewall.check_request("GET", "/v1//health", {}).allowed


def test_header_names_are_case_insensitive_for_security_checks():
    firewall = FirewallApplication()
    result = firewall.check_request("GET", "/", {"X-Forwarded-For": "203.0.113.8"})
    assert not result.allowed
    assert result.status_code == 400


def test_rejects_duplicate_headers_after_case_normalization():
    firewall = FirewallApplication()
    result = firewall.check_request("GET", "/", {"Host": "example.test", "host": "attacker.test"})
    assert not result.allowed


def test_rejects_content_length_mismatch():
    firewall = FirewallApplication()
    result = firewall.check_request(
        "POST", "/v1/data",
        {"content-type": "application/json", "content-length": "3"},
        body=b"{}",
    )
    assert not result.allowed
    assert result.status_code == 400


def test_rejects_oversized_declared_content_length_before_body_read():
    firewall = FirewallApplication(max_body_size=16)
    result = firewall.check_request(
        "POST", "/v1/data",
        {"content-type": "application/json", "content-length": "100"},
    )
    assert not result.allowed
    assert result.status_code == 413


def test_rejects_header_count_and_value_limit():
    firewall = FirewallApplication(max_header_count=1, max_header_value_length=4)
    assert firewall.check_request("GET", "/", {"host": "a"}).allowed
    too_many = firewall.check_request("GET", "/", {"host": "a", "accept": "*/*"})
    too_large = FirewallApplication(max_header_value_length=4).check_request(
        "GET", "/", {"host": "example.test"}
    )
    assert too_many.status_code == 431
    assert too_large.status_code == 431


def test_benign_command_names_are_not_blocked_as_injection():
    firewall = FirewallApplication()
    result = firewall.check_request(
        "POST", "/v1/search", {"content-type": "application/json"},
        body=b'{"tool": "curl", "description": "download a report"}',
    )
    assert result.allowed


def test_concurrent_limiter_does_not_over_release_or_leak_keys():
    limiter = ConcurrentLimiter(max_per_ip=1)
    assert limiter.acquire("192.0.2.1")
    assert not limiter.acquire("192.0.2.1")
    limiter.release("192.0.2.1")
    limiter.release("192.0.2.1")
    assert limiter.acquire("192.0.2.1")
    limiter.release("192.0.2.1")
    assert limiter.tracked_clients == 0


def test_rejects_malformed_header_values():
    firewall = FirewallApplication()
    result = firewall.check_request("GET", "/", {"host": "example.test\r\nX-Evil: 1"})
    assert not result.allowed


def test_rejects_non_bytes_body():
    result = FirewallApplication().check_request(
        "POST", "/", {"content-type": "application/json"}, body="{}"
    )
    assert not result.allowed
