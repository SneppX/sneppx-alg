"""Regression tests for default-deny routes and bounded firewall state."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from SneppX_ALG.interface_bindings.firewall import FirewallConfig, FirewallRunner, create_firewall
from SneppX_ALG.interface_bindings.firewall_application import FirewallApplication
from SneppX_ALG.interface_bindings.firewall_network import FirewallNetwork


def _request(runner, path, ip="192.0.2.20"):
    return runner.check_request(client_ip=ip, method="GET", path=path, headers={})


def test_default_deny_allows_only_explicit_routes_and_prefixes():
    runner = FirewallRunner(FirewallConfig(
        block_all_by_default=True,
        application={"allowed_paths": ["/health", "/v1/*"]},
    ))
    assert _request(runner, "/health").allowed
    assert _request(runner, "/v1/models").allowed
    assert _request(runner, "/v1").allowed
    denied = _request(runner, "/admin")
    assert not denied.allowed
    assert denied.status_code == 403
    assert "not allowlisted" in denied.reason
    assert runner.summary["allowed"] == 3
    assert runner.summary["denied"] == 1


def test_default_deny_without_routes_blocks_every_path():
    runner = FirewallRunner(FirewallConfig(block_all_by_default=True))
    result = _request(runner, "/health")
    assert not result.allowed
    assert result.status_code == 403


def test_allowed_path_patterns_reject_traversal_and_unsupported_wildcards():
    with pytest.raises(ValueError):
        FirewallApplication(allowed_paths=["/admin/../secret"])
    with pytest.raises(ValueError):
        FirewallApplication(allowed_paths=["/v1/*/admin"])


def test_cli_override_can_configure_default_deny_routes():
    runner = create_firewall(
        env_overrides=False,
        block_all_by_default=True,
        allowed_paths=["/ready"],
    )
    assert _request(runner, "/ready").allowed
    assert not _request(runner, "/private").allowed


def test_blocked_ip_registry_has_a_hard_capacity_limit():
    network = FirewallNetwork(
        rate_limit={"enabled": True, "max_requests": 1, "window_seconds": 60},
        max_blocked_ips=1,
    )
    assert network.check_request("192.0.2.1").allowed
    assert not network.check_request("192.0.2.1").allowed
    assert network.check_request("192.0.2.2").allowed
    assert not network.check_request("192.0.2.2").allowed
    assert network.summary["blocked_ips"] == 1
    assert network.summary["max_blocked_ips"] == 1
    assert not network.check_request("192.0.2.2").allowed


def test_blocked_ip_registry_rejects_zero_capacity():
    with pytest.raises(ValueError):
        FirewallNetwork(max_blocked_ips=0)


def test_firewall_statistics_are_consistent_under_concurrent_requests():
    runner = FirewallRunner()

    def run(index):
        return _request(runner, "/health", ip=f"198.51.100.{index + 1}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(32)))
    assert all(result.allowed for result in results)
    assert runner.summary["allowed"] == 32
    assert runner.summary["denied"] == 0
