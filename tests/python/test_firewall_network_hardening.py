"""Regression tests for network firewall policy and bounded state."""

import pytest

from SneppX_ALG.interface_bindings.firewall_network import (
    ConnectionTracker,
    FirewallNetwork,
    SlidingWindowCounter,
)


def test_ipv4_and_ipv6_addresses_are_supported():
    firewall = FirewallNetwork()
    assert firewall.check_request("192.0.2.10").allowed
    assert firewall.check_request("2001:db8::10").allowed


def test_ipv6_cidr_allowlist_and_denylist():
    firewall = FirewallNetwork(
        allowlist=["2001:db8::/32"],
        denylist=["2001:db8:dead::/48"],
    )
    assert firewall.check_request("2001:db8:1::1").allowed
    result = firewall.check_request("2001:db8:dead::1")
    assert not result.allowed
    assert result.status_code == 403


def test_denylist_takes_precedence_over_allowlist():
    firewall = FirewallNetwork(
        allowlist=["10.0.0.0/8"],
        denylist=["10.1.0.0/16"],
    )
    result = firewall.check_request("10.1.2.3")
    assert not result.allowed
    assert result.reason == "IP in denylist"


def test_explicit_empty_allowlist_fails_closed():
    firewall = FirewallNetwork(allowlist=[], allowlist_enabled=True)
    result = firewall.check_request("192.0.2.1")
    assert not result.allowed
    assert result.status_code == 403


def test_rate_limit_configuration_mapping_is_applied():
    firewall = FirewallNetwork(
        rate_limit={"enabled": True, "max_requests": 1, "window_seconds": 30}
    )
    assert firewall.check_request("192.0.2.2").allowed
    assert firewall.check_request("192.0.2.2").status_code == 429


def test_rate_limiter_uses_sliding_window_and_bounds_keys():
    limiter = SlidingWindowCounter(2, 10, max_keys=2)
    assert limiter.allow("a", now=100)
    assert limiter.allow("a", now=101)
    assert not limiter.allow("a", now=102)
    assert limiter.allow("a", now=111)
    limiter.allow("b", now=111)
    limiter.allow("c", now=111)
    assert limiter.tracked_keys == 2


def test_rate_limiter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SlidingWindowCounter(0)
    with pytest.raises(ValueError):
        SlidingWindowCounter(1, 0)


def test_connection_tracker_refreshes_lru_order_and_bounds_entries():
    tracker = ConnectionTracker(max_connections=2, timeout=30)
    tracker.check("192.0.2.1", now=1)
    tracker.check("192.0.2.2", now=1)
    tracker.check("192.0.2.1", now=2)
    tracker.check("192.0.2.3", now=3)
    assert tracker.size == 2
    assert "192.0.2.1" in tracker._table
    assert "192.0.2.2" not in tracker._table


def test_enabled_geoip_without_provider_fails_closed():
    result = FirewallNetwork(geoip_enabled=True).check_request("192.0.2.4")
    assert not result.allowed
    assert result.status_code == 503


def test_invalid_cidr_is_rejected_at_configuration_time():
    with pytest.raises(ValueError):
        FirewallNetwork(denylist=["not-a-network"])


def test_invalid_client_ip_is_rejected():
    result = FirewallNetwork().check_request("not-an-ip")
    assert not result.allowed
    assert result.status_code == 400


def test_legacy_http_header_port_knocking_is_not_trusted():
    result = FirewallNetwork(knock_ports=[7000, 8000]).check_request(
        "192.0.2.5", knock_header="7000,8000"
    )
    assert not result.allowed
    assert result.status_code == 501
