import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.firewall_network import FirewallNetwork, SlidingWindowCounter, ConnectionTracker


class TestSlidingWindowCounter:
    def test_allow_within_limit(self):
        c = SlidingWindowCounter(3, 10.0)
        assert c.allow("ip1")
        assert c.allow("ip1")
        assert c.allow("ip1")

    def test_block_at_limit(self):
        c = SlidingWindowCounter(2, 10.0)
        assert c.allow("ip1")
        assert c.allow("ip1")
        assert not c.allow("ip1")

    def test_window_expires(self):
        c = SlidingWindowCounter(1, 0.01)
        assert c.allow("ip1")
        time.sleep(0.02)
        assert c.allow("ip1")


class TestConnectionTracker:
    def test_tracks_connection(self):
        t = ConnectionTracker(max_connections=5, timeout=60)
        assert t.check(0x01020304)

    def test_evicts_oldest(self):
        t = ConnectionTracker(max_connections=3, timeout=60)
        t.check(1)
        t.check(2)
        t.check(3)
        t.check(4)
        assert 1 not in t._table


class TestFirewallNetwork:
    def test_allow_all_by_default(self):
        fw = FirewallNetwork()
        result = fw.check_request("192.168.1.1")
        assert result.allowed

    def test_denylist_blocks(self):
        fw = FirewallNetwork(denylist=["10.0.0.0/8"])
        result = fw.check_request("10.0.0.1")
        assert not result.allowed
        assert result.status_code == 403

    def test_allowlist_blocks_unknown(self):
        fw = FirewallNetwork(allowlist=["192.168.0.0/16"])
        result = fw.check_request("10.0.0.1")
        assert not result.allowed
        result = fw.check_request("192.168.1.1")
        assert result.allowed

    def test_rate_limit_exceeded(self):
        fw = FirewallNetwork(rate_limit_max=2, rate_window_seconds=60.0)
        assert fw.check_request("10.0.0.1").allowed
        assert fw.check_request("10.0.0.1").allowed
        result = fw.check_request("10.0.0.1")
        assert not result.allowed
        assert result.status_code == 429

    def test_invalid_ip(self):
        fw = FirewallNetwork()
        result = fw.check_request("not-an-ip")
        assert not result.allowed
        assert result.status_code == 400

    def test_summary(self):
        fw = FirewallNetwork(denylist=["10.0.0.0/8"])
        s = fw.summary
        assert "denylist" in s
        assert s["rate_limit_max"] == 100
