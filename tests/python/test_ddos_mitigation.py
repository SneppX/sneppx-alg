"""Tests for DDoS Mitigation (S4 Network Security)."""

from SneppX_ALG.interface_bindings.ddos_mitigation import (
    DDoSAction, DDoSConfig, TokenBucket, SlidingWindowRateLimiter,
    IPReputationManager, ConnectionTracker, SYNFloodDetector,
    AdaptiveThresholdManager, DDoSMitigation,
    check_request, release_connection, ban_ip, unban_ip,
    get_ddos_stats, get_ip_info,
)


def test_ddos_action_values():
    assert DDoSAction.MONITOR.value == "monitor"
    assert DDoSAction.BLOCK.value == "block"


def test_ddos_config_defaults():
    cfg = DDoSConfig()
    assert cfg.max_connections_per_ip == 1000
    assert cfg.syn_flood_threshold == 1000


def test_ddos_config_custom():
    cfg = DDoSConfig(max_requests_per_second=500, token_bucket_capacity=2000)
    assert cfg.max_requests_per_second == 500


def test_token_bucket_init():
    tb = TokenBucket(capacity=100, refill_rate=10.0)
    assert tb.available() <= 100


def test_token_bucket_consume():
    tb = TokenBucket(capacity=10, refill_rate=100.0)
    assert tb.consume(5) is True


def test_token_bucket_consume_over_limit():
    tb = TokenBucket(capacity=5, refill_rate=100.0)
    assert tb.consume(10) is False


def test_sliding_window_rate_limiter():
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    assert limiter.check("test_ip") is True


def test_sliding_window_rate_limit_reached():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.check("ip1") is True
    assert limiter.check("ip1") is True
    assert limiter.check("ip1") is False


def test_ip_reputation_manager():
    mgr = IPReputationManager()
    assert mgr.get_score("1.2.3.4") == 0.0


def test_ip_reputation_add():
    mgr = IPReputationManager()
    mgr.record_event("1.2.3.4", "bad_request", severity=10.0)
    assert mgr.get_score("1.2.3.4") >= 10.0


def test_ip_reputation_ban_threshold():
    mgr = IPReputationManager(ban_threshold=30)
    mgr.record_event("bad_ip", "attack", severity=35.0)
    assert mgr.is_banned("bad_ip") is True


def test_connection_tracker():
    tracker = ConnectionTracker()
    assert tracker.get_count("1.2.3.4") == 0


def test_connection_tracker_track():
    tracker = ConnectionTracker(max_per_ip=5)
    assert tracker.acquire("1.2.3.4") is True
    assert tracker.get_count("1.2.3.4") == 1


def test_connection_tracker_release():
    tracker = ConnectionTracker()
    tracker.acquire("1.2.3.4")
    tracker.release("1.2.3.4")
    assert tracker.get_count("1.2.3.4") == 0


def test_syn_flood_detector():
    detector = SYNFloodDetector(threshold=10, window=5.0)
    assert detector.get_rate() >= 0.0


def test_syn_flood_detector_trigger():
    detector = SYNFloodDetector(threshold=3, window=60.0)
    for _ in range(4):
        detector.record_syn()
    assert detector.get_rate() > 0


def test_adaptive_threshold_manager():
    mgr = AdaptiveThresholdManager(learning_window=10)
    assert mgr.get_baseline() >= 0.0


def test_adaptive_threshold_adapt():
    mgr = AdaptiveThresholdManager(learning_window=5)
    mgr.record_request()
    assert mgr.get_threshold_multiplier() > 0


def test_ddos_mitigation_init():
    mitigator = DDoSMitigation()
    assert mitigator is not None


def test_ddos_mitigation_check():
    mitigator = DDoSMitigation()
    allowed, action, msg = mitigator.check_request("1.2.3.4")
    assert isinstance(allowed, bool)
    assert isinstance(action, DDoSAction)


def test_ddos_mitigation_ban():
    mitigator = DDoSMitigation()
    mitigator.ban_ip("10.0.0.99")
    info = mitigator.get_ip_info("10.0.0.99")
    assert info is not None


def test_ddos_mitigation_stats():
    mitigator = DDoSMitigation()
    stats = mitigator.get_stats()
    assert isinstance(stats, dict)


def test_check_request_default():
    allowed, action, msg = check_request("1.2.3.4")
    assert isinstance(allowed, bool)
    assert isinstance(action, DDoSAction)


def test_release_connection():
    release_connection("1.2.3.4")


def test_ban_ip():
    ban_ip("10.0.0.1")


def test_unban_ip():
    unban_ip("10.0.0.1")


def test_get_ddos_stats():
    stats = get_ddos_stats()
    assert isinstance(stats, dict)


def test_get_ip_info():
    info = get_ip_info("1.2.3.4")
    assert isinstance(info, dict)


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
