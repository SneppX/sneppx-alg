"""Tests for Firewall Orchestrator (S4)."""

from SneppX_ALG.interface_bindings.firewall import (
    FirewallConfig, FirewallRunner, create_firewall,
)


def test_firewall_config_defaults():
    cfg = FirewallConfig()
    assert cfg.enabled is True
    assert cfg.block_all_by_default is False


def test_firewall_config_custom():
    cfg = FirewallConfig(enabled=False, block_all_by_default=True)
    assert cfg.enabled is False
    assert cfg.block_all_by_default is True


def test_firewall_runner_init():
    runner = FirewallRunner()
    assert runner is not None


def test_firewall_runner_check():
    runner = FirewallRunner()
    result = runner.check_request(client_ip="1.2.3.4", path="/api/test")
    assert result is not None


def test_firewall_runner_check_default():
    runner = FirewallRunner()
    result = runner.check_request()
    assert result is not None


def test_create_firewall_default():
    fw = create_firewall()
    assert fw is not None


def test_create_firewall_with_cli_overrides():
    fw = create_firewall(env_overrides=False, enabled=True)
    assert fw is not None


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
