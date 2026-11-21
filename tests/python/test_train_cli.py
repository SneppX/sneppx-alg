"""Tests for Train CLI."""

from SneppX_ALG.interface_bindings.train_cli import (
    main as train_main, build_model_from_config, build_dummy_data,
)


def test_train_cli_import():
    from SneppX_ALG.interface_bindings import train_cli
    assert train_cli is not None


def test_train_cli_has_main():
    assert callable(train_main)


def test_build_model_from_config():
    config = {"model_type": "mlp", "hidden_dim": 16, "num_classes": 2}
    model = build_model_from_config(config)
    assert model is not None


def test_build_dummy_data():
    from types import SimpleNamespace
    config = SimpleNamespace(training=SimpleNamespace(batch_size=4))
    loader = build_dummy_data(config)
    assert loader is not None


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
