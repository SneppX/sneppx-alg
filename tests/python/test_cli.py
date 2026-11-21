"""Tests for CLI entry points (eval, quantize, rlhf). Tests are import and help-only
to avoid blocking on model loading or training."""

import sys
import io


def _capture_help(module_main, args=None):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        try:
            if args:
                sys.argv = ["prog"] + args
            else:
                sys.argv = ["prog", "--help"]
            try:
                module_main()
                return sys.stdout.getvalue(), sys.stderr.getvalue(), 0
            except SystemExit as e:
                out = sys.stdout.getvalue()
                err = sys.stderr.getvalue()
                return out, err, e.code if e.code is not None else 0
        except Exception as e:
            return sys.stdout.getvalue(), sys.stderr.getvalue(), str(e)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def test_eval_cli_import():
    from SneppX_ALG.interface_bindings.eval_cli import main
    assert callable(main)
    print("  test_eval_cli_import PASS")


def test_eval_cli_help():
    from SneppX_ALG.interface_bindings.eval_cli import main
    out, err, code = _capture_help(main)
    assert code == 0
    print("  test_eval_cli_help PASS")


def test_quantize_cli_import():
    from SneppX_ALG.interface_bindings.quantize_cli import main
    assert callable(main)
    print("  test_quantize_cli_import PASS")


def test_quantize_cli_help():
    from SneppX_ALG.interface_bindings.quantize_cli import main
    out, err, code = _capture_help(main)
    assert code == 0
    print("  test_quantize_cli_help PASS")


def test_rlhf_cli_import():
    from SneppX_ALG.interface_bindings.rlhf_cli import main
    assert callable(main)
    print("  test_rlhf_cli_import PASS")


def test_rlhf_cli_help():
    from SneppX_ALG.interface_bindings.rlhf_cli import main
    out, err, code = _capture_help(main)
    assert code == 0
    print("  test_rlhf_cli_help PASS")


if __name__ == "__main__":
    test_eval_cli_import()
    test_eval_cli_help()
    test_quantize_cli_import()
    test_quantize_cli_help()
    test_rlhf_cli_import()
    test_rlhf_cli_help()
    print("ALL CLI TESTS PASS")
