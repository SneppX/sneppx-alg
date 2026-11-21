"""sneppx-onnx CLI entry point (delegates to the standalone ``onnx`` package).

Registers the ``sneppx-onnx`` console script. The full ONNX toolkit lives in
the top-level ``onnx/`` package (numpy-only); this thin wrapper adds the repo
root to ``sys.path`` if needed and forwards to ``onnx.cli.main``.
"""

import os
import sys


def _ensure_onnx_importable() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    # when installed as a wheel, onnx/ may be packaged separately
    if root not in sys.path:
        sys.path.insert(0, os.path.dirname(root))


def main(argv=None) -> int:
    _ensure_onnx_importable()
    try:
        from onnx.cli import main as _cli_main
    except ImportError:
        raise SystemExit(
            "sneppx-onnx requires the standalone onnx/ package "
            "(pip install -e . from the repo root)"
        )
    return _cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
