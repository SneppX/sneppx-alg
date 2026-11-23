"""Regression tests for the standalone CTM utility tools."""

import json
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[2] / "ctm_tools" / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import ctm_count_functions  # noqa: E402


def test_count_functions_counts_python_js_and_c_definitions(tmp_path):
    (tmp_path / "a.py").write_text(
        "def one():\n    pass\n\nasync def two():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "b.js").write_text(
        "function f() {}\nconst g = (x) => x;\nif (x) { }\n", encoding="utf-8"
    )
    (tmp_path / "c.c").write_text(
        "int add(int a, int b) {\n  if (a) { return b; }\n  return a;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("def nope(): pass\n", encoding="utf-8")

    report = ctm_count_functions.count_functions(tmp_path)

    assert report["functions"] == 5
    assert report["by_language"] == {"C": 1, "JavaScript": 2, "Python": 2}
    assert report["files_scanned"] == 3
    assert report["heuristic"] is True
    assert json.loads(ctm_count_functions.render(report, "json"))["functions"] == 5
