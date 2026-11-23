"""Regression tests for the standalone CTM utility tools."""

import json
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[2] / "ctm_tools" / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import ctm_count_functions  # noqa: E402
import ctm_count_headers  # noqa: E402
import ctm_diff_loc  # noqa: E402


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


def test_count_headers_classifies_headers_and_sources(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (tmp_path / "src" / "b.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "src" / "c.hpp").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "notes.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.h").write_text("x\n", encoding="utf-8")

    report = ctm_count_headers.count_headers(tmp_path)

    assert report["header_files"] == 2
    assert report["source_files"] == 1
    assert report["headers_per_source"] == 2.0
    assert report["files_scanned"] == 4
    assert report["headers"] == ["src/b.h", "src/c.hpp"]
    assert report["by_language"]["Python"] == 1


def test_diff_loc_reports_per_language_and_total_deltas(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "a.py").write_text("x = 1\n", encoding="utf-8")
    (new / "a.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    (new / "b.md").write_text("# Title\n", encoding="utf-8")

    report = ctm_diff_loc.diff_loc(old, new)

    assert report["by_language"]["Python"]["loc"] == 2
    assert report["by_language"]["Markdown"]["loc"] == 1
    expected_bytes = (
        (new / "a.py").stat().st_size
        + (new / "b.md").stat().st_size
        - (old / "a.py").stat().st_size
    )
    assert report["totals"] == {"files": 1, "loc": 3, "bytes": expected_bytes}
    assert f"TOTAL,1,3,{expected_bytes}" in ctm_diff_loc.render(report, "csv")
