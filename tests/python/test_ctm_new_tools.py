"""Regression tests for the standalone CTM utility tools."""

import json
import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[2] / "ctm_tools" / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import ctm_count_functions  # noqa: E402
import ctm_count_headers  # noqa: E402
import ctm_diff_loc  # noqa: E402
import ctm_list_large_files  # noqa: E402
import ctm_recent_files  # noqa: E402
import ctm_todo_scan  # noqa: E402
import ctm_doc_links  # noqa: E402
import ctm_duplicate_files  # noqa: E402
import ctm_churn  # noqa: E402
import ctm_license_inventory  # noqa: E402


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


def test_list_large_files_sorts_and_formats_sizes(tmp_path):
    (tmp_path / "small.bin").write_bytes(b"a" * 10)
    (tmp_path / "large.bin").write_bytes(b"b" * 2048)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "huge.bin").write_bytes(b"c" * 4096)

    report = ctm_list_large_files.list_large_files(tmp_path, min_size=0, limit=2)

    assert [item["path"] for item in report["files"]] == ["large.bin", "small.bin"]
    assert report["files"][0]["size"] == "2.0 KiB"
    assert "2.0 KiB  large.bin" in ctm_list_large_files.render(report, "text")


def test_recent_files_orders_by_mtime_and_honors_limit(tmp_path):
    import os
    import time

    old = tmp_path / "old.py"
    recent = tmp_path / "recent.py"
    ignored = tmp_path / "node_modules" / "ignored.py"
    tmp_path.joinpath("node_modules").mkdir()
    for path in (old, recent, ignored):
        path.write_text("x = 1\n", encoding="utf-8")
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(recent, (now, now))

    report = ctm_recent_files.recent_files(tmp_path, limit=1)

    assert [item["path"] for item in report["files"]] == ["recent.py"]
    assert report["files"][0]["modified"].endswith("+00:00")
    assert len(json.loads(ctm_recent_files.render(report, "json"))["files"]) == 1


def test_all_new_tools_run_as_scripts(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    tools = {
        "ctm_count_functions": [str(tmp_path)],
        "ctm_count_headers": [str(tmp_path)],
        "ctm_diff_loc": [str(tmp_path), str(tmp_path)],
        "ctm_list_large_files": [str(tmp_path)],
        "ctm_recent_files": [str(tmp_path)],
    }
    for tool, args in tools.items():
        result = subprocess.run(
            [sys.executable, str(PY_DIR / f"{tool}.py"), *args, "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        assert payload["tool"] == tool


def test_todo_scan_finds_markers_with_line_numbers(tmp_path):
    (tmp_path / "a.py").write_text(
        "# TODO: add tests\n# FIXME: handle None\nx = 1  # HACK\n", encoding="utf-8"
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.py").write_text("# TODO ignored\n", encoding="utf-8")

    report = ctm_todo_scan.todo_scan(tmp_path)

    assert report["total"] == 3
    assert report["counts"] == {"FIXME": 1, "HACK": 1, "TODO": 1}
    assert report["occurrences"][0]["line"] == 1
    assert all("node_modules" not in item["path"] for item in report["occurrences"])


def test_doc_links_flags_missing_relative_targets_only(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    (tmp_path / "docs" / "index.md").write_text(
        "[ok](../target.md)\n[bad](./missing.md)\n"
        "[external](https://example.com)\n[anchor](../target.md#section)\n",
        encoding="utf-8",
    )

    report = ctm_doc_links.check_links(tmp_path)

    assert report["links_checked"] == 3
    assert report["broken"] == 1
    assert report["issues"] == [{"path": "docs/index.md", "line": 2, "target": "./missing.md"}]


def test_duplicate_files_groups_identical_content(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_bytes(b"identical\n")
    (tmp_path / "b.py").write_bytes(b"identical\n")
    (tmp_path / "c.py").write_bytes(b"different\n")

    report = ctm_duplicate_files.duplicate_files(tmp_path)

    assert len(report["groups"]) == 1
    assert report["groups"][0]["files"] == ["b.py", "src/a.py"]
    assert report["groups"][0]["wasted_bytes"] == 10
    assert report["wasted_bytes"] == 10


def test_churn_aggregates_git_history(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "edit"], cwd=tmp_path, check=True)

    report = ctm_churn.churn(tmp_path)

    assert report["commits"] == 2
    assert report["additions"] == 2
    assert report["deletions"] == 1
    assert report["files"] == [{"path": "a.py", "changes": 2, "added": 2, "deleted": 1}]
    assert "path,changes,added,deleted" in ctm_churn.render(report, "csv")


def test_license_inventory_finds_spdx_and_manifests(tmp_path):
    (tmp_path / "a.py").write_text(
        "# SPDX-License-Identifier: MIT\n# MIT License\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT License\n", encoding="utf-8")

    report = ctm_license_inventory.license_inventory(tmp_path)

    assert report["files_with_license_header"] == 1
    assert report["by_spdx"] == {"MIT": 1}
    assert report["by_heuristic"] == {"MIT License": 1}
    assert report["manifests"] == ["pyproject.toml"]
    assert report["license_files"] == ["LICENSE"]
    assert report["files"][0]["path"] == "a.py"
