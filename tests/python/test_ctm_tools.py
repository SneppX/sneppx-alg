"""Regression tests for the standalone CTM scanner."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PY_DIR = Path(__file__).resolve().parents[2] / "ctm_tools" / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

MODULE_PATH = PY_DIR / "ctm.py"
SPEC = importlib.util.spec_from_file_location("sneppx_ctm", MODULE_PATH)
ctm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctm)

LARGE_MODULE_PATH = MODULE_PATH.parent / "ctm_find_large_files.py"
LARGE_SPEC = importlib.util.spec_from_file_location("ctm_large_files", LARGE_MODULE_PATH)
ctm_large_files = importlib.util.module_from_spec(LARGE_SPEC)
LARGE_SPEC.loader.exec_module(ctm_large_files)


def test_scan_counts_source_and_blank_lines_but_skips_generated_and_binary(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n\nprint(x)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / "src" / "blob.py").write_bytes(b"\x00\x01\x02")
    (tmp_path / "node_modules" / "ignored.py").write_text("bad = True\n", encoding="utf-8")

    report = ctm.scan(tmp_path)

    assert report["files"] == 2
    assert report["directories"] == 1
    assert report["lines_of_code"] == 4
    assert report["languages"]["Python"]["loc"] == 3
    assert report["languages"]["Markdown"]["loc"] == 1
    assert report["skipped_binary_or_symlink"] == 1


def test_iter_all_files_includes_binaries_but_skips_ignored_dirs(tmp_path):
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "blob.exe").write_bytes(b"\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("x\n", encoding="utf-8")

    assert sorted(p.name for p in ctm.iter_all_files(tmp_path)) == ["a.py", "blob.exe"]


def test_scan_reports_skipped_by_reason_breakdown(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "big.png").write_bytes(b"x")
    (tmp_path / "notes.custom").write_text("x\n", encoding="utf-8")

    report = ctm.scan(tmp_path)

    assert report["skipped_by_reason"] == {"binary_suffix": 1, "unsupported_suffix": 1}

    real = tmp_path / "real.py"
    real.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this platform")

    report = ctm.scan(tmp_path)
    assert report["skipped_by_reason"]["symlink_file"] == 1


def test_render_html_includes_bar_chart_and_skip_breakdown(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"x")

    report = ctm.scan(tmp_path)
    rendered = ctm.render_html(report)

    assert "Language breakdown" in rendered
    assert "bar-row" in rendered
    assert "Skipped-file breakdown" in rendered
    assert "unsupported_suffix" in rendered
    assert "<script>alert(1)</script>" not in rendered


def test_line_reader_handles_universal_newlines(tmp_path):
    source = tmp_path / "mixed.txt"
    source.write_bytes(b"one\rtwo\r\nthree")
    assert ctm._read_text_lines(source) == 3


def test_unknown_utf8_text_is_opt_in(tmp_path):
    (tmp_path / "notes.custom").write_text("first\nsecond", encoding="utf-8")

    default = ctm.scan(tmp_path)
    inclusive = ctm.scan(tmp_path, include_other_text=True)

    assert default["files"] == 0
    assert inclusive["files"] == 1
    assert inclusive["languages"]["Other text"]["loc"] == 2


def test_text_files_uses_same_filters_as_scan(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "blob.py").write_bytes(b"\x00\x01")
    (tmp_path / "node_modules" / "ignored.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.custom").write_text("text\n", encoding="utf-8")

    default = [p.name for p in ctm.text_files(tmp_path)]
    inclusive = [p.name for p in ctm.text_files(tmp_path, include_other_text=True)]
    report = ctm.scan(tmp_path)

    assert default == ["main.py"]
    assert len(default) == report["files"]
    assert sorted(inclusive) == ["main.py", "notes.custom"]
    assert len(inclusive) == ctm.scan(tmp_path, include_other_text=True)["files"]


def test_iter_files_skips_path_level_exclusions_without_reading_content(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "blob.py").write_bytes(b"\x00\x01")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("x\n", encoding="utf-8")

    paths = [p.name for p in ctm.iter_files(tmp_path)]

    assert paths == ["a.py", "blob.py"]
    assert [p.name for p in ctm.text_files(tmp_path)] == ["a.py"]


def test_scan_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        ctm.scan(tmp_path / "missing")


def test_csv_contains_language_rows_and_total(tmp_path):
    (tmp_path / "a.py").write_text("pass\n", encoding="utf-8")
    report = ctm.scan(tmp_path)
    rendered = ctm.render_csv(report)

    assert "Python,1,1," in rendered
    assert "TOTAL,1,1," in rendered


def test_cli_json_output_is_machine_readable(tmp_path):
    (tmp_path / "a.py").write_text("print('ok')\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(tmp_path), "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(result.stdout)

    assert report["tool"] == "ctm"
    assert report["files"] == 1
    assert report["lines_of_code"] == 1


def test_html_dashboard_escapes_untrusted_root_path(tmp_path):
    (tmp_path / "main.py").write_text("pass\n", encoding="utf-8")
    report = ctm.scan(tmp_path)
    # Windows forbids < and > in path names, so inject the untrusted string
    # into the report instead of trying to create such a directory.
    report["root"] = "C:/repo/<script>alert(1)</script>"

    rendered = ctm.render_html(report)

    assert "&lt;script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert "Language breakdown" in rendered


def test_large_file_scanner_filters_threshold_and_sorts(tmp_path):
    (tmp_path / "small.bin").write_bytes(b"123")
    (tmp_path / "large-a.bin").write_bytes(b"a" * 16)
    (tmp_path / "large-b.bin").write_bytes(b"b" * 32)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.bin").write_bytes(b"x" * 128)

    report = ctm_large_files.find_large_files(tmp_path, min_size=8)

    assert [item["path"] for item in report["files"]] == ["large-b.bin", "large-a.bin"]
    assert [item["bytes"] for item in report["files"]] == [32, 16]


def test_legacy_largest_files_delegates_to_unified_scanner(tmp_path):
    import ctm_largest_files

    (tmp_path / "a.bin").write_bytes(b"a" * 8)
    (tmp_path / "b.bin").write_bytes(b"b" * 16)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.bin").write_bytes(b"x" * 128)

    assert ctm_largest_files.largest_files(tmp_path, 1) == [(16, "b.bin")]


def test_legacy_count_empty_uses_unified_filters(tmp_path):
    import ctm_count_empty_files

    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "full.py").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("", encoding="utf-8")

    assert ctm_count_empty_files.count_empty(tmp_path) == 1


def test_legacy_count_by_ext_uses_unified_filters(tmp_path):
    import ctm_count_files_by_ext

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "notes.custom").write_text("text\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("x\n", encoding="utf-8")

    assert ctm_count_files_by_ext.count_by_ext(tmp_path) == {
        ".py": 1,
        ".h": 1,
        ".custom": 1,
    }


def test_legacy_count_includes_uses_unified_filters(tmp_path):
    import ctm_count_includes

    (tmp_path / "a.c").write_text('#include <stdio.h>\n#include "local.h"\n', encoding="utf-8")
    (tmp_path / "notes.py").write_text('#include "ignored.h"\n', encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.h").write_text('#include "ignored.h"\n', encoding="utf-8")

    report = ctm_count_includes.count_includes(tmp_path)

    assert report == {"#include <stdio.h>": 1, '#include "local.h"': 1}


def test_legacy_comments_blank_code_uses_unified_filters(tmp_path):
    import ctm_count_comments_blank_code

    (tmp_path / "a.py").write_text("\n# a comment\nx = 1\n", encoding="utf-8")
    (tmp_path / "notes.custom").write_text("# ignored\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.py").write_text("# ignored\n", encoding="utf-8")

    report = ctm_count_comments_blank_code.count_lines(tmp_path)

    assert report == {"total": 3, "blank": 1, "comments": 1, "code": 1}


def test_legacy_loc_by_file_uses_unified_filters(tmp_path):
    import ctm_count_loc_by_file

    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# title\n", encoding="utf-8")
    (tmp_path / "blob.exe").write_bytes(b"\x00\x01")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("x\n", encoding="utf-8")

    report = ctm_count_loc_by_file.count_loc_by_file(tmp_path)

    assert report == [
        {"file": "a.py", "loc": 2},
        {"file": "notes.md", "loc": 1},
    ]


def test_legacy_count_dirs_uses_unified_filters(tmp_path):
    import ctm_count_total_dirs

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored").mkdir()

    assert ctm_count_total_dirs.count_dirs(tmp_path) == 3


def test_legacy_loc_by_language_uses_unified_filters(tmp_path):
    import ctm_count_loc_by_language

    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    (tmp_path / "b.custom").write_text("text\n", encoding="utf-8")
    (tmp_path / "blob.exe").write_bytes(b"\x00\x01")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "c.py").write_text("x\n", encoding="utf-8")

    stats, files = ctm_count_loc_by_language.count_loc_by_language(tmp_path)

    assert dict(stats) == {"Python": 2, "Other(.custom)": 1}
    assert dict(files) == {"Python": 1, "Other(.custom)": 1}


def test_symlink_files_are_not_followed_when_supported(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this platform")

    report = ctm.scan(tmp_path)

    assert report["files"] == 1
    assert report["skipped_binary_or_symlink"] == 1


def _make_directory_link(link: Path, target: Path) -> None:
    """Create a directory symlink, falling back to a Windows junction."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
    pytest.skip("directory symlink/junction creation is not permitted")


def test_symlink_directories_are_not_followed_and_are_counted(tmp_path):
    target = tmp_path / "target_dir"
    target.mkdir()
    (target / "inside.py").write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "linked_dir"
    _make_directory_link(link, target)

    report = ctm.scan(tmp_path)

    assert report["files"] == 1
    assert report["directories"] == 1
    assert report["skipped_binary_or_symlink"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows junctions only")
def test_windows_junctions_are_not_followed(tmp_path):
    target = tmp_path / "target_dir"
    target.mkdir()
    (target / "inside.py").write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "linked_dir"
    _make_directory_link(link, target)

    assert ctm.is_link(link) is True
    report = ctm.scan(tmp_path)
    assert report["files"] == 1
    assert report["skipped_binary_or_symlink"] == 1
    assert [p.name for p in ctm.text_files(tmp_path)] == ["inside.py"]
