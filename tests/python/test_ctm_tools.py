"""Regression tests for the standalone CTM scanner."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "ctm_tools" / "py" / "ctm.py"
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


def test_line_reader_handles_universal_newlines(tmp_path):
    source = tmp_path / "mixed.txt"
    source.write_bytes(b"one\\rtwo\\r\\nthree")
    assert ctm._read_text_lines(source) == 3


def test_unknown_utf8_text_is_opt_in(tmp_path):
    (tmp_path / "notes.custom").write_text("first\nsecond", encoding="utf-8")

    default = ctm.scan(tmp_path)
    inclusive = ctm.scan(tmp_path, include_other_text=True)

    assert default["files"] == 0
    assert inclusive["files"] == 1
    assert inclusive["languages"]["Other text"]["loc"] == 2


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
        check=True, capture_output=True, text=True, timeout=10,
    )
    report = json.loads(result.stdout)

    assert report["tool"] == "ctm"
    assert report["files"] == 1
    assert report["lines_of_code"] == 1


def test_html_dashboard_escapes_untrusted_root_path(tmp_path):
    report_root = tmp_path / 'repo<script>alert(1)</script>'
    report_root.mkdir(parents=True)
    (report_root / 'main.py').write_text('pass\\n', encoding='utf-8')

    rendered = ctm.render_html(ctm.scan(report_root))

    assert '&lt;script&gt;' in rendered
    assert '<script>alert(1)</script>' not in rendered
    assert 'Language breakdown' in rendered


def test_large_file_scanner_filters_threshold_and_sorts(tmp_path):
    (tmp_path / "small.bin").write_bytes(b"123")
    (tmp_path / "large-a.bin").write_bytes(b"a" * 16)
    (tmp_path / "large-b.bin").write_bytes(b"b" * 32)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.bin").write_bytes(b"x" * 128)

    report = ctm_large_files.find_large_files(tmp_path, min_size=8)

    assert [item["path"] for item in report["files"]] == ["large-b.bin", "large-a.bin"]
    assert [item["bytes"] for item in report["files"]] == [32, 16]


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
