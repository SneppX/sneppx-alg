#!/usr/bin/env python3
"""CTM: deterministic, dependency-free repository metrics.

This scanner reports filesystem metrics, not semantic source-code analysis.
It deliberately skips binary files, generated directories, and symlinks.
"""
from __future__ import annotations

import argparse
import csv
import json
import html
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".venv", "venv", "node_modules", "build",
    "build-deps", "build-deps2", "dist", "site", ".next", ".cache",
    ".idea", ".vscode",
}
IGNORED_SUFFIXES = {
    ".pyc", ".pyo", ".o", ".obj", ".a", ".lib", ".so", ".dll", ".dylib",
    ".exe", ".pyd", ".class", ".jar", ".zip", ".gz", ".7z", ".png",
    ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".mp4", ".mp3",
    ".wav", ".woff", ".woff2", ".ttf", ".otf",
}
LANGUAGES = {
    ".c": "C", ".h": "C headers", ".cc": "C++", ".cpp": "C++",
    ".cxx": "C++", ".hpp": "C++ headers", ".hxx": "C++ headers",
    ".py": "Python", ".pyi": "Python", ".js": "JavaScript",
    ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".swift": "Swift",
    ".rb": "Ruby", ".php": "PHP", ".sh": "Shell", ".bash": "Shell",
    ".ps1": "PowerShell", ".bat": "Batch", ".cmd": "Batch",
    ".cu": "CUDA", ".cuh": "CUDA headers", ".asm": "Assembly",
    ".s": "Assembly", ".sql": "SQL", ".html": "HTML", ".css": "CSS",
    ".scss": "SCSS", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML", ".xml": "XML", ".md": "Markdown", ".rst": "reStructuredText",
    ".cmake": "CMake", ".make": "Makefile",
}


def _read_text_lines(path: Path) -> int | None:
    """Stream-count UTF-8 physical lines; return None for binary/undecodable files."""
    try:
        line_count = 0
        with path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
            for line in stream:
                if "\x00" in line:
                    return None
                line_count += 1
        return line_count
    except UnicodeError:
        return None


def _git_counts(root: Path) -> dict[str, int] | None:
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        ).stdout
        return {
            "tracked_files": len([item for item in tracked.split(b"\0") if item]),
            "untracked_files": len([item for item in untracked.split(b"\0") if item]),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def scan(root: str | Path, include_other_text: bool = False) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"scan root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"scan root is not a directory: {base}")

    languages: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "loc": 0, "bytes": 0})
    files = dirs = total_bytes = loc = skipped_binary = unreadable = 0

    for current, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if dirname not in IGNORED_DIRS and not child.is_symlink():
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        dirs += len(kept_dirs)

        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink() or path.suffix.lower() in IGNORED_SUFFIXES:
                skipped_binary += 1
                continue
            suffix = path.suffix.lower()
            language = LANGUAGES.get(suffix)
            if language is None and not include_other_text:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                unreadable += 1
                continue
            try:
                line_count = _read_text_lines(path)
            except OSError:
                unreadable += 1
                continue
            if line_count is None:
                skipped_binary += 1
                continue
            label = language or "Other text"
            files += 1
            total_bytes += size
            loc += line_count
            languages[label]["files"] += 1
            languages[label]["loc"] += line_count
            languages[label]["bytes"] += size

    result: dict[str, Any] = {
        "tool": "ctm",
        "schema_version": 1,
        "root": str(base),
        "files": files,
        "directories": dirs,
        "lines_of_code": loc,
        "bytes": total_bytes,
        "skipped_binary_or_symlink": skipped_binary,
        "unreadable_files": unreadable,
        "languages": dict(sorted(languages.items())),
    }
    git = _git_counts(base)
    if git is not None:
        result["git"] = git
    return result


def render_csv(report: dict[str, Any]) -> str:
    from io import StringIO
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["language", "files", "loc", "bytes"])
    for language, metrics in report["languages"].items():
        writer.writerow([language, metrics["files"], metrics["loc"], metrics["bytes"]])
    writer.writerow(["TOTAL", report["files"], report["lines_of_code"], report["bytes"]])
    return stream.getvalue()


def render_html(report: dict[str, Any]) -> str:
    """Render a self-contained, escaped HTML dashboard without external assets."""
    root = html.escape(report["root"], quote=True)
    rows = []
    for language, metrics in report["languages"].items():
        rows.append(
            "<tr><th scope=\"row\">" + html.escape(language) + "</th>"
            f"<td>{metrics['files']:,}</td><td>{metrics['loc']:,}</td>"
            f"<td>{metrics['bytes']:,}</td></tr>"
        )
    table_rows = "\n".join(rows) or '<tr><td colspan="4">No supported text files found.</td></tr>'
    git = report.get("git")
    git_summary = (
        f"<p>Tracked files: {git['tracked_files']:,} · Untracked files: {git['untracked_files']:,}</p>"
        if git else "<p>Git metadata unavailable for this directory.</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>CTM repository report</title>
<style>
:root {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; color-scheme: light dark; }}
body {{ margin: 0; background: Canvas; color: CanvasText; line-height: 1.5; }}
main {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
header {{ margin-bottom: 1.5rem; }}
h1 {{ margin-bottom: .25rem; letter-spacing: -.03em; }}
.muted {{ color: color-mix(in srgb, CanvasText 65%, transparent); overflow-wrap: anywhere; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: .8rem; margin: 1.5rem 0; }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius: 14px; padding: 1rem; }}
.value {{ display: block; font-size: 1.65rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.label {{ font-size: .85rem; color: color-mix(in srgb, CanvasText 65%, transparent); }}
section {{ margin-top: 2rem; }}
.table-wrap {{ overflow-x: auto; border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius: 12px; }}
table {{ width: 100%; border-collapse: collapse; text-align: left; }}
th,td {{ padding: .7rem .85rem; border-bottom: 1px solid color-mix(in srgb, CanvasText 14%, transparent); }}
thead {{ background: color-mix(in srgb, CanvasText 6%, Canvas); }}
td {{ text-align: right; font-variant-numeric: tabular-nums; }}
td:first-child, th:first-child {{ text-align: left; }}
footer {{ margin-top: 2rem; font-size: .85rem; }}
</style>
</head>
<body><main>
<header><p class="muted">CODEBASE TELEMETRY &amp; METRICS</p><h1>Repository report</h1><p class="muted">{root}</p></header>
<div class="cards">
<div class="card"><span class="value">{report['files']:,}</span><span class="label">Included text files</span></div>
<div class="card"><span class="value">{report['directories']:,}</span><span class="label">Descendant directories</span></div>
<div class="card"><span class="value">{report['lines_of_code']:,}</span><span class="label">Physical text lines</span></div>
<div class="card"><span class="value">{report['bytes']:,}</span><span class="label">Included bytes</span></div>
</div>
<section><h2>Language breakdown</h2><div class="table-wrap"><table>
<thead><tr><th scope="col">Language</th><th scope="col">Files</th><th scope="col">Lines</th><th scope="col">Bytes</th></tr></thead>
<tbody>{table_rows}</tbody></table></div></section>
<section><h2>Scan diagnostics</h2><p>Skipped binary or symlink entries: {report['skipped_binary_or_symlink']:,}</p><p>Unreadable files: {report['unreadable_files']:,}</p>{git_summary}</section>
<footer class="muted">Generated by CTM schema v{report['schema_version']}. Physical line counts include blank and comment lines; they are not a code-quality or security score.</footer>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CTM repository metrics scanner")
    parser.add_argument("root", nargs="?", default=".", help="directory to scan (default: current directory)")
    parser.add_argument("--format", choices=("text", "json", "csv", "html"), default="text")
    parser.add_argument("--output", help="write report to a file instead of stdout")
    parser.add_argument("--include-other-text", action="store_true", help="also count UTF-8 text files with unknown extensions")
    args = parser.parse_args(argv)

    try:
        report = scan(args.root, include_other_text=args.include_other_text)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.format == "json":
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    elif args.format == "html":
        rendered = render_html(report)
    elif args.format == "csv":
        rendered = render_csv(report)
    else:
        rendered = (
            f"CTM report: {report['root']}\n"
            f"Files: {report['files']}\nDirectories: {report['directories']}\n"
            f"Lines of code: {report['lines_of_code']}\n"
            f"Bytes: {report['bytes']}\n"
            f"Skipped binary/symlink: {report['skipped_binary_or_symlink']}\n"
            f"Unreadable: {report['unreadable_files']}\n"
        )
        for language, metrics in report["languages"].items():
            rendered += f"  {language}: {metrics['files']} files, {metrics['loc']} LOC\n"
        if "git" in report:
            rendered += (
                f"Git tracked: {report['git']['tracked_files']}\n"
                f"Git untracked: {report['git']['untracked_files']}\n"
            )

    if args.output:
        with Path(args.output).expanduser().open("w", encoding="utf-8", newline="") as output_file:
            output_file.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
