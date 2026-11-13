#!/usr/bin/env python3
"""CTM: deterministic, dependency-free repository metrics.

This scanner reports filesystem metrics, not semantic source-code analysis.
It deliberately skips binary files, generated directories, and symlinks.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    """Return line count for UTF-8 text, or None for binary/undecodable files."""
    try:
        data = path.read_bytes()
        if b"\x00" in data:
            return None
        if not data:
            return 0
        text = data.decode("utf-8")
        return len(text.splitlines()) + (1 if text.endswith(("\n", "\r")) and text.splitlines() else 0)
    except (OSError, UnicodeError):
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
            line_count = _read_text_lines(path)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CTM repository metrics scanner")
    parser.add_argument("root", nargs="?", default=".", help="directory to scan (default: current directory)")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output", help="write report to a file instead of stdout")
    parser.add_argument("--include-other-text", action="store_true", help="also count UTF-8 text files with unknown extensions")
    args = parser.parse_args(argv)

    try:
        report = scan(args.root, include_other_text=args.include_other_text)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.format == "json":
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
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
        with Path(args.output).expanduser().open("w", encoding="utf-8", newline="") as output_file:\n            output_file.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
