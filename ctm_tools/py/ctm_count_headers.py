#!/usr/bin/env python3
"""Count C/C++ header and source files with ctm.py's inclusion filters."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ctm

HEADER_SUFFIXES = {".h", ".hpp", ".hxx", ".cuh"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}


def count_headers(root: str | Path, include_other_text: bool = False) -> dict[str, Any]:
    """Report header/source file counts and per-language totals."""
    base = Path(root).expanduser().resolve()
    by_language: dict[str, int] = defaultdict(int)
    headers: list[str] = []
    header_count = source_count = files_scanned = 0

    for path in ctm.text_files(base, include_other_text=include_other_text):
        files_scanned += 1
        suffix = path.suffix.lower()
        language = ctm.LANGUAGES.get(suffix, "Other text")
        by_language[language] += 1
        relative = path.relative_to(base).as_posix()
        if suffix in HEADER_SUFFIXES:
            header_count += 1
            headers.append(relative)
        elif suffix in SOURCE_SUFFIXES:
            source_count += 1

    headers.sort()
    return {
        "tool": "ctm_count_headers",
        "root": str(base),
        "files_scanned": files_scanned,
        "header_files": header_count,
        "source_files": source_count,
        "headers_per_source": round(header_count / source_count, 3) if source_count else None,
        "by_language": dict(sorted(by_language.items())),
        "headers": headers,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["header"])
        for header in report["headers"]:
            writer.writerow([header])
        return stream.getvalue()
    ratio = report["headers_per_source"]
    lines = [
        f"CTM header count: {report['root']}",
        f"Files scanned: {report['files_scanned']}",
        f"Headers: {report['header_files']}",
        f"Sources: {report['source_files']}",
        f"Headers per source: {ratio if ratio is not None else 'n/a'}",
    ]
    lines.extend(f"  {language}: {count}" for language, count in report["by_language"].items())
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count C/C++ headers and sources")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = count_headers(args.root, args.include_other_text)
    except OSError as exc:
        parser.error(str(exc))
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
