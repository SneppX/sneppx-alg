#!/usr/bin/env python3
"""List the largest included files with human-readable sizes.

This is a presentation wrapper over :mod:`ctm_find_large_files`, so file
selection, sorting, and symlink safety remain identical to the tested scanner.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import ctm_find_large_files


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - unreachable


def list_large_files(
    root: str | Path = ".", min_size: int = 0, limit: int = 20
) -> dict[str, Any]:
    """Return the largest files, smallest threshold 0, sorted by size."""
    report = ctm_find_large_files.find_large_files(root, min_size=min_size, limit=limit)
    for item in report["files"]:
        item["size"] = human_size(item["bytes"])
    report["tool"] = "ctm_list_large_files"
    return report


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes", "size"])
        for item in report["files"]:
            writer.writerow([item["path"], item["bytes"], item["size"]])
        return stream.getvalue()
    lines = [
        f"CTM largest files: {report['root']}",
        f"Minimum size: {report['min_size']} bytes",
    ]
    lines.extend(f"  {item['size']:>10}  {item['path']}" for item in report["files"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the largest included files")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.min_size < 0 or args.limit < 0:
        parser.error("--min-size and --limit must be non-negative")
    try:
        report = list_large_files(args.root, args.min_size, args.limit)
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
