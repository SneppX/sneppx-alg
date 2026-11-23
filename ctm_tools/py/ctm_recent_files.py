#!/usr/bin/env python3
"""List the most recently modified included files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ctm


def recent_files(
    root: str | Path, limit: int = 20, include_other_text: bool = False
) -> dict[str, Any]:
    """Return included files ordered by modification time (newest first)."""
    base = Path(root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    unreadable = 0
    for path in ctm.text_files(base, include_other_text=include_other_text):
        try:
            stat = path.stat()
        except OSError:
            unreadable += 1
            continue
        files.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "_mtime": stat.st_mtime,
            }
        )
    files.sort(key=lambda item: (-item["_mtime"], item["path"]))
    for item in files:
        del item["_mtime"]
    if limit > 0:
        files = files[:limit]
    return {
        "tool": "ctm_recent_files",
        "root": str(base),
        "limit": limit,
        "files": files,
        "unreadable": unreadable,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes", "modified"])
        for item in report["files"]:
            writer.writerow([item["path"], item["bytes"], item["modified"]])
        return stream.getvalue()
    lines = [f"CTM recent files: {report['root']}"]
    lines.extend(f"  {item['modified']}  {item['path']}" for item in report["files"])
    if not report["files"]:
        lines.append("  No included files found.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List recently modified files")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=20, help="0 means no limit")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    try:
        report = recent_files(args.root, args.limit, args.include_other_text)
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
