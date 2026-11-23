#!/usr/bin/env python3
"""Explain which files the default CTM filters exclude, and why."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ctm


def _reason(path: Path, base: Path) -> str:
    relative = path.relative_to(base)
    if any(part in ctm.IGNORED_DIRS for part in relative.parts[:-1]):
        return "ignored_dir"
    if ctm.is_link(path):
        return "symlink"
    suffix = path.suffix.lower()
    if suffix in ctm.IGNORED_SUFFIXES:
        return "binary_suffix"
    if suffix not in ctm.LANGUAGES:
        return "unsupported_suffix"
    try:
        if ctm._read_text_lines(path) is None:
            return "binary_content"
    except OSError:
        return "unreadable"
    return "included"


def ignore_audit(root: str | Path, limit: int = 50) -> dict[str, Any]:
    """Classify every file under root by CTM inclusion reason."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"scan root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"scan root is not a directory: {base}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for current, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if ctm.is_link(child):
                counts["symlink"] += 1
                grouped["symlink"].append({"path": child.relative_to(base).as_posix(), "bytes": 0})
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = current_path / filename
            reason = _reason(path, base)
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            counts[reason] += 1
            totals[reason] += size
            grouped[reason].append({"path": path.relative_to(base).as_posix(), "bytes": size})

    for items in grouped.values():
        items.sort(key=lambda item: (-item["bytes"], item["path"]))
    examples: dict[str, list[dict[str, Any]]] = {}
    for reason, items in grouped.items():
        examples[reason] = items[:limit] if limit > 0 else items

    excluded = sum(count for reason, count in counts.items() if reason != "included")
    return {
        "tool": "ctm_ignore_audit",
        "root": str(base),
        "files_total": sum(counts.values()),
        "files_included": counts.get("included", 0),
        "files_excluded": excluded,
        "count_by_reason": dict(sorted(counts.items())),
        "bytes_by_reason": dict(sorted(totals.items())),
        "examples": dict(sorted(examples.items())),
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["reason", "path", "bytes"])
        for reason, items in report["examples"].items():
            for item in items:
                writer.writerow([reason, item["path"], item["bytes"]])
        return stream.getvalue()
    lines = [
        f"CTM ignore audit: {report['root']}",
        f"Total: {report['files_total']}  Included: {report['files_included']}  "
        f"Excluded: {report['files_excluded']}",
    ]
    for reason, count in report["count_by_reason"].items():
        lines.append(f"  {reason}: {count} files, {report['bytes_by_reason'][reason]:,} bytes")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain CTM file exclusions")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=50, help="0 means no limit")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = ignore_audit(args.root, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
