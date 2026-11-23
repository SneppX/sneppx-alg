#!/usr/bin/env python3
"""Find byte-identical files using streaming SHA-256 digests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import ctm

CHUNK = 1024 * 1024


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_files(root: str | Path, min_size: int = 0) -> dict[str, Any]:
    """Group files with identical content; report only groups of two or more."""
    if min_size < 0:
        raise ValueError("min_size must be non-negative")
    base = Path(root).expanduser().resolve()
    by_digest: dict[str, list[dict[str, Any]]] = {}
    scanned = unreadable = 0

    for path in ctm.iter_files(base, include_other_text=True):
        try:
            stat = path.stat()
            if stat.st_size < min_size:
                continue
            digest = _digest(path)
        except OSError:
            unreadable += 1
            continue
        scanned += 1
        by_digest.setdefault(digest, []).append(
            {"path": path.relative_to(base).as_posix(), "bytes": stat.st_size}
        )

    groups = []
    for digest, items in by_digest.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda item: item["path"])
        groups.append(
            {
                "sha256": digest,
                "bytes_each": items[0]["bytes"],
                "files": [item["path"] for item in items],
                "wasted_bytes": items[0]["bytes"] * (len(items) - 1),
            }
        )
    groups.sort(key=lambda item: (-item["wasted_bytes"], item["files"][0]))
    return {
        "tool": "ctm_duplicate_files",
        "root": str(base),
        "min_size": min_size,
        "files_scanned": scanned,
        "unreadable": unreadable,
        "groups": groups,
        "wasted_bytes": sum(item["wasted_bytes"] for item in groups),
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sha256", "path", "bytes"])
        for group in report["groups"]:
            for path in group["files"]:
                writer.writerow([group["sha256"], path, group["bytes_each"]])
        return stream.getvalue()
    lines = [
        f"CTM duplicates: {report['root']}",
        f"Groups: {len(report['groups'])}  Wasted bytes: {report['wasted_bytes']:,}",
    ]
    for group in report["groups"]:
        lines.append(f"  {group['bytes_each']:,} bytes x {len(group['files'])}:")
        lines.extend(f"    {path}" for path in group["files"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find byte-identical files")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = duplicate_files(args.root, args.min_size)
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
