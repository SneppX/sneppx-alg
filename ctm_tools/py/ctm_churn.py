#!/usr/bin/env python3
"""Rank files by recent change frequency using `git log --numstat`."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return result.stdout


def churn(root: str | Path, limit: int = 20) -> dict[str, Any]:
    """Aggregate additions/deletions per path from the whole git history."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    base = Path(root).expanduser().resolve()
    _git(base, "rev-parse", "--is-inside-work-tree")
    output = _git(base, "log", "--numstat", "--format=%x00COMMIT", "--no-renames")
    commits = 0
    totals: dict[str, dict[str, int]] = {}
    additions = deletions = 0
    for line in output.splitlines():
        if line.startswith("\0COMMIT"):
            commits += 1
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        added = int(fields[0]) if fields[0].isdigit() else 0
        removed = int(fields[1]) if fields[1].isdigit() else 0
        path = fields[2]
        entry = totals.setdefault(path, {"changes": 0, "added": 0, "deleted": 0})
        entry["changes"] += 1
        entry["added"] += added
        entry["deleted"] += removed
        additions += added
        deletions += removed

    files = [{"path": path, **values} for path, values in totals.items()]
    files.sort(key=lambda item: (-(item["added"] + item["deleted"]), item["path"]))
    if limit > 0:
        files = files[:limit]
    return {
        "tool": "ctm_churn",
        "root": str(base),
        "commits": commits,
        "additions": additions,
        "deletions": deletions,
        "files": files,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "changes", "added", "deleted"])
        for item in report["files"]:
            writer.writerow(
                [item["path"], item["changes"], item["added"], item["deleted"]]
            )
        return stream.getvalue()
    lines = [
        f"CTM churn: {report['root']}",
        f"Commits: {report['commits']}  +{report['additions']} / -{report['deletions']}",
    ]
    lines.extend(
        f"  {item['added'] + item['deleted']:>7}  {item['path']}"
        for item in report["files"]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank files by git churn")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--limit", type=int, default=20, help="0 means no limit")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = churn(args.root, args.limit)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(str(exc))
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
