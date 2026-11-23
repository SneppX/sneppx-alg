#!/usr/bin/env python3
"""Compare physical text-line counts between two directory trees."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import ctm

FIELDS = ("files", "loc", "bytes")


def diff_loc(
    old_root: str | Path,
    new_root: str | Path,
    include_other_text: bool = False,
) -> dict[str, Any]:
    """Return per-language and total deltas using ctm.scan on both roots."""
    old = ctm.scan(old_root, include_other_text=include_other_text)
    new = ctm.scan(new_root, include_other_text=include_other_text)
    by_language: dict[str, dict[str, int]] = {}
    for language in sorted(set(old["languages"]) | set(new["languages"])):
        left = old["languages"].get(language, {})
        right = new["languages"].get(language, {})
        by_language[language] = {
            field: right.get(field, 0) - left.get(field, 0) for field in FIELDS
        }
    totals = {
        "files": new["files"] - old["files"],
        "loc": new["lines_of_code"] - old["lines_of_code"],
        "bytes": new["bytes"] - old["bytes"],
    }
    return {
        "tool": "ctm_diff_loc",
        "old_root": old["root"],
        "new_root": new["root"],
        "by_language": by_language,
        "totals": totals,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["language", "files_delta", "loc_delta", "bytes_delta"])
        for language, values in report["by_language"].items():
            writer.writerow([language, values["files"], values["loc"], values["bytes"]])
        totals = report["totals"]
        writer.writerow(["TOTAL", totals["files"], totals["loc"], totals["bytes"]])
        return stream.getvalue()
    lines = [
        f"CTM LOC diff: {report['old_root']} -> {report['new_root']}",
    ]
    for language, values in report["by_language"].items():
        lines.append(f"  {language}: {values['loc']:+d} lines, {values['files']:+d} files")
    totals = report["totals"]
    lines.append(
        f"TOTAL: {totals['loc']:+d} lines, {totals['files']:+d} files, {totals['bytes']:+d} bytes"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff LOC between two trees")
    parser.add_argument("old_root")
    parser.add_argument("new_root")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = diff_loc(args.old_root, args.new_root, args.include_other_text)
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
