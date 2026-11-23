#!/usr/bin/env python3
"""Scan included text files for TODO/FIXME/HACK/XXX markers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ctm

DEFAULT_MARKERS = ("TODO", "FIXME", "HACK", "XXX")


def _pattern(markers: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(marker) for marker in markers)
    return re.compile(rf"\b({alternatives})\b[^\r\n]*")


def todo_scan(
    root: str | Path,
    markers: list[str] | None = None,
    include_other_text: bool = False,
) -> dict[str, Any]:
    """Return marker counts and occurrences with file/line references."""
    markers = list(markers or DEFAULT_MARKERS)
    pattern = _pattern(markers)
    base = Path(root).expanduser().resolve()
    counts: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    occurrences: list[dict[str, Any]] = []

    for path in ctm.text_files(base, include_other_text=include_other_text):
        language = ctm.LANGUAGES.get(path.suffix.lower(), "Other text")
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for number, line in enumerate(stream, start=1):
                for match in pattern.finditer(line):
                    kind = match.group(1)
                    counts[kind] += 1
                    by_language[language] += 1
                    occurrences.append(
                        {
                            "path": path.relative_to(base).as_posix(),
                            "line": number,
                            "marker": kind,
                            "text": match.group(0).strip()[:160],
                        }
                    )

    occurrences.sort(key=lambda item: (item["path"], item["line"]))
    return {
        "tool": "ctm_todo_scan",
        "root": str(base),
        "markers": markers,
        "total": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "by_language": dict(sorted(by_language.items())),
        "occurrences": occurrences,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "line", "marker", "text"])
        for item in report["occurrences"]:
            writer.writerow([item["path"], item["line"], item["marker"], item["text"]])
        return stream.getvalue()
    lines = [f"CTM marker scan: {report['root']}", f"Total: {report['total']}"]
    lines.extend(f"  {kind}: {count}" for kind, count in report["counts"].items())
    lines.extend(
        f"  {item['path']}:{item['line']} {item['text']}" for item in report["occurrences"]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for code markers")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--markers", help="comma-separated marker names")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    markers = [m.strip() for m in args.markers.split(",")] if args.markers else None
    if markers and not all(markers):
        parser.error("--markers must contain non-empty names")
    try:
        report = todo_scan(args.root, markers, args.include_other_text)
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
