#!/usr/bin/env python3
"""Check relative Markdown links against files on disk."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import ctm

LINK = re.compile(r"\]\(([^)]+)\)")
SKIP_SCHEMES = ("http://", "https://", "mailto:", "data:", "ftp://", "#")


def _target(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    else:
        value = value.split()[0]
    if not value or value.startswith(SKIP_SCHEMES):
        return None
    return unquote(value.split("#", 1)[0].split("?", 1)[0])


def check_links(root: str | Path) -> dict[str, Any]:
    """Return broken relative links found in included Markdown files."""
    base = Path(root).expanduser().resolve()
    checked = 0
    broken = 0
    issues: list[dict[str, Any]] = []
    for path in ctm.text_files(base):
        if path.suffix.lower() != ".md":
            continue
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for number, line in enumerate(stream, start=1):
                for match in LINK.finditer(line):
                    target = _target(match.group(1))
                    if target is None:
                        continue
                    checked += 1
                    resolved = (path.parent / target).resolve()
                    if not resolved.exists():
                        broken += 1
                        issues.append(
                            {
                                "path": path.relative_to(base).as_posix(),
                                "line": number,
                                "target": target,
                            }
                        )
    return {
        "tool": "ctm_doc_links",
        "root": str(base),
        "links_checked": checked,
        "broken": broken,
        "issues": issues,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "line", "target"])
        for issue in report["issues"]:
            writer.writerow([issue["path"], issue["line"], issue["target"]])
        return stream.getvalue()
    lines = [
        f"CTM doc links: {report['root']}",
        f"Checked: {report['links_checked']}  Broken: {report['broken']}",
    ]
    lines.extend(
        f"  {issue['path']}:{issue['line']} -> {issue['target']}"
        for issue in report["issues"]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check relative Markdown links")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = check_links(args.root)
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
