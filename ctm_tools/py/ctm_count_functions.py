#!/usr/bin/env python3
"""Count function definitions with a conservative, text-based heuristic.

This is not an AST parser: it counts recognizable definition lines per
language and deliberately ignores control-flow keywords. Use it for rough
telemetry only; no code is executed and results are deterministic.
"""

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

PYTHON_DEF = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
JS_DEF = re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(")
JS_ARROW = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(")
C_DEF = re.compile(
    r"^\s*(?:static\s+|inline\s+|extern\s+)*"
    r"(?:[A-Za-z_][\w:<>,\s\*&]*\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
CONTROL = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "else",
    "do",
}


def _patterns(language: str) -> tuple[re.Pattern[str], ...]:
    if language == "Python":
        return (PYTHON_DEF,)
    if language == "JavaScript" or language == "TypeScript":
        return (JS_DEF, JS_ARROW)
    if language in {"C", "C headers", "C++", "C++ headers", "CUDA", "CUDA headers"}:
        return (C_DEF,)
    return ()


def count_functions(
    root: str | Path, include_other_text: bool = False
) -> dict[str, Any]:
    """Count heuristic function definitions using ctm.py's file filters."""
    by_language: dict[str, int] = defaultdict(int)
    by_file: list[dict[str, Any]] = []
    files_scanned = 0
    base = Path(root).expanduser().resolve()

    for path in ctm.text_files(base, include_other_text=include_other_text):
        files_scanned += 1
        language = ctm.LANGUAGES.get(path.suffix.lower(), "Other text")
        patterns = _patterns(language)
        count = 0
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                for pattern in patterns:
                    match = pattern.search(line)
                    if match and match.group(1) not in CONTROL:
                        count += 1
                        break
        if count:
            by_language[language] += count
            by_file.append(
                {"path": path.relative_to(base).as_posix(), "functions": count}
            )
    by_file.sort(key=lambda item: (-item["functions"], item["path"]))
    return {
        "tool": "ctm_count_functions",
        "root": str(base),
        "files_scanned": files_scanned,
        "functions": sum(by_language.values()),
        "by_language": dict(sorted(by_language.items())),
        "by_file": by_file,
        "heuristic": True,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["file", "functions"])
        for item in report["by_file"]:
            writer.writerow([item["path"], item["functions"]])
        return stream.getvalue()
    lines = [
        f"CTM function count: {report['root']}",
        f"Files scanned: {report['files_scanned']}",
        f"Functions (heuristic): {report['functions']}",
    ]
    lines.extend(
        f"  {language}: {count}" for language, count in report["by_language"].items()
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count function definitions heuristically"
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = count_functions(args.root, args.include_other_text)
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
