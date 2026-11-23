#!/usr/bin/env python3
"""Inventory encoding and newline styles of candidate files (bytes, not text)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import ctm

CHUNK = 256 * 1024
BOMS = {
    b"\xef\xbb\xbf": "utf-8-sig",
    b"\xff\xfe": "utf-16-le",
    b"\xfe\xff": "utf-16-be",
}


def _inspect(path: Path) -> dict[str, Any]:
    bom = None
    has_nul = False
    terms: set[str] = set()
    invalid_utf8 = False
    decoder = __import__("codecs").getincrementaldecoder("utf-8")("strict")
    first = True
    carry = b""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            if first:
                for marker, name in BOMS.items():
                    if chunk.startswith(marker):
                        bom = name
                        break
                first = False
            data = carry + chunk
            if b"\x00" in data:
                has_nul = True
            if not invalid_utf8:
                try:
                    decoder.decode(data, final=False)
                except UnicodeDecodeError:
                    invalid_utf8 = True
            if data.endswith(b"\r"):
                carry = b"\r"
                data = data[:-1]
            else:
                carry = b""
            if b"\r\n" in data:
                terms.add("crlf")
            data_no_crlf = data.replace(b"\r\n", b"")
            if b"\n" in data_no_crlf:
                terms.add("lf")
            if b"\r" in data_no_crlf:
                terms.add("cr")
    if carry == b"\r":
        terms.add("cr")
    if not invalid_utf8:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            invalid_utf8 = True
    return {
        "bom": bom,
        "invalid_utf8": invalid_utf8,
        "has_nul": has_nul,
        "newlines": sorted(terms),
        "mixed_newlines": len(terms) > 1,
    }


def encoding_report(root: str | Path) -> dict[str, Any]:
    """Summarize encoding signals across all path-included files."""
    base = Path(root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    summary = {
        "files": 0,
        "with_bom": 0,
        "invalid_utf8": 0,
        "with_nul": 0,
        "crlf": 0,
        "mixed_newlines": 0,
    }
    for path in ctm.iter_files(base, include_other_text=True):
        try:
            info = _inspect(path)
        except OSError:
            continue
        relative = path.relative_to(base).as_posix()
        entry = {"path": relative, **info}
        files.append(entry)
        summary["files"] += 1
        summary["with_bom"] += int(info["bom"] is not None)
        summary["invalid_utf8"] += int(info["invalid_utf8"])
        summary["with_nul"] += int(info["has_nul"])
        summary["crlf"] += int("crlf" in info["newlines"])
        summary["mixed_newlines"] += int(info["mixed_newlines"])
    files.sort(key=lambda item: item["path"])
    return {
        "tool": "ctm_encoding_report",
        "root": str(base),
        "summary": summary,
        "files": files,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bom", "invalid_utf8", "has_nul", "newlines"])
        for item in report["files"]:
            writer.writerow(
                [
                    item["path"],
                    item["bom"] or "",
                    int(item["invalid_utf8"]),
                    int(item["has_nul"]),
                    ",".join(item["newlines"]),
                ]
            )
        return stream.getvalue()
    lines = [f"CTM encoding report: {report['root']}"]
    for key, value in report["summary"].items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report file encodings and newline styles"
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = encoding_report(args.root)
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
