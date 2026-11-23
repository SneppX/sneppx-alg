#!/usr/bin/env python3
"""Compatibility wrapper for the unified CTM physical file counter."""

import argparse
import json
from pathlib import Path

import ctm


def count_files(root: str | Path = ".") -> int:
    """Count included text files using the unified scanner filters."""
    return len(list(ctm.text_files(root)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count repository text files")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    count = count_files(args.root)
    result = {
        "tool": "ctm_count_total_files",
        "root": str(Path(args.root).resolve()),
        "count": count,
    }
    if args.format == "json":
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
    elif args.format == "csv":
        rendered = f"tool,root,count\nctm_count_total_files,{result['root']},{count}\n"
    else:
        rendered = str(count)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
