#!/usr/bin/env python3
"""Compatibility wrapper for the unified CTM directory counter."""

import sys
import json
from pathlib import Path

from ctm import scan


def count_dirs(root="."):
    """Count the scan root plus every non-ignored subdirectory."""
    return scan(root)["directories"] + 1


def main():
    args = sys.argv[1:]
    fmt = "text"
    output = None
    root = "."
    i = 0
    while i < len(args):
        if args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
            continue
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
            continue
        if not args[i].startswith("-"):
            root = args[i]
        i += 1

    count = count_dirs(root)
    result = {
        "tool": "ctm_count_total_dirs",
        "root": str(Path(root).resolve()),
        "count": count,
    }

    if fmt == "json":
        out = json.dumps(result, indent=2)
    elif fmt == "csv":
        out = f"tool,root,count\nctm_count_total_dirs,{result['root']},{count}\n"
    else:
        out = str(count)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
