#!/usr/bin/env python3
"""Count C/C++ ``#include`` lines via the unified CTM text-file filters."""

import sys
import json
from pathlib import Path
from collections import defaultdict

import ctm

SOURCE_SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".cc"}


def count_includes(root="."):
    includes = defaultdict(int)
    for path in ctm.text_files(root):
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#include"):
                includes[s] += 1
    return includes


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

    includes = count_includes(root)
    result = {
        "tool": "ctm_count_includes",
        "root": str(Path(root).resolve()),
        "includes": dict(sorted(includes.items(), key=lambda x: x[1], reverse=True)),
    }

    if fmt == "json":
        out = json.dumps(result, indent=2)
    elif fmt == "csv":
        out = "include,count\n"
        for k, v in sorted(includes.items(), key=lambda x: x[1], reverse=True):
            out += f'"{k}",{v}\n'
    else:
        out = "\n".join(
            f"{v:6d} {k}"
            for k, v in sorted(includes.items(), key=lambda x: x[1], reverse=True)[:50]
        )

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
