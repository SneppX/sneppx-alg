#!/usr/bin/env python3
"""Count physical lines per file using the unified CTM text-file filters."""

import sys
import json
from pathlib import Path

import ctm


def count_loc_by_file(root="."):
    base = Path(root).resolve()
    results = []
    for path in ctm.text_files(root, include_other_text=True):
        try:
            lines = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            continue
        results.append({"file": str(path.relative_to(base)), "loc": lines})
    results.sort(key=lambda x: x["loc"], reverse=True)
    return results


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

    res = count_loc_by_file(root)
    result = {
        "tool": "ctm_count_loc_by_file",
        "root": str(Path(root).resolve()),
        "files": res,
    }

    if fmt == "json":
        out = json.dumps(result, indent=2)
    elif fmt == "csv":
        out = "file,loc\n"
        for r in res:
            out += f'"{r["file"]}",{r["loc"]}\n'
    else:
        out = "\n".join(f'{r["loc"]:6d} {r["file"]}' for r in res)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
