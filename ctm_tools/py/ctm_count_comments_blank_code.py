#!/usr/bin/env python3
"""Count blank, comment and code lines via the unified CTM text-file filters."""

import sys
import json
from pathlib import Path

import ctm

CODE_EXTS = {
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hxx",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".cu",
    ".cuh",
}


def count_lines(root="."):
    total = blank = comments = code = 0
    for path in ctm.text_files(root):
        if path.suffix.lower() not in CODE_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            total += 1
            s = line.strip()
            if not s:
                blank += 1
            elif s.startswith("//") or s.startswith("#") or s.startswith("/*") or s.startswith("*"):
                comments += 1
            else:
                code += 1
    return {"total": total, "blank": blank, "comments": comments, "code": code}


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

    stats = count_lines(root)
    result = {
        "tool": "ctm_count_comments_blank_code",
        "root": str(Path(root).resolve()),
        **stats,
    }

    if fmt == "json":
        out = json.dumps(result, indent=2)
    elif fmt == "csv":
        out = "total,blank,comments,code\n"
        out += f"{stats['total']},{stats['blank']},{stats['comments']},{stats['code']}\n"
    else:
        out = f"Total: {stats['total']}\nBlank: {stats['blank']}\nComments: {stats['comments']}\nCode: {stats['code']}\n"

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
