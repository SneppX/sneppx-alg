#!/usr/bin/env python3
"""List large regular files without following symlinks or scanning generated trees."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "build-deps",
    "build-deps2",
    "dist",
    "site",
    ".next",
    ".cache",
}


def find_large_files(root=".", min_size=1024 * 1024, limit=50):
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"scan root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"scan root is not a directory: {base}")
    results = []
    errors = 0
    for current, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORE_DIRS and not (current_path / name).is_symlink()
        ]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                errors += 1
                continue
            if not path.is_file() or stat.st_size < min_size:
                continue
            results.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "bytes": stat.st_size,
                }
            )
    results.sort(key=lambda item: (-item["bytes"], item["path"]))
    if limit > 0:
        results = results[:limit]
    return {"root": str(base), "min_size": min_size, "files": results, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find large regular files")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--min-size", type=int, default=1024 * 1024, help="minimum size in bytes"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="maximum results; 0 means no limit"
    )
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.min_size < 0 or args.limit < 0:
        parser.error("--min-size and --limit must be non-negative")
    try:
        report = find_large_files(args.root, args.min_size, args.limit)
    except OSError as exc:
        parser.error(str(exc))
    if args.format == "json":
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    elif args.format == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "bytes"])
        for item in report["files"]:
            writer.writerow([item["path"], item["bytes"]])
        rendered = stream.getvalue()
    else:
        rendered = "".join(
            f"{item['bytes']:>12,}  {item['path']}\n" for item in report["files"]
        )
        if not rendered:
            rendered = "No files matched the size threshold.\n"
        rendered += f"Unreadable entries: {report['errors']}\n"
    if args.output:
        with Path(args.output).expanduser().open(
            "w", encoding="utf-8", newline=""
        ) as output:
            output.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
