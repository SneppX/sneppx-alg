#!/usr/bin/env python3
"""Compatibility summary report built on the unified CTM scanner."""
import argparse
import json
from pathlib import Path

from ctm import scan


def count_files_dirs_loc(root):
    report = scan(root, include_other_text=True)
    return {
        "files": report["files"],
        "dirs": report["directories"] + 1,
        "loc": report["lines_of_code"],
        "bytes": report["bytes"],
        "skipped_binary_or_symlink": report["skipped_binary_or_symlink"],
        "unreadable_files": report["unreadable_files"],
    }


def main():
    parser = argparse.ArgumentParser(description="Create a CTM repository summary")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = scan(args.root, include_other_text=True)
    git = report.get("git", {})
    result = {
        "tool": "ctm_summary_report",
        "root": report["root"],
        "files": report["files"],
        "dirs": report["directories"] + 1,
        "loc": report["lines_of_code"],
        "bytes": report["bytes"],
        "tracked_files": git.get("tracked_files"),
        "untracked_files": git.get("untracked_files"),
        "skipped_binary_or_symlink": report["skipped_binary_or_symlink"],
        "unreadable_files": report["unreadable_files"],
        "languages": report["languages"],
    }

    if args.format == "json":
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
    elif args.format == "csv":
        rendered = "files,dirs,loc,bytes,tracked_files,untracked_files\\n"
        rendered += (
            f"{result['files']},{result['dirs']},{result['loc']},{result['bytes']},"
            f"{result['tracked_files']},{result['untracked_files']}\\n"
        )
    else:
        rendered = (
            f"Root: {result['root']}\\nFiles: {result['files']}\\n"
            f"Dirs (including root): {result['dirs']}\\nLOC: {result['loc']}\\n"
            f"Bytes: {result['bytes']}\\n"
            f"Tracked files (git): {result['tracked_files']}\\n"
            f"Untracked files (git): {result['untracked_files']}\\n"
            f"Skipped binary/symlink: {result['skipped_binary_or_symlink']}\\n"
            f"Unreadable files: {result['unreadable_files']}\\n"
        )

    if args.output:
        with Path(args.output).expanduser().open("w", encoding="utf-8", newline="") as output_file:
            output_file.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
