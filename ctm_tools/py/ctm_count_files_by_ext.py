#!/usr/bin/env python3
"""Count candidate files by extension using the unified CTM path filters."""

import sys

import ctm


def count_by_ext(root="."):
    stats = {}
    for path in ctm.iter_files(root, include_other_text=True):
        ext = path.suffix.lower() or "(no_ext)"
        stats[ext] = stats.get(ext, 0) + 1
    return stats


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for k, v in sorted(count_by_ext(root).items(), key=lambda x: x[1], reverse=True):
        print(f"{v:6d} {k}")


if __name__ == "__main__":
    main()
