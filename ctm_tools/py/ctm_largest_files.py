#!/usr/bin/env python3
"""Legacy large-file listing — delegates to :mod:`ctm_find_large_files`.

Kept for backwards compatibility with existing invocations. New code should
use ``sneppx-ctm-large`` / ``ctm_find_large_files``, which shares the unified
scanner's path filters (generated directories, binary suffixes, symlinks).
"""

import sys

from ctm_find_large_files import find_large_files


def largest_files(root=".", n=50):
    """Return ``(size_bytes, relative_path)`` pairs, largest first."""
    report = find_large_files(root, min_size=0, limit=n)
    return [(item["bytes"], item["path"]) for item in report["files"]]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    for sz, p in largest_files(root, n):
        print(f"{sz:12d} {p}")


if __name__ == "__main__":
    main()
