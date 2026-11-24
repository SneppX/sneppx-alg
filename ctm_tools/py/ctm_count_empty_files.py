#!/usr/bin/env python3
"""Count zero-length files using the unified CTM path filters."""

import sys

import ctm


def count_empty(root="."):
    count = 0
    for path in ctm.iter_files(root, include_other_text=True):
        try:
            if path.stat().st_size == 0:
                count += 1
        except OSError:
            continue
    return count


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(count_empty(root))


if __name__ == "__main__":
    main()
