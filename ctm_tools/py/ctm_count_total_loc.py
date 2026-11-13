#!/usr/bin/env python3
"""Compatibility wrapper for the unified CTM physical line counter."""
import argparse
from ctm import scan


def count_loc(root=".", count_all_text=True):
    """Count physical lines in readable text files, excluding known binaries."""
    return scan(root, include_other_text=count_all_text)["lines_of_code"]


def main():
    parser = argparse.ArgumentParser(description="Count repository text lines")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--source-only", action="store_true", help="count recognized source/documentation extensions only")
    args = parser.parse_args()
    print(count_loc(args.root, count_all_text=not args.source_only))


if __name__ == "__main__":
    main()
