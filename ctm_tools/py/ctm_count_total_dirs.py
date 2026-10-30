#!/usr/bin/env python3
import os
import sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_dirs(root='.'):
    root = Path(root).resolve()
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath):
            continue
        # count this dir (skip root if desired? count all dirs including root)
        count += 1
    return count

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(count_dirs(root))

if __name__ == '__main__':
    main()
