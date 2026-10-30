#!/usr/bin/env python3
import os
import sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_loc_by_file(root='.', format='text'):
    root = Path(root).resolve()
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath):
            continue
        for fn in filenames:
            full = Path(dirpath) / fn
            if should_ignore(str(full)):
                continue
            if full.suffix.lower() in IGNORE_EXTS:
                continue
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                rel = full.relative_to(root)
                results.append((str(rel), lines))
            except:
                continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    res = count_loc_by_file(root)
    for p, l in res:
        print(f'{l:6d} {p}')

if __name__ == '__main__':
    main()
