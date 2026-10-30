#!/usr/bin/env python3
import os, sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    for p in Path(path).parts:
        if p in IGNORE_DIRS: return True
    return False

def count_by_ext(root='.'):
    root = Path(root).resolve()
    stats = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath): continue
        for fn in filenames:
            full = Path(dirpath)/fn
            if should_ignore(str(full)): continue
            if full.suffix.lower() in IGNORE_EXTS: continue
            ext = full.suffix.lower() or '(no_ext)'
            stats[ext] = stats.get(ext,0)+1
    return stats

def main():
    root = sys.argv[1] if len(sys.argv)>1 else '.'
    for k,v in sorted(count_by_ext(root).items(), key=lambda x:x[1], reverse=True):
        print(f'{v:6d} {k}')

if __name__=='__main__': main()
