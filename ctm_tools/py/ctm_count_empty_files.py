#!/usr/bin/env python3
import os, sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    for p in Path(path).parts:
        if p in IGNORE_DIRS: return True
    return False

def count_empty(root='.'):
    root = Path(root).resolve()
    count=0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath): continue
        for fn in filenames:
            full = Path(dirpath)/fn
            if should_ignore(str(full)): continue
            if full.suffix.lower() in IGNORE_EXTS: continue
            try:
                if full.stat().st_size == 0: count+=1
            except: pass
    return count

def main():
    root = sys.argv[1] if len(sys.argv)>1 else '.'
    print(count_empty(root))

if __name__=='__main__': main()
