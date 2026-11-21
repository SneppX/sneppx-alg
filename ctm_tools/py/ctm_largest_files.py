#!/usr/bin/env python3
import os, sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    for p in Path(path).parts:
        if p in IGNORE_DIRS: return True
    return False

def largest_files(root='.', n=50):
    root = Path(root).resolve()
    res=[]
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath): continue
        for fn in filenames:
            full = Path(dirpath)/fn
            if should_ignore(str(full)): continue
            if full.suffix.lower() in IGNORE_EXTS: continue
            try:
                sz = full.stat().st_size
                rel = full.relative_to(root)
                res.append((sz, str(rel)))
            except: pass
    res.sort(reverse=True)
    return res[:n]

def main():
    root = sys.argv[1] if len(sys.argv)>1 else '.'
    n = int(sys.argv[2]) if len(sys.argv)>2 else 50
    for sz,p in largest_files(root,n):
        print(f'{sz:12d} {p}')

if __name__=='__main__': main()
