#!/usr/bin/env python3
import os, sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}

def should_ignore(path):
    for p in Path(path).parts:
        if p in IGNORE_DIRS: return True
    return False

def dir_sizes(root='.'):
    root = Path(root).resolve()
    sizes = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath): continue
        total = 0
        for fn in filenames:
            full = Path(dirpath)/fn
            if should_ignore(str(full)): continue
            try: total += full.stat().st_size
            except: pass
        sizes[str(Path(dirpath).relative_to(root) or '.')] = total
    return sizes

def main():
    root = sys.argv[1] if len(sys.argv)>1 else '.'
    for k,v in sorted(dir_sizes(root).items(), key=lambda x:x[1], reverse=True)[:40]:
        print(f'{v:12d} {k}')

if __name__=='__main__': main()
