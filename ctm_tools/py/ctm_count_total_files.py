#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_files(root='.'):
    root = Path(root).resolve()
    count = 0
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
            count += 1
    return count

def main():
    args = sys.argv[1:]
    fmt = 'text'
    output = None
    root = '.'
    i = 0
    while i < len(args):
        if args[i] == '--format' and i+1 < len(args):
            fmt = args[i+1]
            i += 2
            continue
        if args[i] == '--output' and i+1 < len(args):
            output = args[i+1]
            i += 2
            continue
        if not args[i].startswith('-'):
            root = args[i]
        i += 1
    
    count = count_files(root)
    result = {'tool': 'ctm_count_total_files', 'root': str(Path(root).resolve()), 'count': count}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = f'tool,root,count\nctm_count_total_files,{result["root"]},{count}\n'
    else:
        out = str(count)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
