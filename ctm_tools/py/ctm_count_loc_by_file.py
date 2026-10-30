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

def count_loc_by_file(root='.'):
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
                results.append({'file': str(rel), 'loc': lines})
            except:
                continue
    results.sort(key=lambda x: x['loc'], reverse=True)
    return results

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
    
    res = count_loc_by_file(root)
    result = {'tool': 'ctm_count_loc_by_file', 'root': str(Path(root).resolve()), 'files': res}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'file,loc\n'
        for r in res:
            out += f'"{r["file"]}",{r["loc"]}\n'
    else:
        out = '\n'.join(f'{r["loc"]:6d} {r["file"]}' for r in res)
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
