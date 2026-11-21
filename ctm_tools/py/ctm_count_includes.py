#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_includes(root='.'):
    root = Path(root).resolve()
    includes = defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath):
            continue
        for fn in filenames:
            if not (fn.endswith('.c') or fn.endswith('.h') or fn.endswith('.cpp') or fn.endswith('.hpp') or fn.endswith('.cc')):
                continue
            full = Path(dirpath) / fn
            if should_ignore(str(full)):
                continue
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith('#include'):
                            includes[s] += 1
            except:
                continue
    return includes

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
    
    includes = count_includes(root)
    result = {'tool': 'ctm_count_includes', 'root': str(Path(root).resolve()), 'includes': dict(sorted(includes.items(), key=lambda x: x[1], reverse=True))}

    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'include,count\n'
        for k,v in sorted(includes.items(), key=lambda x: x[1], reverse=True):
            out += f'"{k}",{v}\n'
    else:
        out = '\n'.join(f'{v:6d} {k}' for k,v in sorted(includes.items(), key=lambda x: x[1], reverse=True)[:50])
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
