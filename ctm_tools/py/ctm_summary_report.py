#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

def count_files_dirs_loc(root):
    root = Path(root).resolve()
    files = dirs = loc = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        dirs += 1
        for fn in filenames:
            full = Path(dirpath) / fn
            if full.suffix.lower() in IGNORE_EXTS: continue
            files += 1
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    loc += sum(1 for _ in f)
            except: pass
    return {'files': files, 'dirs': dirs, 'loc': loc}

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
    
    stats = count_files_dirs_loc(root)
    try:
        r = subprocess.run(['git', '-C', root, 'ls-files'], stdout=subprocess.PIPE)
        tracked = len(r.stdout.decode('utf-8', errors='ignore').splitlines())
    except:
        tracked = 0
    result = {'tool': 'ctm_summary_report', 'root': str(Path(root).resolve()), **stats, 'tracked_files': tracked}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'files,dirs,loc,tracked_files\n'
        out += f'{stats["files"]},{stats["dirs"]},{stats["loc"]},{tracked}\n'
    else:
        out = f'Root: {result["root"]}\nFiles: {stats["files"]}\nDirs: {stats["dirs"]}\nLOC: {stats["loc"]}\nTracked files (git): {tracked}\n'
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
