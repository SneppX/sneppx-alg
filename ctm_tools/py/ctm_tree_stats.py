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

def tree_stats(root='.'):
    root = Path(root).resolve()
    depth_count = defaultdict(int)
    maxd = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        rel = Path(dirpath).relative_to(root)
        d = len(rel.parts) if str(rel) != '.' else 0
        if d > maxd: maxd = d
        depth_count[d] += len([f for f in filenames if Path(dirpath)/f not in []])
    return {'max_depth': maxd, 'depth_distribution': dict(depth_count)}

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
    
    stats = tree_stats(root)
    result = {'tool': 'ctm_tree_stats', 'root': str(Path(root).resolve()), **stats}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'depth,count\n'
        for k,v in sorted(stats['depth_distribution'].items()):
            out += f'{k},{v}\n'
    else:
        out = f'Max depth: {stats["max_depth"]}\n'
        for k,v in sorted(stats['depth_distribution'].items()):
            out += f'Depth {k}: {v} dirs\n'
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
