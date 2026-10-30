#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

CODE_EXTS = {'.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.py', '.js', '.ts', '.java', '.go', '.rs', '.cu', '.cuh'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_lines(root='.'):
    root = Path(root).resolve()
    total = blank = comments = code = 0
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
            if full.suffix.lower() not in CODE_EXTS:
                continue
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        total += 1
                        s = line.strip()
                        if not s:
                            blank += 1
                        elif s.startswith('//') or s.startswith('#') or s.startswith('/*') or s.startswith('*'):
                            comments += 1
                        else:
                            code += 1
            except:
                continue
    return {'total': total, 'blank': blank, 'comments': comments, 'code': code}

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
    
    stats = count_lines(root)
    result = {'tool': 'ctm_count_comments_blank_code', 'root': str(Path(root).resolve()), **stats}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'total,blank,comments,code\n'
        out += f'{stats["total"]},{stats["blank"]},{stats["comments"]},{stats["code"]}\n'
    else:
        out = f'Total: {stats["total"]}\nBlank: {stats["blank"]}\nComments: {stats["comments"]}\nCode: {stats["code"]}\n'
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
