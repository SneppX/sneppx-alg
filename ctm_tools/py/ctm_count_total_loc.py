#!/usr/bin/env python3
import os
import sys
from pathlib import Path

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}
CODE_EXTS = {'.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.py', '.java', '.go', '.rs', '.js', '.ts', '.jsx', '.tsx', '.sh', '.bash', '.ps1', '.rb', '.php', '.swift', '.kt', '.m', '.mm', '.cu', '.cuh', '.asm', '.s'}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_loc(root='.', count_all_text=True):
    root = Path(root).resolve()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        if should_ignore(dirpath):
            continue
        for fn in filenames:
            full = Path(dirpath) / fn
            if should_ignore(str(full)):
                continue
            if full.suffix.lower() in IGNORE_EXTS and not count_all_text:
                continue
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        total += 1
            except:
                continue
    return total

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(count_loc(root))

if __name__ == '__main__':
    main()
