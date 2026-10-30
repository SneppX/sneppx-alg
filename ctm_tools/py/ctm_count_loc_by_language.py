#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from collections import defaultdict

IGNORE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.next', '.cache', '.pytest_cache', '.ruff_cache'}
IGNORE_EXTS = {'.pyc', '.pyo', '.o', '.obj', '.exe', '.dll', '.so', '.dylib', '.class', '.jar'}

LANG_MAP = {
    '.c': 'C', '.h': 'C/Header', '.cpp': 'C++', '.cc': 'C++', '.cxx': 'C++', '.hpp': 'C++/Header', '.hxx': 'C++/Header',
    '.py': 'Python', '.java': 'Java', '.go': 'Go', '.rs': 'Rust', '.js': 'JavaScript', '.ts': 'TypeScript',
    '.jsx': 'JavaScript', '.tsx': 'TypeScript', '.sh': 'Shell', '.bash': 'Shell', '.ps1': 'PowerShell',
    '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift', '.kt': 'Kotlin', '.m': 'Objective-C', '.mm': 'Objective-C++',
    '.cu': 'CUDA', '.cuh': 'CUDA/Header', '.asm': 'Assembly', '.s': 'Assembly', '.sql': 'SQL', '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML', '.xml': 'XML', '.md': 'Markdown'
}

def should_ignore(path):
    parts = Path(path).parts
    for p in parts:
        if p in IGNORE_DIRS:
            return True
    return False

def count_loc_by_language(root='.'):
    root = Path(root).resolve()
    stats = defaultdict(int)
    files = defaultdict(int)
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
            ext = full.suffix.lower()
            lang = LANG_MAP.get(ext, f'Other({ext})')
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                stats[lang] += lines
                files[lang] += 1
            except:
                continue
    return stats, files

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    stats, files = count_loc_by_language(root)
    # sort by lines desc
    for lang, lines in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f'{lines:8d} {files[lang]:6d} {lang}')

if __name__ == '__main__':
    main()
