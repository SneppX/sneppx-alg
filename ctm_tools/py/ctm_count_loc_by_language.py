#!/usr/bin/env python3
import os
import sys
import json
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
    
    stats, files = count_loc_by_language(root)
    result = {'tool': 'ctm_count_loc_by_language', 'root': str(Path(root).resolve()), 'languages': []}
    for lang, lines in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        result['languages'].append({'language': lang, 'files': files[lang], 'loc': lines})
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'language,files,loc\n'
        for item in result['languages']:
            out += f'{item["language"]},{item["files"]},{item["loc"]}\n'
    else:
        out = '\n'.join(f'{lines:8d} {files[lang]:6d} {lang}' for lang, lines in sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
