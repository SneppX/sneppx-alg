#!/usr/bin/env python3
import subprocess
import sys
import json
from pathlib import Path

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
    
    try:
        r = subprocess.run(['git', '-C', root, 'ls-files'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tracked = len(r.stdout.decode('utf-8', errors='ignore').splitlines())
        r2 = subprocess.run(['git', '-C', root, 'ls-files', '--others', '--exclude-standard'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        untracked = len(r2.stdout.decode('utf-8', errors='ignore').splitlines())
        result = {'tool': 'ctm_git_tracked_vs_untracked', 'root': str(Path(root).resolve()), 'tracked': tracked, 'untracked': untracked}
    except Exception as e:
        result = {'tool': 'ctm_git_tracked_vs_untracked', 'root': str(Path(root).resolve()), 'error': str(e)}
    
    if fmt == 'json':
        out = json.dumps(result, indent=2)
    elif fmt == 'csv':
        out = 'tracked,untracked\n'
        out += f'{result.get("tracked",0)},{result.get("untracked",0)}\n'
    else:
        out = f'Tracked: {result.get("tracked",0)}\nUntracked: {result.get("untracked",0)}\n'
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        print(out)

if __name__ == '__main__':
    main()
