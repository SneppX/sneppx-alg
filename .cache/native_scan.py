"""Static native-API audit: declared prototypes in headers vs definitions in .c/.cpp."""
import os
import re
import sys

ROOT = r"C:\Users\PC\SneppX\sneppx-alg"
SKIP = {"build", "site", ".git", ".cache", "node_modules", "__pycache__", "archive"}

PROTO = re.compile(
    r'^\s*(?:SNEPPX_API\s+|SNEPPX_EXPORT\s+)?'
    r'(?:[A-Za-z_][A-Za-z0-9_]*\s*\**\s*)+'
    r'(SNEPPX_[A-Za-z0-9_]+)\s*\(([^;{]*)\)\s*;'
)
DEF = re.compile(r'^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*\**\s*)+(SNEPPX_[A-Za-z0-9_]+)\s*\(')


def walk(exts):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for fn in filenames:
            if os.path.splitext(fn)[1] in exts:
                yield os.path.join(dirpath, fn)


headers = list(walk({".h", ".hpp"}))
sources = list(walk({".c", ".cpp", ".cu"}))

declared = {}
for path in headers:
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                m = PROTO.match(line)
                if m:
                    declared.setdefault(m.group(1), (rel, i))
    except OSError:
        pass

defined = {}
for path in sources:
    rel = os.path.relpath(path, ROOT)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = DEF.match(line)
                if m and m.group(1) in declared:
                    defined.setdefault(m.group(1), rel)
    except OSError:
        pass

missing = {k: v for k, v in declared.items() if k not in defined}

print(f"headers scanned      : {len(headers)}")
print(f"sources scanned      : {len(sources)}")
print(f"distinct SNEPPX_* protos declared: {len(declared)}")
print(f"  ...with a definition found    : {len(defined)}")
print(f"  ...NO definition found        : {len(missing)}")
print()
groups = {}
for name, (hdr, line) in missing.items():
    groups.setdefault(hdr, []).append((name, line))
for hdr in sorted(groups, key=lambda h: -len(groups[h]))[:30]:
    print(f"{len(groups[hdr]):4d}  {hdr}")
    for name, line in sorted(groups[hdr])[:6]:
        print(f"          {name}:{line}")