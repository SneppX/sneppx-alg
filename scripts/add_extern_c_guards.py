#!/usr/bin/env python3
"""
SNEPPX - add_extern_c_guards.py

Adds `extern "C"` linkage guards to the SNEPPX C headers that are included by
the migrated test suite. The test sources are compiled as C++ (GoogleTest),
so every C header they pull in must declare its functions with C linkage or
the names get mangled and fail to link against the C libraries.

Only headers without an existing `extern "C"` are touched. Headers that
already declare C linkage are left alone.

Usage:
    python scripts/add_extern_c_guards.py [--dry-run]
"""

import argparse
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TESTS_ROOT = os.path.join(ROOT, "tests")
DIRS = ("unit", "integration", "security")

# Mirrors target_include_directories() in tests/CMakeLists.txt
INCLUDE_DIRS = [
    os.path.join(ROOT, "include"),
    os.path.join(ROOT, "include/neural_core/kernel"),
    os.path.join(ROOT, "include/neural_core/architecture"),
    os.path.join(ROOT, "include/neural_core/security"),
    os.path.join(ROOT, "tests"),
    os.path.join(ROOT, "lib/internal"),
    os.path.join(ROOT, "drivers/cuda"),
    os.path.join(ROOT, "drivers/tpu"),
    os.path.join(ROOT, "drivers/rocm"),
    os.path.join(ROOT, "drivers/http"),
    os.path.join(ROOT, "drivers/zk"),
    os.path.join(ROOT, "drivers/amd"),
    os.path.join(ROOT, "drivers/intel"),
    os.path.join(ROOT, "drivers/qualcomm"),
    os.path.join(ROOT, "drivers/npu"),
    os.path.join(ROOT, "drivers/sgx"),
    os.path.join(ROOT, "drivers/shim"),
    os.path.join(ROOT, "drivers/vulkan"),
    os.path.join(ROOT, "fs/format"),
    os.path.join(ROOT, "security/monitor"),
    os.path.join(ROOT, "security/memory"),
    os.path.join(ROOT, "security/crypto/c"),
    os.path.join(ROOT, "security/network"),
    os.path.join(ROOT, "security/updates"),
    os.path.join(ROOT, "security/formal"),
    os.path.join(ROOT, "security/pentest"),
    os.path.join(ROOT, "security/ai"),
    os.path.join(ROOT, "security/ui"),
    os.path.join(ROOT, "kernel"),
]

QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
GUARD_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_]\w*)")
GUARD_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*$")
PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once")
CONDITIONAL_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
ENDIF_RE = re.compile(r"^\s*#\s*endif\b")

OPEN_BLOCK = [
    "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n",
]
CLOSE_BLOCK = [
    "\n#ifdef __cplusplus\n}\n#endif\n",
]


def resolve_include(includer_path, name):
    """Resolve a quoted include relative to includer dir, then include dirs."""
    cand = os.path.normpath(os.path.join(os.path.dirname(includer_path), name))
    if os.path.isfile(cand):
        return cand
    for d in INCLUDE_DIRS:
        cand = os.path.normpath(os.path.join(d, name))
        if os.path.isfile(cand):
            return cand
    return None


def collect_headers():
    """Collect every header (direct + transitive) included by test sources."""
    headers = {}          # path -> list of includers
    pending = []
    seen = set()

    def note(path, includer):
        if path not in headers:
            headers[path] = []
        headers[path].append(includer)

    for d in DIRS:
        base = os.path.join(TESTS_ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, names in os.walk(base):
            for n in sorted(names):
                if n.endswith(".c"):
                    pending.append(os.path.join(root, n))

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    m = QUOTED_INCLUDE_RE.match(ln)
                    if not m:
                        continue
                    inc = resolve_include(path, m.group(1))
                    if inc and inc not in seen:
                        note(inc, path)
                        pending.append(inc)
        except OSError:
            continue
    return headers


def find_guard(lines):
    """Return (guard_name, open_idx, define_idx, close_idx) or None."""
    open_idx = define_idx = None
    name = None
    for i, ln in enumerate(lines):
        if ln.startswith(("/*", "*", "//")) or not ln.strip():
            continue
        m = GUARD_IFNDEF_RE.match(ln)
        if m:
            name = m.group(1)
            open_idx = i
            break
        if PRAGMA_ONCE_RE.match(ln):
            return None
    if open_idx is None:
        return None
    for j in range(open_idx + 1, len(lines)):
        m = GUARD_DEFINE_RE.match(lines[j])
        if m and m.group(1) == name:
            define_idx = j
            break
    if define_idx is None:
        return None
    depth = 0
    for k in range(open_idx, len(lines)):
        if CONDITIONAL_RE.match(lines[k]):
            depth += 1
        elif ENDIF_RE.match(lines[k]):
            depth -= 1
            if depth == 0:
                return (name, open_idx, define_idx, k)
    return None


def needs_guard(lines):
    text = "".join(lines)
    return "extern \"C\"" not in text


def patch_header(path, lines):
    guard = find_guard(lines)
    if guard:
        name, open_idx, define_idx, close_idx = guard
        if define_idx + 1 <= len(lines):
            before = lines[: define_idx + 1]
            after = lines[define_idx + 1 : close_idx]
            tail = lines[close_idx:]
        else:
            return lines
        out = before + OPEN_BLOCK + after + CLOSE_BLOCK + tail
        return out
    # no include guard: wrap the entire file
    out = OPEN_BLOCK + lines
    out.extend(CLOSE_BLOCK)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report headers without writing")
    args = ap.parse_args()

    headers = collect_headers()
    patched = 0
    skipped = 0
    for path in sorted(headers):
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        if not needs_guard(lines):
            skipped += 1
            continue
        if not path.endswith((".h", ".hpp")):
            print("  [skip non-header] %s" % rel)
            continue
        if os.path.basename(path) == "test_gtest.h":
            # C++ shim - must NOT get C linkage
            skipped += 1
            continue
        new_lines = patch_header(path, lines)
        if args.dry_run:
            print("  [patch] %s" % rel)
            patched += 1
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(new_lines)
        print("  [patched] %s" % rel)
        patched += 1

    print("\nSummary: %d unique headers, %d patched, %d already guarded" %
          (len(headers), patched, skipped))


if __name__ == "__main__":
    main()
