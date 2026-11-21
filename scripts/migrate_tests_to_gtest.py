#!/usr/bin/env python3
"""
SNEPPX - migrate_tests_to_gtest.py

Mechanically migrates the legacy C test suite (tests/unit, tests/integration,
tests/security) from the hand-rolled harness to GoogleTest.

Patterns found in the 135 .c test files:

  A  (96 files)  per-file ASSERT* macros + run_test() registrations + main()
  B  (13 files)  per-file TEST(name, expr) macro + main() calling test_*
  C  (26 files)  program-shaped int main() using inline CHECKs / logic

Transformations applied:

  * C99 compound literals  (T[]){...}  ->  SX_ARR_C(T, n, ...)     (shim)
  * #include "test_common.h"           ->  #include "test_gtest.h"
  * harness #define blocks removed, call sites renamed SX_*
  * legacy main() removed/rewrapped; gtest_main provides the entry point

The result stays in .c files but is compiled as C++ by tests/CMakeLists.txt.
Run with --dry-run to preview before writing.

Usage:
    python scripts/migrate_tests_to_gtest.py [--dry-run] [--only NAME...]
"""

import argparse
import os
import re
import sys

TESTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
DIRS = ("unit", "integration", "security")

MACRO_NAMES = frozenset([
    "ASSERT", "ASSERT_EQ", "ASSERT_NEAR", "ASSERT_NEAR_ARR",
    "ASSERT_STREQ", "ASSERT_STR_EQ", "ASSERT_NULL", "ASSERT_NOT_NULL",
    "TEST", "FLOAT_CLOSE", "CHECK",
])

COUNTER_RE = re.compile(r"^\s*static\s+int\s+(tests_passed|tests_failed|tests_skipped|pass|fail|failures)\b.*$", re.M)
DEFINE_RE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)", re.M)
RUN_TEST_DEF_RE = re.compile(r"^\s*(?:static\s+)?void\s+run_test\b", re.M)
INCLUDE_RE = re.compile(r"^(\s*#include[^\n]*\n)", re.M)
TEST_COMMON_RE = re.compile(r'#include\s+"[../]*test_common\.h"')

RENAMES = [
    ("ASSERT_NEAR_ARR", "SX_ASSERT_NEAR_ARR"),
    ("ASSERT_NOT_NULL", "SX_ASSERT_NOT_NULL"),
    ("ASSERT_STR_EQ", "SX_ASSERT_STR_EQ"),
    ("ASSERT_STREQ", "SX_ASSERT_STREQ"),
    ("ASSERT_NEAR", "SX_ASSERT_NEAR"),
    ("ASSERT_NULL", "SX_ASSERT_NULL"),
    ("ASSERT_EQ", "SX_ASSERT_EQ"),
    ("ASSERT", "SX_ASSERT"),
    ("FLOAT_CLOSE", "SX_FLOAT_CLOSE"),
]

TYPE_RE = re.compile(r"\(\s*([A-Za-z_]\w*)\s*\[\]\s*\)")
SCALAR_RE = re.compile(r"&\s*\(\s*([A-Za-z_]\w*)\s*\)\s*")

TEST_CALL_RE = re.compile(r"\bTEST\(\s*(\"(?:\\.|[^\"\\])*\")\s*,\s*(.*?)\s*\);", re.S)
RUN_TEST_CALL_RE = re.compile(r"\brun_test\(\s*\"((?:\\.|[^\"\\])*)\"\s*,\s*([A-Za-z_]\w*)\s*\)")
DIRECT_CALL_RE = re.compile(r"^\s*(test_\w+)\s*\(\s*\)\s*;", re.M)
MAIN_RE = re.compile(r"^\s*int\s+main\s*\([^)]*\)\s*\{", re.M)
CHECK_DEF_RE = re.compile(r"#define\s+CHECK\s*\(([^)]*)\)")
CHECK_2ARG_RE = re.compile(r"\bCHECK\(")
RETURN_NUM_RE = re.compile(r"\breturn\s+(-?\d+)\s*;")
CNT_PRINT_RE = re.compile(r"\bprintf\s*\([^;]*\b(pass|fail|failures)\b")
CNT_RETURN_RE = re.compile(r"\breturn\s+[^;]*\b(fail|pass|failures)\b")


def split_elements(s):
    """Split a braced literal body on top-level commas (ignoring nested
    braces and quoted strings)."""
    elems = []
    depth = 0
    instr = None
    start = 0
    for idx, ch in enumerate(s):
        if instr:
            if ch == "\\":
                continue
            if ch == instr:
                instr = None
            continue
        if ch in ("\"", "'"):
            instr = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            elems.append(s[start:idx].strip())
            start = idx + 1
    elems.append(s[start:].strip())
    return [e for e in elems if e]


def _rewrite_compound(content, type_re, prefix):
    """Rewrite compound literals matched by type_re into SX_ARR_C(...).

    prefix is the text emitted before the type (e.g. '' for arrays,
    '&' handled by dropping it)."""
    out = []
    i = 0
    while i < len(content):
        m = type_re.search(content, i)
        if not m:
            out.append(content[i:])
            break
        out.append(content[i:m.start()])
        j = m.end()
        while j < len(content) and content[j] in " \t\r\n":
            j += 1
        if j >= len(content) or content[j] != "{":
            out.append(content[m.start():m.end()])
            i = m.end()
            continue
        depth = 0
        instr = None
        k = j
        while k < len(content):
            ch = content[k]
            if instr:
                if ch == "\\":
                    k += 2
                    continue
                if ch == instr:
                    instr = None
            else:
                if ch in ("\"", "'"):
                    instr = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            k += 1
        if depth != 0:
            out.append(content[m.start():])
            break
        inner = content[j + 1:k]
        elems = split_elements(inner)
        out.append("SX_ARR_C(%s, %d, %s)" % (m.group(1), len(elems), ",".join(elems)))
        i = k + 1
    return "".join(out)


def fix_compound_literals(content):
    """Rewrite C99 compound literals to the SX_ARR_C shim helper."""
    # array form:   (size_t[]){1, 4}
    content = _rewrite_compound(content, TYPE_RE, "")
    # scalar form:  &(float){ 2.0f }   (the '&' is dropped; SX_ARR_C is a pointer)
    content = _rewrite_compound(content, SCALAR_RE, "")
    return content


def remove_define_blocks(lines):
    """Drop #define blocks for the legacy harness macro names."""
    out = []
    i = 0
    while i < len(lines):
        m = DEFINE_RE.match(lines[i])
        if m and m.group(1) in MACRO_NAMES:
            while i < len(lines) and lines[i].rstrip("\r\n").endswith("\\"):
                i += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def remove_function(lines, name_re):
    """Remove a function whose signature matches name_re (first match)."""
    i = 0
    while i < len(lines):
        if name_re.search(lines[i]):
            j = i
            while j < len(lines) and "{" not in lines[j]:
                j += 1
            if j >= len(lines):
                return lines
            depth = 0
            k = j
            while k < len(lines):
                depth += lines[k].count("{") - lines[k].count("}")
                if depth == 0:
                    return lines[:i] + lines[k + 1:]
                k += 1
            return lines
        i += 1
    return lines


def find_main_span(lines):
    """Return (start, end) line indexes of the int main(...) function."""
    for i, ln in enumerate(lines):
        if MAIN_RE.search(ln):
            j = i
            while j < len(lines) and "{" not in lines[j]:
                j += 1
            if j >= len(lines):
                return None
            depth = 0
            k = j
            while k < len(lines):
                depth += lines[k].count("{") - lines[k].count("}")
                if depth == 0:
                    return (i, k)
                k += 1
            return (i, len(lines) - 1)
    return None


def sanitize_ident(s):
    s2 = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    if not s2 or not s2[0].isalpha():
        s2 = "t_" + s2
    return s2


def gen_wrappers(suite, calls):
    """calls: list of (label, fn) or plain fn strings. Returns TEST wrapper lines."""
    used = {}
    lines = []
    for item in calls:
        if isinstance(item, str):
            label = fn = item
        else:
            label, fn = item
        tid = sanitize_ident(label)
        base = tid
        n = used.get(base, 0)
        if n:
            tid = "%s_%d" % (base, n + 1)
        used[base] = used.get(base, 0) + 1
        lines.append("TEST(%s, %s) { %s(); }" % (suite, tid, fn))
    return lines


def classify(content):
    has_test_common = bool(TEST_COMMON_RE.search(content))
    has_define_test = bool(re.search(r"^\s*#define\s+TEST\b", content, re.M))
    has_define_check = bool(re.search(r"^\s*#define\s+CHECK\b", content, re.M))
    has_counters = bool(re.search(r"^\s*static\s+int\s+(pass|fail|failures)\b", content, re.M))
    has_run_test = bool(RUN_TEST_DEF_RE.search(content))
    if has_define_test:
        return "B"
    if has_define_check or (has_counters and not has_run_test):
        return "C"
    if has_run_test or has_test_common:
        return "A"
    return "C"


def migrate(content, suite):
    pattern = classify(content)

    # Capture CHECK arity from the RAW content: the #define is removed below,
    # so it must be inspected before then (vizmon uses a 1-arg CHECK).
    check_arity = None
    if pattern == "C":
        m = CHECK_DEF_RE.search(content)
        if m:
            check_arity = len([p for p in m.group(1).split(",") if p.strip()])

    # 1) C99 compound literals -> SX_ARR_C
    content = fix_compound_literals(content)

    # 2) shim include
    if TEST_COMMON_RE.search(content):
        content = TEST_COMMON_RE.sub('#include "test_gtest.h"', content)
    elif "test_gtest.h" not in content:
        m = INCLUDE_RE.search(content)
        if m:
            content = content[:m.end()] + '#include "test_gtest.h"\n' + content[m.end():]
        else:
            content = '#include "test_gtest.h"\n' + content

    # 3) remove harness macro blocks
    lines = content.splitlines(keepends=True)
    lines = remove_define_blocks(lines)

    # 4) remove counter declarations (Pattern A/B only)
    if pattern in ("A", "B"):
        text = "".join(lines)
        text = COUNTER_RE.sub("", text)
        lines = text.splitlines(keepends=True)

    # 5) remove run_test definition (Pattern A)
    if pattern == "A":
        lines = remove_function(lines, RUN_TEST_DEF_RE)

    content = "".join(lines)

    # 6) rename assertion call sites
    for old, new in RENAMES:
        content = re.sub(r"\b" + old + r"\b", new, content)

    # 7) Pattern B: per-file TEST(name, expr) -> SX_TEST
    if pattern == "B":
        content = TEST_CALL_RE.sub(r"SX_TEST(\1, \2);", content)

    # 8) Pattern C: CHECK -> SX_CHECK / EXPECT_TRUE
    if pattern == "C":
        if check_arity == 1:
            content = CHECK_2ARG_RE.sub("EXPECT_TRUE(", content)
        else:
            content = CHECK_2ARG_RE.sub("SX_CHECK(", content)

    # 9) main() handling
    lines = content.splitlines(keepends=True)
    span = find_main_span(lines)
    warnings = []
    if span is None:
        warnings.append("no int main found")
        return "".join(lines), pattern, warnings

    i, k = span
    main_body = "".join(lines[i + 1:k])

    if pattern in ("A", "B"):
        if pattern == "A":
            calls = RUN_TEST_CALL_RE.findall(main_body)
            if not calls:
                calls = DIRECT_CALL_RE.findall(main_body)
        else:
            calls = DIRECT_CALL_RE.findall(main_body)
        if not calls:
            warnings.append("main had no registered tests")
        out = lines[:i] + lines[k + 1:]
        out += ["\n"]
        out += [w + "\n" for w in gen_wrappers(suite, calls)]
        result = "".join(out)
    else:
        # Pattern C: wrap the whole main body in a single TEST
        body_lines = []
        for ln in lines[i + 1:k]:
            if CNT_PRINT_RE.search(ln) or CNT_RETURN_RE.search(ln):
                continue
            rm = RETURN_NUM_RE.search(ln)
            if rm:
                val = rm.group(1)
                if val == "0":
                    body_lines.append(ln.replace(rm.group(0), "return;"))
                else:
                    body_lines.append(
                        ln.replace(rm.group(0),
                                   'FAIL() << "early exit (legacy return %s)";' % val))
                continue
            body_lines.append(ln)
        out = lines[:i]
        out.append("TEST(%s, suite) {\n" % suite)
        out += body_lines
        out.append("}\n")
        out += lines[k + 1:]
        result = "".join(out)

    # 9b) edge-case: test bodies that manually bump legacy pass/fail counters
    #     (e.g. trainer tests use `tests_passed++; return;` as a skip signal)
    result = re.sub(r"\btests_passed\+\+;",
                    'GTEST_SKIP() << "legacy pass-on-skip";', result)
    result = re.sub(r"\btests_failed\+\+;",
                    'ADD_FAILURE() << "legacy failure";', result)

    # 10) sanity checks
    if re.search(r"\brun_test\b", result):
        warnings.append("residual run_test reference")
    if re.search(r"^\s*int\s+main\b", result, re.M):
        warnings.append("residual int main")
    if re.search(r"^\s*#define\s+(ASSERT|TEST|CHECK|FLOAT_CLOSE)\b", result, re.M):
        warnings.append("residual harness #define")
    if re.search(r"\btests_passed\b|\btests_failed\b", result):
        warnings.append("residual tests_passed/tests_failed reference")
    return result, pattern, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing files")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only migrate files whose stem matches one of these names")
    args = ap.parse_args()

    files = []
    for d in DIRS:
        base = os.path.join(TESTS_ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, names in os.walk(base):
            for n in sorted(names):
                if n.endswith(".c"):
                    files.append(os.path.join(root, n))

    stats = {"A": 0, "B": 0, "C": 0}
    problems = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.only and not any(stem.startswith(o) for o in args.only):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        if '#include "test_gtest.h"' in content:
            continue
        suite = sanitize_ident(stem)
        new_content, pattern, warnings = migrate(content, suite)
        stats[pattern] += 1
        if args.dry_run:
            print("[dry-run] %-40s pattern=%s warnings=%s" % (stem, pattern, warnings or "-"))
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new_content)
        print("[migrated] %-40s pattern=%s" % (stem, pattern))
        if warnings:
            problems.append((path, warnings))

    print("\nSummary: %d files, patterns A=%d B=%d C=%d" %
          (len(files), stats["A"], stats["B"], stats["C"]))
    if problems:
        print("\nFiles needing manual attention:")
        for path, warns in problems:
            print("  %s: %s" % (path, "; ".join(warns)))


if __name__ == "__main__":
    main()
