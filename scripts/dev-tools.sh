#!/usr/bin/env bash
# SNEPPX Developer Tool Chain — run all 7 tools against the ARIX_Algo codebase.
# Usage:
#   ./scripts/dev-tools.sh [--all] [--no-test] [--build-dir build]
#
# Requires: sneppx-analyze, sneppx-format, sneppx-deps, sneppx-stats,
#           sneppx-test, sneppx-bench, sneppx-serve (all via `pip install sneppx-toolkit[all]`)
set -uo pipefail

ALL=0
NO_TEST=0
BUILD_DIR="build"

while [ $# -gt 0 ]; do
    case "$1" in
        --all) ALL=1; shift ;;
        --no-test) NO_TEST=1; shift ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
skip()  { printf "\033[33m  SKIP  %s\033[0m\n" "$*"; }

check() {
    local name="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        green "  PASS  $name"
    else
        red   "  FAIL  $name"
        FAILED=1
    fi
}

has() { command -v "$1" >/dev/null 2>&1; }

echo "SNEPPX Developer Tool Chain"
echo "Root: $ROOT_DIR"
echo

# 1. Security scan
if has sneppx-analyze; then
    check "Security scan (kernel, algorithms, net, security)" \
        sneppx-analyze --dirs kernel algorithms net security
else
    skip "sneppx-analyze (pip install sneppx-toolkit[all])"
fi

# 2. Format / lint
if has sneppx-format; then
    check "Formatting / lint (C/C++ core)" \
        sneppx-format --lint kernel, algorithms, net
else
    skip "sneppx-format (pip install sneppx-toolkit[all])"
fi

# 3. Circular dependencies
if has sneppx-deps; then
    check "Circular dependency check" sneppx-deps --circular "$ROOT_DIR"
else
    skip "sneppx-deps (pip install sneppx-toolkit[all])"
fi

# 4. Code statistics
if has sneppx-stats; then
    mkdir -p "$ROOT_DIR/.sneppx"
    check "Code statistics" sneppx-stats --save "$ROOT_DIR/.sneppx/stats.json" "$ROOT_DIR"
else
    skip "sneppx-stats (pip install sneppx-toolkit[all])"
fi

# 5. Test suite
if [ "$NO_TEST" -eq 0 ] && [ -d "$ROOT_DIR/$BUILD_DIR" ]; then
    if has sneppx-test; then
        check "Test suite (sneppx-test)" \
            sneppx-test --build-dir "$ROOT_DIR/$BUILD_DIR" --exclude cuda
    else
        check "Test suite (ctest)" bash -c "
            cmake --build '$ROOT_DIR/$BUILD_DIR' --config Release &&
            ctest --test-dir '$ROOT_DIR/$BUILD_DIR' -C Release --output-on-failure
        "
    fi
else
    skip "Test suite (no $BUILD_DIR directory — run scripts/build.sh first)"
fi

# 6. Benchmarks
if [ "$ALL" -eq 1 ] && has sneppx-bench && [ -d "$ROOT_DIR/$BUILD_DIR" ]; then
    check "Benchmarks" sneppx-bench --build-dir "$ROOT_DIR/$BUILD_DIR"
fi

if [ "$FAILED" -eq 1 ]; then
    red "\nDev tool chain FAILED — fix issues before committing."
    exit 1
fi

green "\nDev tool chain PASSED."
exit 0
