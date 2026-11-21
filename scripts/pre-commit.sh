#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }

check() {
    local name="$1"
    shift
    if "$@"; then
        green "  PASS  $name"
    else
        red   "  FAIL  $name"
        FAILED=1
    fi
}

# 1. No large files (>1 MB)
check "No large files" sh -c "
    find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        -type f -size +1M -print | grep -q . && exit 1 || exit 0
"

# 2. No secrets (AWS keys, private keys, passwords)
check "No secrets" sh -c "
    ! grep -rnE 'AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY|password\s*=\s*['\"'"'"']' \
        "$ROOT_DIR" --include='*.c' --include='*.h' --include='*.cpp' --include='*.hpp' \
        --include='*.py' --include='*.sh' --include='*.yml' --include='*.yaml' --include='*.json' \
        --include='*.toml' --include='*.cfg' --include='*.ini' --include='*.conf' \
        --exclude-dir='.git' --exclude-dir='build' 2>/dev/null | grep -v '^\s*$' | grep -q .
"

# 3. No merge conflict markers
check "No merge conflict markers" sh -c "
    ! find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \
           -o -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' \
           -o -name '*.json' -o -name '*.toml' -o -name '*.md' -o -name '*.txt' \) \
        -exec grep -q '=======' {} \; -print | grep -q . && exit 1 || exit 0
"

# 4. Trailing whitespace
check "No trailing whitespace" sh -c "
    ! grep -rn '[[:space:]]$' "$ROOT_DIR" \
        --include='*.c' --include='*.h' --include='*.cpp' --include='*.hpp' \
        --include='*.py' --include='*.sh' --include='*.yml' --include='*.yaml' \
        --include='*.json' --include='*.toml' --include='*.md' \
        --exclude-dir='.git' --exclude-dir='build' 2>/dev/null | grep -q .
"

# 5. Final newline
check "Final newline" sh -c "
    find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.py' \
           -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' -o -name '*.json' \
           -o -name '*.toml' -o -name '*.md' \) -print | while IFS= read -r f; do
        [ -s "$f" ] && [ \"\$(tail -c 1 \"\$f\")\" != \"\" ] && exit 1
    done 2>/dev/null
"

# 6. clang-format check (C/C++)
if command -v clang-format &>/dev/null; then
    check "clang-format (C/C++)" sh -c "
        find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
            \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) -print | \
        while IFS= read -r f; do
            clang-format --dry-run --Werror \"\$f\" 2>/dev/null || exit 1
        done
    "
else
    echo "  SKIP  clang-format (not installed)"
fi

# 7. Black check (Python)
if command -v black &>/dev/null; then
    check "black (Python)" sh -c "
        black --check --quiet "$ROOT_DIR/src/python" 2>/dev/null
    "
else
    echo "  SKIP  black (not installed)"
fi

# 8. CMake syntax check
if command -v cmake &>/dev/null; then
    check "CMake syntax" sh -c "
        cmake -P "$ROOT_DIR/CMakeLists.txt" 2>&1 | grep -q 'Parse error' && exit 1 || exit 0
    "
else
    echo "  SKIP  cmake (not installed)"
fi

# 9. SNEPPX dev tools (if installed via sneppx-toolkit)
if command -v sneppx-analyze &>/dev/null; then
    check "sneppx-analyze (security)" sh -c "
        cd "$ROOT_DIR" && sneppx-analyze --dirs kernel algorithms net security >/dev/null 2>&1
    "
else
    echo "  SKIP  sneppx-analyze (pip install sneppx-analyze)"
fi

if command -v sneppx-format &>/dev/null; then
    check "sneppx-format (lint)" sh -c "
        cd "$ROOT_DIR" && sneppx-format --lint kernel algorithms net >/dev/null 2>&1
    "
else
    echo "  SKIP  sneppx-format (pip install sneppx-format)"
fi

if command -v sneppx-deps &>/dev/null; then
    check "sneppx-deps (circular)" sh -c "
        cd "$ROOT_DIR" && sneppx-deps --circular . >/dev/null 2>&1
    "
else
    echo "  SKIP  sneppx-deps (pip install sneppx-deps)"
fi

if [ "$FAILED" -eq 1 ]; then
    red "\nPre-commit checks FAILED. Fix errors before committing."
    exit 1
fi

green "\nAll pre-commit checks passed."
