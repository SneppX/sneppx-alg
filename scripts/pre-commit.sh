#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

red() {
    printf "\033[31m%s\033[0m\n" "$*"
}

green() {
    printf "\033[32m%s\033[0m\n" "$*"
}

check() {
    local name="$1"
    shift
    if "$@"; then
        green "  PASS  $name"
    else
        red "  FAIL  $name"
        FAILED=1
    fi
}

no_large_files() {
    ! find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o -type f -size +1M -print -quit | grep -q .
}

no_secrets() {
    ! grep -rnE 'AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY' "$ROOT_DIR" \
        --include='*.c' --include='*.h' --include='*.cpp' --include='*.hpp' \
        --include='*.py' --include='*.sh' --include='*.yml' --include='*.yaml' \
        --include='*.json' --include='*.toml' --include='*.cfg' --include='*.ini' --include='*.conf' \
        --exclude='test_secrets_redact.py' --exclude='pre-commit.sh' --exclude-dir='.git' --exclude-dir='build' 2>/dev/null
}

no_merge_conflicts() {
    local matches
    matches="$(find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \
        -o -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' \
        -o -name '*.json' -o -name '*.toml' -o -name '*.md' -o -name '*.txt' \) \
        -type f -exec grep -nE '^(<<<<<<<|=======|>>>>>>>)' {} + 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        printf '%s\\n' "$matches"
        return 1
    fi
}

no_trailing_whitespace() {
    ! grep -rnE '[[:blank:]]+$' "$ROOT_DIR" \
        --include='*.c' --include='*.h' --include='*.cpp' --include='*.hpp' \
        --include='*.py' --include='*.sh' --include='*.yml' --include='*.yaml' \
        --include='*.json' --include='*.toml' --include='*.md' \
        --exclude-dir='.git' --exclude-dir='build' 2>/dev/null
}

final_newlines() {
    find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.py' \
        -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' -o -name '*.json' \
        -o -name '*.toml' -o -name '*.md' \) -type f \
        -exec sh -c 'for file do
            if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -ne 1 ]; then
                exit 1
            fi
        done' sh {} +
}

clang_format_check() {
    find "$ROOT_DIR" -path '*/build/*' -prune -o -path '*/.git/*' -prune -o \
        \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) -type f \
        -exec sh -c 'for file do clang-format --dry-run --Werror "$file" || exit 1; done' sh {} +
}

check "No large files" no_large_files
check "No secrets" no_secrets
check "No merge conflict markers" no_merge_conflicts
check "No trailing whitespace" no_trailing_whitespace
check "Final newline" final_newlines

if command -v clang-format >/dev/null 2>&1; then
    check "clang-format (C/C++)" clang_format_check
else
    echo "  SKIP  clang-format (not installed)"
fi

if command -v black >/dev/null 2>&1; then
    check "black (Python)" black --check --quiet "$ROOT_DIR/bindings/python"
else
    echo "  SKIP  black (not installed)"
fi

if command -v sneppx-analyze >/dev/null 2>&1; then
    check "sneppx-analyze (security)" bash -c 'cd "$1" && sneppx-analyze --dirs kernel algorithms net security >/dev/null 2>&1' _ "$ROOT_DIR"
else
    echo "  SKIP  sneppx-analyze (not installed)"
fi

if command -v sneppx-format >/dev/null 2>&1; then
    check "sneppx-format (lint)" bash -c 'cd "$1" && sneppx-format --lint kernel algorithms net >/dev/null 2>&1' _ "$ROOT_DIR"
else
    echo "  SKIP  sneppx-format (not installed)"
fi

if command -v sneppx-deps >/dev/null 2>&1; then
    check "sneppx-deps (circular)" bash -c 'cd "$1" && sneppx-deps --circular . >/dev/null 2>&1' _ "$ROOT_DIR"
else
    echo "  SKIP  sneppx-deps (not installed)"
fi

if [ "$FAILED" -eq 1 ]; then
    red "\nPre-commit checks FAILED. Fix errors before committing."
    exit 1
fi

green "\nAll pre-commit checks passed."
