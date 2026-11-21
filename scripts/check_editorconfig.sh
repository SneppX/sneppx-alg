#!/usr/bin/env bash
# SNEPPX-Algo: EditorConfig checker
# Verifies all source files match .editorconfig rules
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

errors=0

check_indent() {
    local file="$1"
    if grep -Pn '^\t' "$file" >/dev/null 2>&1; then
        echo "TABS: $file"
        ((errors++))
    fi
}

check_trailing() {
    local file="$1"
    if grep -Pl '[[:space:]]$' "$file" >/dev/null 2>&1; then
        echo "TRAILING WHITESPACE: $file"
        ((errors++))
    fi
}

check_eol() {
    local file="$1"
    if [ -s "$file" ] && [ "$(tail -c 1 "$file" | xxd -p)" != "0a" ]; then
        echo "NO NEWLINE AT EOF: $file"
        ((errors++))
    fi
}

echo "=== EditorConfig Check ==="
echo ""

while IFS= read -r file; do
    check_indent "$file"
    check_trailing "$file"
    check_eol "$file"
done < <(find . \( -name '*.c' -o -name '*.h' -o -name '*.md' -o -name '*.yml' -o -name '*.yaml' -o -name '*.cmake' -o -name '*.sh' -o -name '*.py' \) \
    -not -path './build/*' -not -path './target/*' -not -path './releases/*' -not -path './.git/*' -not -path './node_modules/*')

if [ "$errors" -eq 0 ]; then
    echo "All files pass EditorConfig checks."
else
    echo "Found $errors file(s) with issues."
    exit 1
fi
