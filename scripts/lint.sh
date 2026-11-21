#!/usr/bin/env bash
# Lint all source files
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== clang-format ==="
find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/ | grep -v releases/ | xargs clang-format -i -style=file

echo "=== markdownlint ==="
find . -name '*.md' -not -path './build/*' -not -path './target/*' -not -path './releases/*' -not -path './node_modules/*' -not -path './.git/*' -exec markdownlint -c .markdownlint.yaml {} \+

echo "=== shellcheck ==="
find . -name '*.sh' -not -path './build/*' -not -path './target/*' -not -path './releases/*' | xargs shellcheck -s bash

echo "=== yamllint ==="
find . -name '*.yml' -o -name '*.yaml' | grep -v build/ | grep -v target/ | grep -v releases/ | xargs yamllint -c .yamllint.yml

echo "=== cmake-format ==="
find . -name 'CMakeLists.txt' -o -name '*.cmake' | grep -v build/ | grep -v target/ | xargs cmake-format -i

echo "All linters passed."
