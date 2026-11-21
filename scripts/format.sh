#!/usr/bin/env bash
# Format all source files in-place
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== clang-format ==="
find . -name '*.c' -o -name '*.h' | grep -v build/ | grep -v target/ | grep -v releases/ | xargs clang-format -i -style=file

echo "=== cmake-format ==="
find . -name 'CMakeLists.txt' -o -name '*.cmake' | grep -v build/ | grep -v target/ | xargs cmake-format -i

echo "=== markdown ==="
find . -name '*.md' -not -path './build/*' -not -path './target/*' -not -path './releases/*' -not -path './.git/*' -exec sed -i 's/[[:space:]]*$//' {} \+

echo "=== yaml ==="
find . -name '*.yml' -o -name '*.yaml' | grep -v build/ | grep -v target/ | xargs sed -i 's/[[:space:]]*$//'

echo "Format complete."
