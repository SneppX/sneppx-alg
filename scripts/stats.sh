#!/usr/bin/env bash
# Display project statistics
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== SNEPPX-Algo Stats ==="
echo ""

# Lines of code
echo "--- Lines of Code ---"
find . \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
    -not -path './build/*' -not -path './target/*' -not -path './releases/*' \
    -print0 | xargs -0 wc -l | tail -1

echo ""
echo "--- File Counts ---"
echo "C source files:  $(find . -name '*.c' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "C headers:       $(find . -name '*.h' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "C++ source:      $(find . -name '*.cpp' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "C++ headers:     $(find . -name '*.hpp' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "Python:          $(find . -name '*.py' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "Markdown:        $(find . -name '*.md' -not -path './build/*' -not -path './target/*' | wc -l)"
echo "CMake:           $(find . -name 'CMakeLists.txt' -o -name '*.cmake' | grep -v build/ | grep -v target/ | wc -l)"
echo "Shell scripts:   $(find . -name '*.sh' -not -path './build/*' -not -path './target/*' | wc -l)"

echo ""
echo "--- Test Counts ---"
if [ -d build/tests/Release ]; then
    total=0
    passed=0
    failed=0
    for exe in build/tests/Release/test_*.exe; do
        if [ -f "$exe" ]; then
            count=$("$exe" 2>/dev/null | grep -oP '\d+ passed' | awk '{s+=$1} END {print s}')
            total=$((total + count))
        fi
    done
    echo "Total passing tests: $total"
else
    echo "Build tests/ not found. Run 'cmake --build build' first."
fi
