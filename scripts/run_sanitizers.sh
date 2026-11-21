#!/bin/bash
# SNEPPX Compute Sanitizer CI Wrapper
# Usage: ./scripts/run_sanitizers.sh [target] [extra_args]

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
TARGET="${1:-}"
EXTRA_ARGS="${@:2}"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "=== SNEPPX Sanitizer CI ==="

# 1. Build with ASan
echo ""
echo "--- Building with AddressSanitizer ---"
cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_USE_ASAN=ON
cmake --build "$BUILD_DIR" --config Debug -j$(nproc)

# 2. Build with UBSan
echo ""
echo "--- Building with UndefinedBehaviorSanitizer ---"
cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_USE_UBSAN=ON
cmake --build "$BUILD_DIR" --config Debug -j$(nproc)

# 3. Run CUDA compute-sanitizer (memcheck) on tests
if command -v compute-sanitizer &> /dev/null; then
    echo ""
    echo "--- Running compute-sanitizer (memcheck) ---"
    for test_bin in "$BUILD_DIR"/tests/*/test_*; do
        if [ -x "$test_bin" ]; then
            echo "Testing: $test_bin"
            compute-sanitizer --tool memcheck --report-api-errors all \
                "$test_bin" $EXTRA_ARGS 2>&1 | tail -20
        fi
    done
else
    echo "WARNING: compute-sanitizer not found. Skipping CUDA memcheck."
fi

# 4. Run compute-sanitizer racecheck (if CUDA)
if command -v compute-sanitizer &> /dev/null; then
    echo ""
    echo "--- Running compute-sanitizer (racecheck) ---"
    for test_bin in "$BUILD_DIR"/tests/*/test_*; do
        if [ -x "$test_bin" ]; then
            compute-sanitizer --tool racecheck "$test_bin" $EXTRA_ARGS 2>&1 | tail -10
        fi
    done
fi

# 5. Run compute-sanitizer initcheck
if command -v compute-sanitizer &> /dev/null; then
    echo ""
    echo "--- Running compute-sanitizer (initcheck) ---"
    for test_bin in "$BUILD_DIR"/tests/*/test_*; do
        if [ -x "$test_bin" ]; then
            compute-sanitizer --tool initcheck "$test_bin" $EXTRA_ARGS 2>&1 | tail -10
        fi
    done
fi

# 6. Run ctest
echo ""
echo "--- Running ctest ---"
cd "$BUILD_DIR"
ctest -C Debug --output-on-failure $EXTRA_ARGS
cd ..

echo ""
echo -e "${GREEN}=== Sanitizer CI Complete ===${NC}"
