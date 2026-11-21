#!/usr/bin/env bash
# Generate code coverage reports
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build-coverage"
REPORT_DIR="${ROOT}/coverage_report"

# Prerequisites
command -v lcov >/dev/null 2>&1 || { echo "ERROR: lcov not found"; exit 1; }
command -v genhtml >/dev/null 2>&1 || { echo "ERROR: genhtml not found"; exit 1; }

echo "=== SNEPPX-Algo Coverage Report ==="

# Configure with coverage flags
echo "Configuring..."
cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_BUILD_TESTS=ON \
    -DCMAKE_C_FLAGS="--coverage -g -O0" \
    -DCMAKE_EXE_LINKER_FLAGS="--coverage"

# Build
echo "Building..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Run tests
echo "Running tests..."
cd "$BUILD_DIR"
ctest --output-on-failure --timeout 60

# Capture coverage
echo "Capturing coverage..."
lcov --directory . --capture --output-file coverage_raw.info
lcov --remove coverage_raw.info \
    '/usr/*' \
    "${ROOT}/tests/*" \
    "${ROOT}/build/*" \
    "${ROOT}/target/*" \
    --output-file coverage_filtered.info

# Generate HTML report
echo "Generating HTML report at ${REPORT_DIR}..."
genhtml coverage_filtered.info --output-directory "$REPORT_DIR"

echo ""
echo "=== Done ==="
echo "Report: file://${REPORT_DIR}/index.html"
