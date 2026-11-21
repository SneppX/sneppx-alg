#!/usr/bin/env bash
# Set up a fresh development environment
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== SNEPPX-Algo Development Setup ==="
echo ""

# Check prerequisites
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found. Install CMake >= 3.16."; exit 1; }
echo "[OK] cmake $(cmake --version | head -1 | awk '{print $3}')"

if command -v python3 >/dev/null 2>&1; then
    echo "[OK] python3 $(python3 --version | awk '{print $2}')"
else
    echo "[WARN] python3 not found (optional, needed for bindings)"
fi

# Configure and build
echo ""
echo "=== Configuring (Debug) ==="
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_BENCHMARKS=ON

echo ""
echo "=== Building ==="
CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
cmake --build build -j"$CORES"

echo ""
echo "=== Testing ==="
cd build && ctest --output-on-failure --timeout 60

echo ""
echo "=== Setup Complete ==="
echo "Build directory: $ROOT/build"
echo "Run tests:      cd $ROOT/build && ctest"
echo "Run benchmarks: $ROOT/build/tests/benchmark/bench_tensor"
