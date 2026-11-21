#!/usr/bin/env bash
# Run all demo programs
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found. Run scripts/setup.sh first."
    exit 1
fi

cd "$BUILD_DIR"

echo "=== SNEPPX-Algo Demos ==="
echo ""

run_demo() {
    local name="$1"
    local binary="$2"
    if [ -f "$binary" ]; then
        echo "--- $name ---"
        "$binary"
        echo ""
    else
        echo "[SKIP] $name (not built: $binary)"
    fi
}

run_demo "HSS"          "examples/Release/hss_demo.exe"
run_demo "SER"          "examples/Release/ser_demo.exe"
run_demo "ARC"          "examples/Release/arc_demo.exe"
run_demo "NPE"          "examples/Release/npe_demo.exe"
run_demo "FM"           "examples/Release/fm_demo.exe"
run_demo "Obfuscation"  "examples/Release/obf_demo.exe"
run_demo "Security"     "examples/Release/security_stress.exe"

echo "=== All demos complete ==="
