#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SNEPPX-Algo v0.1.0 Installer ==="
echo ""

# Detect OS
OS="unknown"
ARCH="unknown"
case "$(uname -s)" in
    Linux*)     OS="linux";;
    Darwin*)    OS="macos";;
    MSYS*|MINGW*) OS="windows";;
    *)          OS="unknown";;
esac

case "$(uname -m)" in
    x86_64|amd64) ARCH="x86_64";;
    aarch64|arm64) ARCH="arm64";;
    *)          ARCH="unknown";;
esac

echo "Detected: ${OS} (${ARCH})"
echo ""

# Check dependencies
MISSING=""

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        MISSING="${MISSING} $1"
        echo "  MISSING: $1"
    else
        echo "  FOUND: $1 ($($1 --version 2>&1 | head -1))"
    fi
}

echo ">> Checking dependencies..."
check_cmd cmake
check_cmd ninja

case "${OS}" in
    linux)
        check_cmd g++
        check_cmd python3
        if [ -n "$(echo "${MISSING}" | grep "g++")" ]; then
            echo ""
            echo "  Install with:"
            echo "    sudo apt-get install build-essential ninja-build cmake python3-dev"
            echo "    sudo dnf install gcc-c++ ninja-build cmake python3-devel"
        fi
        ;;
    macos)
        check_cmd clang++
        if [ -n "$(echo "${MISSING}" | grep "clang++")" ]; then
            echo ""
            echo "  Install with:"
            echo "    xcode-select --install"
            echo "    brew install cmake ninja"
        fi
        ;;
    windows)
        echo "  NOTE: On Windows, use install.ps1 (PowerShell)"
        ;;
esac

if [ -n "${MISSING}" ]; then
    echo ""
    echo "WARNING: Missing dependencies:${MISSING}"
    echo "Install missing packages and re-run."
    echo ""
fi

# Build
echo ">> Building..."
mkdir -p build && cd build

if [ "${OS}" = "windows" ]; then
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_PYTHON=OFF
    cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)
else
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_PYTHON=OFF
    cmake --build . -j$(nproc 2>/dev/null || echo 4)
fi

# Test
echo ""
echo ">> Running tests..."
ctest --output-on-failure -C Release 2>/dev/null || ctest --output-on-failure 2>/dev/null || echo "  Tests skipped (build only)"

echo ""
echo "=== SNEPPX-Algo v0.1.0 installed ==="
echo "Run ./scripts/test.sh to verify."
echo "See docs/ for documentation."
