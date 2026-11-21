#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"
cmake "$ROOT_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSNEPPX_BUILD_TESTS=ON -DSNEPPX_BUILD_BENCHMARKS=ON

if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    cp "${BUILD_DIR}/compile_commands.json" "${ROOT_DIR}/compile_commands.json"
    echo "compile_commands.json copied to project root."
else
    echo "Error: compile_commands.json not generated." >&2
    exit 1
fi
