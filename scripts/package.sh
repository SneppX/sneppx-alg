#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="${ROOT_DIR}/VERSION"

# Read version from VERSION file or argument
if [ $# -ge 1 ]; then
    VERSION="$1"
else
    VERSION="$(head -1 "$VERSION_FILE" | sed 's/.*v//;s/ .*//')"
fi

RELEASE_DIR="${ROOT_DIR}/releases/SNEPPX-algo-v${VERSION}"
TARBALL="${ROOT_DIR}/releases/SNEPPX-algo-v${VERSION}.tar.gz"
ZIPFILE="${ROOT_DIR}/releases/SNEPPX-algo-v${VERSION}.zip"

mkdir -p "$RELEASE_DIR"

# Copy source tree excluding build artifacts and git files
rsync -a --exclude='.git/' --exclude='build/' --exclude='build_standalone/' \
    --exclude='releases/' --exclude='*.tar.gz' --exclude='*.zip' --exclude='*.sig' \
    --exclude='SUMS' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='node_modules/' --exclude='.pytest_cache/' --exclude='.mypy_cache/' \
    --exclude='.vscode/' --exclude='.idea/' \
    "$ROOT_DIR"/ "$RELEASE_DIR/"

# Create tarball
(cd "$ROOT_DIR/releases" && tar czf "$TARBALL" "SNEPPX-algo-v${VERSION}/")

# Create zip
(cd "$ROOT_DIR/releases" && zip -rq "$ZIPFILE" "SNEPPX-algo-v${VERSION}/")

# Generate checksums
cd "$ROOT_DIR/releases"
sha256sum "SNEPPX-algo-v${VERSION}.tar.gz" "SNEPPX-algo-v${VERSION}.zip" > SHA256SUMS
sha512sum "SNEPPX-algo-v${VERSION}.tar.gz" "SNEPPX-algo-v${VERSION}.zip" > SHA512SUMS

# Sign if signing script exists
if [ -x "${ROOT_DIR}/scripts/sign-release.sh" ]; then
    "${ROOT_DIR}/scripts/sign-release.sh" "$TARBALL"
    "${ROOT_DIR}/scripts/sign-release.sh" "$ZIPFILE"
fi

echo "Package created:"
echo "  $TARBALL"
echo "  $ZIPFILE"
echo "  SHA256SUMS, SHA512SUMS"
