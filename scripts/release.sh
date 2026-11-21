#!/usr/bin/env bash
set -e

VERSION="0.1.0"
RELEASE_NAME="SNEPPX-algo-${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="${PROJECT_ROOT}/releases"

echo "=== SNEPPX-Algo Release v${VERSION} ==="
echo ""

# Clean and rebuild
echo ">> Cleaning..."
"${SCRIPT_DIR}/clean.sh" 2>/dev/null || true

echo ">> Building..."
"${SCRIPT_DIR}/build.sh"

echo ">> Testing..."
"${SCRIPT_DIR}/test.sh"

# Create release directory
mkdir -p "${RELEASE_DIR}"

# Create source tarball
echo ">> Creating source tarball..."
cd "${PROJECT_ROOT}"
tar -czf "${RELEASE_DIR}/${RELEASE_NAME}.tar.gz" \
    --exclude='.git' \
    --exclude='releases' \
    --exclude='build' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    -C "${PROJECT_ROOT}" .

# Create zip archive
echo ">> Creating zip archive..."
if command -v zip &>/dev/null; then
    zip -r "${RELEASE_DIR}/${RELEASE_NAME}.zip" . \
        -x '.git/*' \
        -x 'releases/*' \
        -x 'build/*' \
        -x 'node_modules/*' \
        -x '__pycache__/*'
else
    echo "  zip not found, skipping zip archive"
fi

# Generate checksums
echo ">> Generating checksums..."
cd "${RELEASE_DIR}"
if command -v sha256sum &>/dev/null; then
    sha256sum ${RELEASE_NAME}.* > SHA256SUMS 2>/dev/null || true
elif command -v shasum &>/dev/null; then
    shasum -a 256 ${RELEASE_NAME}.* > SHA256SUMS 2>/dev/null || true
fi

if command -v sha512sum &>/dev/null; then
    sha512sum ${RELEASE_NAME}.* > SHA512SUMS 2>/dev/null || true
elif command -v shasum &>/dev/null; then
    shasum -a 512 ${RELEASE_NAME}.* > SHA512SUMS 2>/dev/null || true
fi

if command -v b3sum &>/dev/null; then
    b3sum ${RELEASE_NAME}.* > BLAKE3SUMS 2>/dev/null || true
fi

# Print results
echo ""
echo "=== Release v${VERSION} Complete ==="
echo ""
echo "Files:"
ls -lh "${RELEASE_DIR}/" 2>/dev/null | grep -v "^total"
echo ""
echo "Release directory: ${RELEASE_DIR}"
echo "Push tag: git push origin v${VERSION}"
