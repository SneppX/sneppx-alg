#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tarball-or-zip>" >&2
    exit 1
fi

PACKAGE="$1"
RELEASE_DIR="$(dirname "$PACKAGE")"
BASENAME="$(basename "$PACKAGE")"
FAILED=0

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }

check() {
    local name="$1"
    shift
    if "$@"; then
        green "  PASS  $name"
    else
        red   "  FAIL  $name"
        FAILED=1
    fi
}

# 1. File exists
check "Package exists" test -f "$PACKAGE"

# 2. SHA256 checksum
if [ -f "${RELEASE_DIR}/SHA256SUMS" ]; then
    check "SHA256 checksum" sh -c "
        cd "$RELEASE_DIR" && sha256sum -c SHA256SUMS --ignore-missing 2>/dev/null | grep -q '${BASENAME}: OK'
    "
else
    echo "  SKIP  SHA256 checksum (SHA256SUMS not found)"
fi

# 3. SHA512 checksum
if [ -f "${RELEASE_DIR}/SHA512SUMS" ]; then
    check "SHA512 checksum" sh -c "
        cd "$RELEASE_DIR" && sha512sum -c SHA512SUMS --ignore-missing 2>/dev/null | grep -q '${BASENAME}: OK'
    "
else
    echo "  SKIP  SHA512 checksum (SHA512SUMS not found)"
fi

# 4. Ed25519 signature
SIGFILE="${PACKAGE}.sig"
if [ -f "$SIGFILE" ]; then
    if command -v SNEPPX-verify &>/dev/null; then
        check "Ed25519 signature" sh -c "SNEPPX-verify \"$PACKAGE\" \"$SIGFILE\""
    else
        echo "  SKIP  Ed25519 signature (SNEPPX-verify not installed)"
    fi
else
    echo "  SKIP  Ed25519 signature (${BASENAME}.sig not found)"
fi

# 5. Extract and check for unexpected binaries
EXTRACT_DIR="$(mktemp -d)"
trap 'rm -rf "$EXTRACT_DIR"' EXIT

case "$BASENAME" in
    *.tar.gz) tar xzf "$PACKAGE" -C "$EXTRACT_DIR" ;;
    *.zip)    unzip -q "$PACKAGE" -d "$EXTRACT_DIR" ;;
esac

check "No unexpected binaries" sh -c "
    find "$EXTRACT_DIR" -type f \( -name '*.exe' -o -name '*.dll' -o -name '*.so' \
        -o -name '*.dylib' -o -name '*.o' -o -name '*.a' \) | grep -q . && exit 1 || exit 0
"

if [ "$FAILED" -eq 1 ]; then
    red "\nPackage verification FAILED."
    exit 1
fi

green "\nPackage verification PASSED."
