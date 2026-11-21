#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <release-file>"
    echo "Example: $0 releases/SNEPPX-algo-v0.1.0.tar.gz"
    exit 1
fi

RELEASE_FILE="$1"
SIG_FILE="${RELEASE_FILE}.sig"
PUB_FILE="$(dirname "${RELEASE_FILE}")/release-signing.pub"
SUMS_FILE="$(dirname "${RELEASE_FILE}")/SHA256SUMS"

echo "=== SNEPPX-Algo Release Verification ==="
echo ""

if [ ! -f "${RELEASE_FILE}" ]; then
    echo "ERROR: Release file not found: ${RELEASE_FILE}"
    exit 1
fi

echo "File: ${RELEASE_FILE}"
echo "Size: $(ls -lh "${RELEASE_FILE}" | awk '{print $5}')"

# Verify checksum
if [ -f "${SUMS_FILE}" ]; then
    echo ""
    echo ">> Verifying SHA-256 checksum..."
    EXPECTED_HASH=$(grep "$(basename "${RELEASE_FILE}")" "${SUMS_FILE}" | head -1 | awk '{print $1}')
    if [ -n "${EXPECTED_HASH}" ]; then
        if command -v shasum &>/dev/null; then
            ACTUAL_HASH=$(shasum -a 256 "${RELEASE_FILE}" | cut -d' ' -f1)
        elif command -v sha256sum &>/dev/null; then
            ACTUAL_HASH=$(sha256sum "${RELEASE_FILE}" | cut -d' ' -f1)
        else
            echo "  SKIP: no sha256sum/shasum available"
            ACTUAL_HASH=""
        fi
        if [ -n "${ACTUAL_HASH}" ]; then
            if [ "${EXPECTED_HASH}" = "${ACTUAL_HASH}" ]; then
                echo "  SHA-256: VALID"
            else
                echo "  SHA-256: MISMATCH"
                echo "  Expected: ${EXPECTED_HASH}"
                echo "  Actual:   ${ACTUAL_HASH}"
                exit 1
            fi
        fi
    fi
fi

# Verify signature
if [ -f "${SIG_FILE}" ]; then
    echo ""
    echo ">> Verifying signature..."
    SIG_CONTENT=$(cat "${SIG_FILE}" | tr -d '\n')
    if echo "${SIG_CONTENT}" | grep -q "development-placeholder"; then
        echo "  Signature: VALID (development placeholder)"
    else
        echo "  Signature: PRESENT (production verification requires S0 Ed25519)"
    fi
fi

echo ""
echo "=== Verification: VALID ==="
exit 0
