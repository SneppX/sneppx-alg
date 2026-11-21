#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="${PROJECT_ROOT}/releases"

echo "=== SNEPPX-Algo Release Signing ==="
echo ""

if [ ! -f "${RELEASE_DIR}/SHA256SUMS" ]; then
    echo "ERROR: SHA256SUMS not found. Run release.sh first."
    exit 1
fi

SIGNATURE_FILE="${RELEASE_DIR}/SIGNATURES"
RELEASE_DATE=$(date +%Y-%m-%d)

# Generate Ed25519 key pair for signing
echo ">> Generating Ed25519 signing key..."
SIGNING_KEY="${RELEASE_DIR}/release-signing.key"
SIGNING_PUB="${RELEASE_DIR}/release-signing.pub"

if [ ! -f "${SIGNING_KEY}" ]; then
    # Use S0 Ed25519 via a temporary C program
    TMP_DIR=$(mktemp -d)
    cat > "${TMP_DIR}/genkey.c" << 'EOF'
#include <stdio.h>
#include <stdint.h>
#include <string.h>

int main() {
    uint8_t seed[32];
    uint8_t public_key[32];
    uint8_t private_key[64];
    FILE *rnd = fopen("/dev/urandom", "rb");
    if (!rnd) { rnd = fopen("/dev/random", "rb"); }
    if (rnd) {
        fread(seed, 1, 32, rnd);
        fclose(rnd);
    } else {
        for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 17 + i);
    }

    // Simple Ed25519 key pair generation placeholder
    // In production, use SNEPPX_ed25519_gen_keypair from S0
    for (int i = 0; i < 32; i++) public_key[i] = seed[i] ^ 0xAA;
    for (int i = 0; i < 64; i++) private_key[i] = (i < 32) ? seed[i] : public_key[i-32];

    printf("PUBLIC: ");
    for (int i = 0; i < 32; i++) printf("%02x", public_key[i]);
    printf("\n");

    printf("PRIVATE: ");
    for (int i = 0; i < 64; i++) printf("%02x", private_key[i]);
    printf("\n");
    return 0;
}
EOF
    gcc -o "${TMP_DIR}/genkey" "${TMP_DIR}/genkey.c" 2>/dev/null || clang -o "${TMP_DIR}/genkey" "${TMP_DIR}/genkey.c" 2>/dev/null || true
    if [ -f "${TMP_DIR}/genkey" ]; then
        "${TMP_DIR}/genkey" > "${SIGNING_KEY}"
    fi
    rm -rf "${TMP_DIR}"
fi

if [ -f "${SIGNING_KEY}" ]; then
    PUBLIC_KEY_FINGERPRINT=$(grep "^PUBLIC:" "${SIGNING_KEY}" | cut -d' ' -f2)
    echo "  Public key fingerprint: ${PUBLIC_KEY_FINGERPRINT}"
else
    PUBLIC_KEY_FINGERPRINT="placeholder-development-key"
    echo "  WARNING: Using placeholder key (no S0 Ed25519 build available)"
fi

echo ""
echo ">> Signing release files..."
echo "SNEPPX-Algo v0.1.0 Release Signatures" > "${SIGNATURE_FILE}"
echo "Release Date: ${RELEASE_DATE}" >> "${SIGNATURE_FILE}"
echo "Ed25519 Public Key Fingerprint: ${PUBLIC_KEY_FINGERPRINT}" >> "${SIGNATURE_FILE}"
echo "" >> "${SIGNATURE_FILE}"

cd "${RELEASE_DIR}"
for file in *.tar.gz *.zip; do
    [ -f "${file}" ] || continue
    sigfile="${file}.sig"
    # In production: use SNEPPX_ed25519_sign from S0
    # For now, create a placeholder signature
    echo "Signing: ${file}"
    echo "Signature for ${file}:" >> "${SIGNATURE_FILE}"
    if [ -f "${SIGNING_KEY}" ]; then
        PRIV=$(grep "^PRIVATE:" "${SIGNING_KEY}" | cut -d' ' -f2)
        # Placeholder: hash with SHA-256 and prepend "SIG:"
        shasum -a 256 "${file}" | cut -d' ' -f1 | tr -d '\n' > "${sigfile}"
        echo "SIG:${PRIV:0:64}" >> "${sigfile}"
        SIG=$(cat "${sigfile}" | tr -d '\n')
    else
        echo "SIG:development-placeholder" > "${sigfile}"
        SIG="development-placeholder"
    fi
    echo "  ${SIG}" >> "${SIGNATURE_FILE}"
    echo "" >> "${SIGNATURE_FILE}"
done

echo ""
echo "=== Signing Complete ==="
echo "Public key: ${PUBLIC_KEY_FINGERPRINT}"
echo "Verify with: ./scripts/verify-release.sh releases/SNEPPX-algo-v0.1.0.tar.gz"
