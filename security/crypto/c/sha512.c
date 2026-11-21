#include "sha512_hashing_implementation.h"
#include <string.h>

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define SIGMA1(x) (ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define sigma0(x) (ROTR64(x, 1) ^ ROTR64(x, 8) ^ ((x) >> 7))
#define sigma1(x) (ROTR64(x, 19) ^ ROTR64(x, 61) ^ ((x) >> 6))
/*
 * SNEPPX - SHA-512 Cryptographic Hash
 *
 * WHAT
 *   SHA-512 Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 180-4 SHA-512: 512-bit digest from arbitrary input (64-bit variant of SHA-256).
 *
 * ROLE
 *   Used by HMAC, HKDF, DRBG, and anywhere a 512-bit hash is needed.
 *
 * REFERENCES
 *   FIPS 180-4 (SHA-512).
 */



static const uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static void sha512_transform(SNEPPXSHA512Context* ctx, const uint8_t block[SNEPPX_SHA512_BLOCK_SIZE]) {
    uint64_t W[80];
    for (int t = 0; t < 16; t++) {
        W[t] = (uint64_t)block[t * 8] << 56 | (uint64_t)block[t * 8 + 1] << 48 |
               (uint64_t)block[t * 8 + 2] << 40 | (uint64_t)block[t * 8 + 3] << 32 |
               (uint64_t)block[t * 8 + 4] << 24 | (uint64_t)block[t * 8 + 5] << 16 |
               (uint64_t)block[t * 8 + 6] << 8  | (uint64_t)block[t * 8 + 7];
    }
    for (int t = 16; t < 80; t++)
        W[t] = sigma1(W[t - 2]) + W[t - 7] + sigma0(W[t - 15]) + W[t - 16];

    uint64_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint64_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    for (int t = 0; t < 80; t++) {
        uint64_t T1 = h + SIGMA1(e) + CH(e, f, g) + K[t] + W[t];
        uint64_t T2 = SIGMA0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

/**
 * @brief Initialize Sha512.
 */
void SNEPPX_sha512_init(SNEPPXSHA512Context* ctx) {
    ctx->state[0] = 0x6a09e667f3bcc908ULL;
    ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL;
    ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL;
    ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL;
    ctx->state[7] = 0x5be0cd19137e2179ULL;
    ctx->count[0] = ctx->count[1] = 0;
    ctx->buflen = 0;
}

/**
 * @brief Update Sha512.
 *
 * @param ctx [out] Ctx value.
 * @param data [in] Data value.
 */
void SNEPPX_sha512_update(SNEPPXSHA512Context* ctx, const uint8_t* data, size_t len) {
    uint64_t bitlen = (uint64_t)len * 8;
    ctx->count[1] += (uint64_t)((ctx->count[0] + bitlen) < ctx->count[0]);
    ctx->count[0] += bitlen;

    if (ctx->buflen) {
        size_t fill = SNEPPX_SHA512_BLOCK_SIZE - ctx->buflen;
        if (len < fill) { memcpy(ctx->buffer + ctx->buflen, data, len); ctx->buflen += (unsigned int)len; return; }
        memcpy(ctx->buffer + ctx->buflen, data, fill);
        sha512_transform(ctx, ctx->buffer);
        data += fill; len -= fill; ctx->buflen = 0;
    }
    while (len >= SNEPPX_SHA512_BLOCK_SIZE) {
        sha512_transform(ctx, data);
        data += SNEPPX_SHA512_BLOCK_SIZE; len -= SNEPPX_SHA512_BLOCK_SIZE;
    }
    if (len) { memcpy(ctx->buffer, data, len); ctx->buflen = (unsigned int)len; }
}

/**
 * @brief Perform Sha512 Finish.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_sha512_finish(SNEPPXSHA512Context* ctx, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]) {
    uint8_t pad[SNEPPX_SHA512_BLOCK_SIZE * 2];
    size_t padlen = 0;
    pad[padlen++] = 0x80;
    size_t remaining = SNEPPX_SHA512_BLOCK_SIZE - ((ctx->buflen + 1 + 16) % SNEPPX_SHA512_BLOCK_SIZE);
    if (remaining == SNEPPX_SHA512_BLOCK_SIZE) remaining = 0;
    while (remaining--) pad[padlen++] = 0;
    uint64_t bits = ctx->count[1];
    uint64_t bits_lo = ctx->count[0];
    for (int i = 0; i < 8; i++) pad[padlen++] = (uint8_t)(bits >> (56 - i * 8));
    for (int i = 0; i < 8; i++) pad[padlen++] = (uint8_t)(bits_lo >> (56 - i * 8));
    SNEPPX_sha512_update(ctx, pad, padlen);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            digest[i * 8 + j] = (uint8_t)(ctx->state[i] >> (56 - j * 8));
}

/**
 * @brief Perform Sha512 Finish (Legacy / Pre-Fix Behavior).
 *
 * Reproduces the original (pre-RFC-8032-compliance) length-encoding
 * defect for backward-compatible verification of manifests signed
 * before the fix.  count[] is maintained in bits in update(), but this
 * variant erroneously multiplies by 8 again (<< 3), yielding a length
 * field that is 8× too large.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_sha512_legacy_finish(SNEPPXSHA512Context* ctx, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]) {
    uint8_t pad[SNEPPX_SHA512_BLOCK_SIZE * 2];
    size_t padlen = 0;
    pad[padlen++] = 0x80;
    size_t remaining = SNEPPX_SHA512_BLOCK_SIZE - ((ctx->buflen + 1 + 16) % SNEPPX_SHA512_BLOCK_SIZE);
    if (remaining == SNEPPX_SHA512_BLOCK_SIZE) remaining = 0;
    while (remaining--) pad[padlen++] = 0;
    uint64_t bits = (ctx->count[1] << 3) | (ctx->count[0] >> 61);
    uint64_t bits_lo = ctx->count[0] << 3;
    for (int i = 0; i < 8; i++) pad[padlen++] = (uint8_t)(bits >> (56 - i * 8));
    for (int i = 0; i < 8; i++) pad[padlen++] = (uint8_t)(bits_lo >> (56 - i * 8));
    SNEPPX_sha512_update(ctx, pad, padlen);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            digest[i * 8 + j] = (uint8_t)(ctx->state[i] >> (56 - j * 8));
}

/**
 * @brief Perform Sha512.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_sha512(const uint8_t* data, size_t len, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]) {
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, data, len);
    SNEPPX_sha512_finish(&ctx, digest);
}

/**
 * @brief Legacy SHA-512 (pre-RFC-8032 length encoding).
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_sha512_legacy(const uint8_t* data, size_t len, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]) {
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, data, len);
    SNEPPX_sha512_legacy_finish(&ctx, digest);
}
