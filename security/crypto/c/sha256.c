#include "sha256.h"
#include <string.h>

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
/*
 * SNEPPX - SHA-256 Cryptographic Hash
 *
 * WHAT
 *   SHA-256 Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 180-4 SHA-256: 256-bit digest from arbitrary input via Merkle-Damgard compression.
 *
 * ROLE
 *   Used by HMAC, HKDF, DRBG, and as a general-purpose integrity hash.
 *
 * REFERENCES
 *   FIPS 180-4 (SHA-256).
 */



static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} sha256_ctx_t;

static void sha256_transform(sha256_ctx_t *ctx) {
    uint32_t W[64], a, b, c, d, e, f, g, h, T1, T2;
    for (int i = 0; i < 16; i++)
        W[i] = ((uint32_t)ctx->buffer[i*4]) << 24 |
               ((uint32_t)ctx->buffer[i*4+1]) << 16 |
               ((uint32_t)ctx->buffer[i*4+2]) << 8 |
               ((uint32_t)ctx->buffer[i*4+3]);
    for (int i = 16; i < 64; i++)
        W[i] = SIG1(W[i-2]) + W[i-7] + SIG0(W[i-15]) + W[i-16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        T1 = h + EP1(e) + CH(e,f,g) + K[i] + W[i];
        T2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

/**
 * @brief Perform Sha256.
 *
 * @param in [in] In value.
 */
void SNEPPX_sha256(uint8_t out[32], const uint8_t *in, size_t inlen) {
    sha256_ctx_t ctx;
    ctx.state[0] = 0x6a09e667; ctx.state[1] = 0xbb67ae85;
    ctx.state[2] = 0x3c6ef372; ctx.state[3] = 0xa54ff53a;
    ctx.state[4] = 0x510e527f; ctx.state[5] = 0x9b05688c;
    ctx.state[6] = 0x1f83d9ab; ctx.state[7] = 0x5be0cd19;
    ctx.count = 0;
    size_t pos = 0;
    while (pos + 64 <= inlen) {
        memcpy(ctx.buffer, in + pos, 64);
        sha256_transform(&ctx);
        pos += 64;
    }
    size_t remaining = inlen - pos;
    memcpy(ctx.buffer, in + pos, remaining);
    ctx.buffer[remaining] = 0x80;
    if (remaining >= 56) {
        memset(ctx.buffer + remaining + 1, 0, 64 - remaining - 1);
        sha256_transform(&ctx);
        memset(ctx.buffer, 0, 56);
    } else {
        memset(ctx.buffer + remaining + 1, 0, 56 - remaining - 1);
    }
    uint64_t bits = inlen * 8;
    ctx.buffer[56] = bits >> 56; ctx.buffer[57] = bits >> 48;
    ctx.buffer[58] = bits >> 40; ctx.buffer[59] = bits >> 32;
    ctx.buffer[60] = bits >> 24; ctx.buffer[61] = bits >> 16;
    ctx.buffer[62] = bits >> 8;  ctx.buffer[63] = bits;
    sha256_transform(&ctx);
    for (int i = 0; i < 8; i++) {
        out[i*4] = ctx.state[i] >> 24;
        out[i*4+1] = ctx.state[i] >> 16;
        out[i*4+2] = ctx.state[i] >> 8;
        out[i*4+3] = ctx.state[i];
    }
}
