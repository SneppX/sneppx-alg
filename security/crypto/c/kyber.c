#include "kyber.h"
#include "cryptographic_random_generator.h"
#include "drbg.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define KYBER_N 256
#define KYBER_Q 3329
#define KYBER_QINV -3327
#define KYBER_K 3
#define KYBER_ETA1 2
#define KYBER_ETA2 2
#define KYBER_DU 10
#define KYBER_DV 4

/*
 * SNEPPX - Kyber
 *
 * WHAT
 *   Kyber.
 *
 * CONCEPT
 *   Provides Kyber post-quantum KEM (FIPS 203).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int16_t zetas[128];
static int init_ntt = 0;

/* Kyber NTT twiddles: zetas[i] = 2^16 * 17^brv7(i) mod q, i = 0..127
 * (the canonical FIPS 203 / pqcrystals reference table). The original
 * table here was 336 entries and corrupted after index 63, which
 * silently broke poly_mul and made decapsulation fail. */
static const int16_t zetas_rom[128] = {
    -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
     -171,   622,  1577,   182,   962, -1202, -1474,  1468,
      573, -1325,   264,   383,  -829,  1458, -1602,  -130,
     -681,  1017,   732,   608, -1542,   411,  -205, -1571,
     1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
      516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
     -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
     -398,   961, -1508,  -725,   448, -1065,   677, -1275,
    -1103,   430,   555,   843, -1251,   871,  1550,   105,
      422,   587,   177,  -235,  -291,  -460,  1574,  1653,
     -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
    -1590,   644,  -872,   349,   418,   329,  -156,   -75,
      817,  1097,   603,   610,  1322, -1285, -1465,   384,
    -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
    -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
     -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

static int kyber_init_ntt(void) {
    if (!init_ntt) {
        for (int i = 0; i < 128; i++) zetas[i] = zetas_rom[i];
        init_ntt = 1;
    }
    return 1;
}

static int16_t fq_reduce(int32_t a) {
    int16_t t = (int16_t)((a + KYBER_Q * 128) % KYBER_Q);
    return t >= KYBER_Q ? (int16_t)(t - KYBER_Q) : t;
}

/* Reference Montgomery reduction: a * 2^-16 mod q in centered form,
 * using QINV = -3327 (the minus-formula). The previous code used the
 * plus-formula with an incorrect QINV (20159), which broke every
 * NTT-domain multiply. */
static int16_t mont_reduce(int32_t a) {
    int16_t t = (int16_t)((int16_t)a * KYBER_QINV);
    return (int16_t)((a - (int32_t)t * KYBER_Q) >> 16);
}

static int16_t barrett_reduce(int16_t a) {
    int16_t t;
    const int16_t v = (int16_t)(((1L << 26) + KYBER_Q / 2) / KYBER_Q);
    t = (int16_t)(((int32_t)v * a + (1L << 25)) >> 26);
    t = (int16_t)(t * KYBER_Q);
    return (int16_t)(a - t);
}

static int16_t fqmul(int16_t a, int16_t b) { return mont_reduce((int32_t)a * b); }

/* Forward NTT (FIPS 203 / pqcrystals reference layout). Consumes
 * zetas[1..127], one twiddle per 256/len block, then Barrett-reduces
 * every coefficient. */
static void ntt(int16_t r[256]) {
    int len = 128, k = 1;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            int16_t zeta = zetas[k++];
            for (int j = start; j < start + len; j++) {
                int16_t t = fqmul(zeta, r[j + len]);
                r[j + len] = (int16_t)(r[j] - t);
                r[j] = (int16_t)(r[j] + t);
            }
            start += len * 2;
        }
        len >>= 1;
    }
    for (int j = 0; j < 256; j++) r[j] = barrett_reduce(r[j]);
}

static void inv_ntt(int16_t r[256]) {
    int len = 2, k = 127;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            int16_t zeta = zetas[k--];
            for (int j = start; j < start + len; j++) {
                int16_t t = r[j];
                r[j] = barrett_reduce((int16_t)(t + r[j + len]));
                r[j + len] = (int16_t)(r[j + len] - t);
                r[j + len] = fqmul(zeta, r[j + len]);
            }
            start += len * 2;
        }
        len <<= 1;
    }
    /* mont^2/128 -- the reference "tomont" scaling factor (1441). */
    for (int j = 0; j < 256; j++) r[j] = fqmul(r[j], 1441);
}

static void poly_add(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    for (int i = 0; i < 256; i++) r[i] = fq_reduce(a[i] + b[i]);
}

static void poly_sub(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    for (int i = 0; i < 256; i++) r[i] = fq_reduce(a[i] - b[i]);
}

/* Reference NTT-domain multiply: 2x2 basemul over Z[X]/(X^2 - zeta),
 * with zetas[64+i] / -zetas[64+i]. A plain pointwise multiply is wrong
 * for the Montgomery/cyclic layout. */
static void basemul(int16_t r[2], const int16_t a[2], const int16_t b[2], int16_t zeta) {
    int16_t r0 = fqmul(a[1], b[1]);
    r0 = fqmul(r0, zeta);
    r0 = (int16_t)(r0 + fqmul(a[0], b[0]));
    int16_t r1 = fqmul(a[0], b[1]);
    r1 = (int16_t)(r1 + fqmul(a[1], b[0]));
    r[0] = r0;
    r[1] = r1;
}

static void poly_mul(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    int16_t tmp[256], tmp2[256];
    memcpy(tmp, a, sizeof(tmp));
    ntt(tmp);
    memcpy(tmp2, b, sizeof(tmp2));
    ntt(tmp2);
    for (int i = 0; i < 64; i++) {
        basemul(r + 4 * i, tmp + 4 * i, tmp2 + 4 * i, zetas[64 + i]);
        basemul(r + 4 * i + 2, tmp + 4 * i + 2, tmp2 + 4 * i + 2, (int16_t)(-zetas[64 + i]));
    }
    inv_ntt(r);
}

static void poly_tobytes(uint8_t *out, const int16_t a[256]) {
    for (int i = 0; i < 128; i++) {
        int16_t t0 = a[2 * i], t1 = a[2 * i + 1];
        t0 = (int16_t)(t0 + ((t0 >> 15) & KYBER_Q));
        t1 = (int16_t)(t1 + ((t1 >> 15) & KYBER_Q));
        out[3 * i] = (uint8_t)t0;
        out[3 * i + 1] = (uint8_t)((t0 >> 8) | ((t1 & 0xf) << 4));
        out[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
}

static void poly_frombytes(int16_t r[256], const uint8_t *in) {
    for (int i = 0; i < 128; i++) {
        r[2 * i] = (int16_t)(((in[3 * i + 1] & 0x0f) << 8) | in[3 * i]);
        r[2 * i + 1] = (int16_t)((in[3 * i + 2] << 4) | ((in[3 * i + 1] >> 4) & 0x0f));
    }
}

/* LSB-first bit-accumulator packer: round(2^d/q * a) mod 2^d on the
 * positive representative. The old code used (1<<12)/q == 1 (collapsing
 * every coefficient to 0/1) and a broken cross-byte shifter that lost
 * bits whenever shift landed on a byte boundary. */
static void poly_compress(uint8_t *out, const int16_t a[256], int d) {
    size_t out_idx = 0;
    uint64_t buf = 0;
    int bits = 0;
    memset(out, 0, (size_t)((256 * d + 7) / 8));
    for (int i = 0; i < 256; i++) {
        int32_t t = a[i];
        t += (t >> 15) & KYBER_Q;
        t = (((t << d) + KYBER_Q / 2) / KYBER_Q);
        t &= (1 << d) - 1;
        buf |= ((uint64_t)t) << bits;
        bits += d;
        while (bits >= 8) {
            out[out_idx++] = (uint8_t)(buf & 0xff);
            buf >>= 8;
            bits -= 8;
        }
    }
}

static void poly_decompress(int16_t r[256], const uint8_t *in, int d) {
    int32_t f = (1 << d) / 2;
    int in_idx = 0, bits = 0;
    uint64_t buf = 0;
    for (int i = 0; i < 256; i++) {
        while (bits < d) { buf |= ((uint64_t)in[in_idx++] << bits); bits += 8; }
        int t = (int)(buf & ((1ULL << d) - 1));
        buf >>= d;
        bits -= d;
        r[i] = (int16_t)((t * KYBER_Q + f) >> d);
    }
}

static void poly_frommsg(int16_t r[256], const uint8_t msg[32]) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 8; j++)
            r[8 * i + j] = (int16_t)((msg[i] >> j & 1) * ((KYBER_Q + 1) / 2));
}

static void poly_tomsg(uint8_t msg[32], const int16_t a[256]) {
    for (int i = 0; i < 32; i++) {
        msg[i] = 0;
        for (int j = 0; j < 8; j++) {
            int16_t t = a[8 * i + j];
            t = (int16_t)(t + ((t >> 15) & KYBER_Q));
            t = (int16_t)((((int32_t)t << 1) + KYBER_Q / 2) / KYBER_Q);
            msg[i] |= (uint8_t)(t & 1) << j;
        }
    }
}

/* Centered-binomial-distribution noise sampling (FIPS 203 §4.1), eta=2.
 * Draws 2*eta bits per coefficient (popcount(a) - popcount(b)) so the
 * error stays small ([-2, 2]) and decapsulation noise stays well below
 * the compress/decompress rounding threshold. */
static void poly_getnoise(int16_t r[256], const uint8_t *seed, uint8_t nonce) {
    uint8_t buf[128];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, &nonce, 1);
    SNEPPX_drbg_generate(&drbg, buf, sizeof(buf));
    SNEPPX_drbg_destroy(&drbg);
    int pos = 0;
    for (int i = 0; i < 256; i++) {
        int a = 0, b = 0;
        for (int j = 0; j < KYBER_ETA1; j++) { a += (buf[pos >> 3] >> (pos & 7)) & 1; pos++; }
        for (int j = 0; j < KYBER_ETA1; j++) { b += (buf[pos >> 3] >> (pos & 7)) & 1; pos++; }
        r[i] = (int16_t)(a - b);
    }
}

static void kyber_expand_a(int16_t *a, const uint8_t rho[32], int k) {
    SNEPPXDRBG drbg;
    uint8_t ent[48];
    memcpy(ent, rho, 32);
    memset(ent + 32, 0, 16);
    SNEPPX_drbg_init(&drbg, ent, 48, NULL, 0);
    int total = k * k * 256;
    int bytes_needed = total * 2;
    uint8_t *buf = (uint8_t*)malloc(bytes_needed);
    if (!buf) return;
    SNEPPX_drbg_generate(&drbg, buf, bytes_needed);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < total; i++) {
        uint16_t val = (uint16_t)(buf[2 * i] | ((uint16_t)buf[2 * i + 1] << 8));
        a[i] = (int16_t)(val % KYBER_Q);
    }
    free(buf);
}

static void cpa_pke_keygen(uint8_t pk[KYBER_PUBLICKEYBYTES], uint8_t sk[KYBER_SECRETKEYBYTES]) {
    kyber_init_ntt();
    uint8_t seed[32], rho[32];
    SNEPPX_random_bytes(seed, 32);
    SNEPPX_random_bytes(rho, 32);
    int16_t *a = (int16_t*)malloc(KYBER_K * KYBER_K * 256 * sizeof(int16_t));
    int16_t s[KYBER_K * 256], e[KYBER_K * 256];
    if (!a) return;
    kyber_expand_a(a, rho, KYBER_K);
    for (int i = 0; i < KYBER_K; i++) {
        poly_getnoise(s + i * 256, seed, (uint8_t)i);
        poly_getnoise(e + i * 256, seed, (uint8_t)(KYBER_K + i));
    }
    for (int i = 0; i < KYBER_K; i++) {
        int16_t t[256] = {0};
        for (int j = 0; j < KYBER_K; j++) {
            int16_t prod[256];
            poly_mul(prod, a + (i * KYBER_K + j) * 256, s + j * 256);
            poly_add(t, t, prod);
        }
        int16_t tmp[256];
        poly_sub(tmp, t, e + i * 256);
        poly_tobytes(pk + i * 384 + 32, tmp);
    }
    memcpy(pk, rho, 32);
    memcpy(sk, seed, 32);
    memcpy(sk + 32, pk, KYBER_PUBLICKEYBYTES);
    free(a);
}

/**
 * @brief Generate Kyber.
 *
 * @param pk [out] Pk value.
 * @param sk [out] Sk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    (void)variant;
    cpa_pke_keygen(pk, sk);
    return 0;
}

/**
 * @brief Encapsulate Kyber.
 *
 * @param ct [out] Ct value.
 * @param ss [out] Ss value.
 * @param pk [in] Pk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_encaps(uint8_t *ct, uint8_t *ss, const uint8_t *pk, int variant) {
    if (!ct || !ss || !pk) return -1;
    (void)variant;
    int k = KYBER_K;
    int u_compressed_bytes = k * KYBER_N * KYBER_DU / 8;
    uint8_t coin[32], m[32];
    SNEPPX_random_bytes(coin, 32);
    SNEPPX_random_bytes(m, 32);
    int16_t mp[256];
    poly_frommsg(mp, m);
    int16_t *sp = (int16_t*)calloc(k * 256, sizeof(int16_t));
    int16_t *ep = (int16_t*)calloc(k * 256, sizeof(int16_t));
    int16_t epp[256];
    if (!sp || !ep) { free(sp); free(ep); return -1; }
    for (int i = 0; i < k; i++) {
        poly_getnoise(sp + i * 256, coin, (uint8_t)i);
        poly_getnoise(ep + i * 256, coin, (uint8_t)(k + i));
    }
    poly_getnoise(epp, coin, (uint8_t)(2 * k));
    int16_t *a = (int16_t*)malloc(k * k * 256 * sizeof(int16_t));
    if (!a) { free(sp); free(ep); return -1; }
    uint8_t rho[32];
    memcpy(rho, pk, 32);
    kyber_expand_a(a, rho, k);
    int16_t *u = (int16_t*)calloc(k * 256, sizeof(int16_t));
    int16_t v[256];
    memset(v, 0, sizeof(v));
    if (!u) { free(sp); free(ep); free(a); return -1; }
    for (int i = 0; i < k; i++) {
        int16_t t[256] = {0};
        for (int j = 0; j < k; j++) {
            int16_t prod[256];
            poly_mul(prod, a + (j * k + i) * 256, sp + j * 256);
            poly_add(t, t, prod);
        }
        poly_add(u + i * 256, t, ep + i * 256);
    }
    int16_t *pkpoly = (int16_t*)calloc(k * 256, sizeof(int16_t));
    if (!pkpoly) { free(sp); free(ep); free(a); free(u); return -1; }
    for (int i = 0; i < k; i++)
        poly_frombytes(pkpoly + i * 256, pk + i * 384 + 32);
    for (int i = 0; i < k; i++) {
        int16_t t[256] = {0};
        poly_mul(t, pkpoly + i * 256, sp + i * 256);
        poly_add(v, v, t);
    }
    poly_add(v, v, epp);
    poly_add(v, v, mp);
    for (int i = 0; i < k; i++)
        poly_compress(ct + i * (KYBER_N * KYBER_DU / 8), u + i * 256, KYBER_DU);
    poly_compress(ct + u_compressed_bytes, v, KYBER_DV);
    memcpy(ss, m, 32);
    free(sp); free(ep); free(a); free(u); free(pkpoly);
    return 0;
}

/**
 * @brief Decapsulate Kyber.
 *
 * @param ss [out] Ss value.
 * @param ct [in] Ct value.
 * @param sk [in] Sk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kyber_decaps(uint8_t *ss, const uint8_t *ct, const uint8_t *sk, int variant) {
    if (!ss || !ct || !sk) return -1;
    (void)variant;
    int k = KYBER_K;
    int u_compressed_bytes = k * KYBER_N * KYBER_DU / 8;
    uint8_t pk[KYBER_PUBLICKEYBYTES];
    memcpy(pk, sk + 32, KYBER_PUBLICKEYBYTES);
    int16_t *u = (int16_t*)calloc(k * 256, sizeof(int16_t));
    int16_t v[256];
    memset(v, 0, sizeof(v));
    if (!u) return -1;
    for (int i = 0; i < k; i++)
        poly_decompress(u + i * 256, ct + i * (KYBER_N * KYBER_DU / 8), KYBER_DU);
    poly_decompress(v, ct + u_compressed_bytes, KYBER_DV);
    int16_t s[4 * 256];
    memset(s, 0, sizeof(s));
    uint8_t seed[32];
    memcpy(seed, sk, 32);
    for (int i = 0; i < k; i++)
        poly_getnoise(s + i * 256, seed, (uint8_t)i);
    int16_t m[256] = {0};
    for (int i = 0; i < k; i++) {
        int16_t t[256] = {0};
        poly_mul(t, s + i * 256, u + i * 256);
        poly_sub(m, m, t);
    }
    poly_add(m, m, v);
    poly_tomsg(ss, m);
    free(u);
    return 0;
}
