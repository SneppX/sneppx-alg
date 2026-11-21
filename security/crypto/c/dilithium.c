#include "dilithium.h"
#include "cryptographic_random_generator.h"
#include "drbg.h"
#include "sha512_hashing_implementation.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>

#define DILITHIUM_N 256
#define DILITHIUM_Q 8380417
#define DILITHIUM_ROOT 1753
/*
 * SNEPPX - Dilithium ML-DSA (Post-Quantum Digital Signatures)
 *
 * WHAT
 *   Dilithium ML-DSA (Post-Quantum Digital Signatures).
 *
 * CONCEPT
 *   Module-lattice-based digital signature scheme for FIPS 204 quantum-safe signatures.
 *
 * ROLE
 *   Layer S0 post-quantum crypto of the S0-S9 security stack for quantum-safe signing.
 *
 * REFERENCES
 *   FIPS 204 (Dilithium / ML-DSA), NIST PQC Round 3 finalist.
 */



/* Fixed 256-entry NTT twiddle table (bit-reversed order).
 * Ground truth: pq-crystals/dilithium ref/ntt.c. Must be int32_t:
 * q = 8380417, coefficients reach ~+-4M (far outside int16_t). */
static const int32_t dilithium_zetas[256] = {
          0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
  1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
  2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
 -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
  2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
 -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
 -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
   811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
 -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
 -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
  3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
  -671102, -1228525,   -22981,  -1308169,  -381987,  1349076,  1852771, -1430430, -3343383,
   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618, -3724342,
    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,   189548,
 -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,  1285669,
 -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,  2091667,
  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,   266997,
  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,   900702,
  1859098,   909542,  819034,  495491, -1613174,   -43260,  -522500,  -655327,
 -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,   342297,
   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,  2842341,
  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974, -3767016,
  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970, -1333058,
  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642, -1279661,
  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,  -542412,
 -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993, -2013608,
  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385, -3183426,
   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107, -3038916,
  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,  -426683,
  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893, -2939036,
 -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,  -554416,
  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};

static int32_t dilithium_mont_reduce(int64_t a) {
    uint64_t ua = (uint64_t)a;
    uint32_t t = (uint32_t)(ua * (uint64_t)58728449);
    uint32_t r = (uint32_t)((ua - (uint64_t)t * (uint64_t)8380417) >> 32);
    return (int32_t)r;
}

static int32_t dilithium_reduce(int32_t a) {
    int32_t t = (a + (1 << 22)) >> 23;
    t = a - t * 8380417;
    return t;
}

static void dilithium_ntt(int32_t a[256]) {
    unsigned int len, start, j, k;
    int32_t zeta, t;
    k = 0;
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < 256; start = j + len) {
            zeta = dilithium_zetas[++k];
            for (j = start; j < start + len; ++j) {
                t = dilithium_mont_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
    }
}

static void dilithium_inv_ntt(int32_t a[256]) {
    unsigned int start, len, j, k;
    int32_t t, zeta;
    const int32_t f = 41978;
    k = 256;
    for (len = 1; len < 256; len <<= 1) {
        for (start = 0; start < 256; start = j + len) {
            zeta = -dilithium_zetas[--k];
            for (j = start; j < start + len; ++j) {
                t = a[j];
                a[j] = t + a[j + len];
                a[j + len] = t - a[j + len];
                a[j + len] = dilithium_mont_reduce((int64_t)zeta * a[j + len]);
            }
        }
    }
    for (j = 0; j < 256; ++j) a[j] = dilithium_mont_reduce((int64_t)f * a[j]);
}

static void dilithium_poly_add(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    for (int i = 0; i < 256; i++) {
        int32_t s = dilithium_reduce(a[i] + b[i]);
        if (s < 0) s += DILITHIUM_Q;
        r[i] = s;
    }
}

static void dilithium_poly_sub(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    for (int i = 0; i < 256; i++) { r[i] = a[i] - b[i]; if (r[i] < 0) r[i] += DILITHIUM_Q; }
}

static void dilithium_poly_mul(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    int32_t ta[256], tb[256];
    memcpy(ta, a, sizeof(ta)); dilithium_ntt(ta);
    memcpy(tb, b, sizeof(tb)); dilithium_ntt(tb);
    for (int i = 0; i < 256; i++) ta[i] = dilithium_mont_reduce((int64_t)ta[i] * tb[i]);
    dilithium_inv_ntt(ta);
    /* Center into (-Q/2, Q/2] so sparse products like c*s1 keep their
     * true small signed value (e.g. -5, not q-5) used by z=y+c*s1. */
    for (int i = 0; i < 256; i++) ta[i] = dilithium_reduce(ta[i]);
    memcpy(r, ta, sizeof(ta));
}

/* Compressed w1 encoding (FIPS 204 polyw1_pack):
 *   gamma2 == (Q-1)/32  -> w1 in [0,15], 4 coeffs per 2 bytes (128 bytes/poly)
 *   gamma2 == (Q-1)/88  -> w1 in [0,43], 4 coeffs per 3 bytes (192 bytes/poly) */
static void dilithium_poly_w1_pack(uint8_t *out, const int32_t w1[256], int32_t gamma2) {
    if (gamma2 == (DILITHIUM_Q - 1) / 32) {
        for (int i = 0; i < 64; i++) {
            uint32_t t = ((uint32_t)(w1[4*i]   & 0x0F))
                       | ((uint32_t)(w1[4*i+1] & 0x0F) << 4)
                       | ((uint32_t)(w1[4*i+2] & 0x0F) << 8)
                       | ((uint32_t)(w1[4*i+3] & 0x0F) << 12);
            out[2*i]   = (uint8_t)(t & 0xFF);
            out[2*i+1] = (uint8_t)((t >> 8) & 0xFF);
        }
    } else {
        for (int i = 0; i < 64; i++) {
            uint32_t t = ((uint32_t)(w1[4*i]   & 0x3F))
                       | ((uint32_t)(w1[4*i+1] & 0x3F) << 6)
                       | ((uint32_t)(w1[4*i+2] & 0x3F) << 12)
                       | ((uint32_t)(w1[4*i+3] & 0x3F) << 18);
            out[3*i]   = (uint8_t)(t & 0xFF);
            out[3*i+1] = (uint8_t)((t >> 8) & 0xFF);
            out[3*i+2] = (uint8_t)((t >> 16) & 0xFF);
        }
    }
}

static int dilithium_poly_frombytes(int32_t r[256], const uint8_t *in) {
    for (int i = 0; i < 64; i++) {
        r[4*i] = in[13*i] | ((int32_t)in[13*i+1] << 8) | ((int32_t)(in[13*i+2] & 0x1f) << 16);
        r[4*i+1] = (in[13*i+2] >> 5) | ((int32_t)in[13*i+3] << 3) | ((int32_t)in[13*i+4] << 11) | ((int32_t)(in[13*i+5] & 0x03) << 19);
        r[4*i+2] = (in[13*i+5] >> 2) | ((int32_t)in[13*i+6] << 6) | ((int32_t)in[13*i+7] << 14) | ((int32_t)(in[13*i+8] & 0x0f) << 22);
        r[4*i+3] = (in[13*i+8] >> 4) | ((int32_t)in[13*i+9] << 4) | ((int32_t)in[13*i+10] << 12) | ((int32_t)in[13*i+11] << 20) | ((int32_t)in[13*i+12] << 28);
        for (int j = 0; j < 4; j++) if (r[4*i+j] >= DILITHIUM_Q) r[4*i+j] -= DILITHIUM_Q;
    }
    return 0;
}

static void pack_10bit(uint8_t *out, const int32_t in[256]) {
    for (int i = 0; i < 64; i++) {
        uint16_t buf[4];
        for (int j = 0; j < 4; j++) buf[j] = (uint16_t)(in[4*i + j] & 0x3FF);
        uint64_t v = 0;
        for (int j = 0; j < 4; j++) v |= (uint64_t)buf[j] << (10 * j);
        for (int j = 0; j < 5; j++) out[5*i + j] = (v >> (8 * j)) & 0xFF;
    }
}

static void unpack_10bit(int32_t out[256], const uint8_t *in) {
    for (int i = 0; i < 64; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 5; j++) v |= (uint64_t)in[5*i + j] << (8 * j);
        for (int j = 0; j < 4; j++) out[4*i + j] = (int32_t)((v >> (10 * j)) & 0x3FF);
    }
}

static void pack_eta_signed(uint8_t *out, const int32_t in[256], int eta) {
    int bits = (eta == 2) ? 3 : 4;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] + eta;
        buf |= (uint64_t)(val & ((1 << bits) - 1)) << bitpos;
        bitpos += bits;
        while (bitpos >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
    }
    while (bitpos > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
}

static void unpack_eta_signed(int32_t out[256], const uint8_t *in, int eta) {
    int bits = (eta == 2) ? 3 : 4;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bitpos < bits) { buf |= (uint64_t)in[pos++] << bitpos; bitpos += 8; }
        int val = (int)(buf & ((1 << bits) - 1));
        out[i] = val - eta;
        buf >>= bits;
        bitpos -= bits;
    }
}

static void pack_13bit_signed(uint8_t *out, const int32_t in[256]) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] & 0x1FFF;
        buf |= (uint64_t)val << bits;
        bits += 13;
        while (bits >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
    }
    while (bits > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
}

static void unpack_13bit_signed(int32_t out[256], const uint8_t *in) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bits < 13) { buf |= (uint64_t)in[pos++] << bits; bits += 8; }
        out[i] = (int32_t)(buf & 0x1FFF);
        if (out[i] >= 4096) out[i] -= 8192;
        buf >>= 13;
        bits -= 13;
    }
}

static void pack_z(uint8_t *out, const int32_t in[256], int gamma1) {
    int bound = 2 * gamma1;
    int bits = 0;
    while ((1 << bits) < bound) bits++;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] + gamma1;
        buf |= (uint64_t)(val & ((1 << bits) - 1)) << bitpos;
        bitpos += bits;
        while (bitpos >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
    }
    while (bitpos > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
}

static void unpack_z(int32_t out[256], const uint8_t *in, int gamma1) {
    int bound = 2 * gamma1;
    int bits = 0;
    while ((1 << bits) < bound) bits++;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bitpos < bits) { buf |= (uint64_t)in[pos++] << bitpos; bitpos += 8; }
        int val = buf & ((1 << bits) - 1);
        out[i] = val - gamma1;
        buf >>= bits;
        bitpos -= bits;
    }
}

static void dilithium_poly_power2round(int32_t r1[256], int32_t r0[256], const int32_t a[256]) {
    for (int i = 0; i < 256; i++) {
        int32_t a1 = (a[i] + (1 << 12)) >> 13;
        r1[i] = a1;
        r0[i] = a[i] - (a1 << 13);
    }
}

static int32_t dilithium_decompose_scalar(int32_t *a0, int32_t a, int32_t gamma2) {
    int32_t alpha = 2 * gamma2;
    int32_t r = a % DILITHIUM_Q;
    if (r < 0) r += DILITHIUM_Q;               /* r mod+ q, in [0, q-1] */
    int32_t r0 = r % alpha;                    /* central remainder mod+- alpha */
    if (r0 > alpha / 2) r0 -= alpha;
    int32_t r1 = (r - r0) / alpha;             /* HighBits (r - r0 is a multiple of alpha) */
    if (r - r0 == DILITHIUM_Q - 1) {           /* FIPS 204 Decompose special case */
        r1 = 0;
        r0 = r0 - 1;
    }
    *a0 = r0;
    return r1;
}

static void dilithium_poly_decompose(int32_t r1[256], int32_t r0[256], const int32_t a[256], int32_t alpha) {
    int32_t gamma2 = alpha / 2;
    for (int i = 0; i < 256; i++) r1[i] = dilithium_decompose_scalar(&r0[i], a[i], gamma2);
}

static unsigned int dilithium_make_hint_scalar(int32_t a0, int32_t a1, int32_t gamma2) {
    if (a0 > gamma2 || a0 < -gamma2 || (a0 == -gamma2 && a1 != 0)) return 1;
    return 0;
}

static int dilithium_poly_make_hint(int32_t h[256], const int32_t a0[256], const int32_t a1[256], int32_t alpha) {
    int32_t gamma2 = alpha / 2;
    int cnt = 0;
    for (int i = 0; i < 256; i++) { h[i] = (int32_t)dilithium_make_hint_scalar(a0[i], a1[i], gamma2); cnt += h[i]; }
    return cnt;
}

static int32_t dilithium_use_hint_scalar(int32_t a, int hint, int32_t gamma2) {
    int32_t a0, a1 = dilithium_decompose_scalar(&a0, a, gamma2);
    if (hint == 0) return a1;
    if (gamma2 == (DILITHIUM_Q - 1) / 32) {
        return (a0 > 0) ? ((a1 + 1) & 15) : ((a1 - 1) & 15);
    } else { /* (Q-1)/88 */
        return (a0 > 0) ? ((a1 == 43) ? 0 : a1 + 1) : ((a1 == 0) ? 43 : a1 - 1);
    }
}

static void dilithium_poly_use_hint(int32_t r[256], const int32_t a[256], const int32_t h[256], int32_t alpha) {
    int32_t gamma2 = alpha / 2;
    for (int i = 0; i < 256; i++) r[i] = dilithium_use_hint_scalar(a[i], (int)h[i], gamma2);
}

static int32_t dilithium_gamma2(int variant) {
    return (variant == 2) ? (DILITHIUM_Q - 1) / 88 : (DILITHIUM_Q - 1) / 32;
}


static void cbd_eta2(int32_t r[256], const uint8_t seed[32], uint8_t nonce) {
    uint8_t buf[128];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, &nonce, 1);
    SNEPPX_drbg_generate(&drbg, buf, 128);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < 256; i++) {
        int a = (buf[i/4] >> (2*(i%4))) & 3;
        int b = (buf[i/4 + 64] >> (2*(i%4))) & 3;
        int a_bits = (a & 1) + ((a >> 1) & 1);
        int b_bits = (b & 1) + ((b >> 1) & 1);
        r[i] = a_bits - b_bits;
    }
}

static void cbd_eta4(int32_t r[256], const uint8_t seed[32], uint8_t nonce) {
    uint8_t buf[256];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, &nonce, 1);
    SNEPPX_drbg_generate(&drbg, buf, 256);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < 256; i++) {
        int a = (buf[i/2] >> (4*(i%2))) & 15;
        int b = (buf[i/2 + 128] >> (4*(i%2))) & 15;
        int a_bits = (a & 1) + ((a>>1)&1) + ((a>>2)&1) + ((a>>3)&1);
        int b_bits = (b & 1) + ((b>>1)&1) + ((b>>2)&1) + ((b>>3)&1);
        r[i] = a_bits - b_bits;
    }
}

static void dilithium_expand_a(int32_t *a, const uint8_t rho[32], int k) {
    int total = k * k * 256;
    int bytes_needed = total * 3;
    uint8_t *buf = (uint8_t*)malloc(bytes_needed);
    if (!buf) return;
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, rho, 32, NULL, 0);
    SNEPPX_drbg_generate(&drbg, buf, bytes_needed);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < total; i++) {
        a[i] = (int32_t)(buf[3*i] | ((uint32_t)buf[3*i+1] << 8) | ((uint32_t)buf[3*i+2] << 16));
        if (a[i] >= DILITHIUM_Q) a[i] %= DILITHIUM_Q;
    }
    free(buf);
}

static void dilithium_poly_challenge(int32_t c[256], const uint8_t seed[32], int tau) {
    uint8_t buf[640];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, NULL, 0);
    SNEPPX_drbg_generate(&drbg, buf, sizeof(buf));
    SNEPPX_drbg_destroy(&drbg);
    memset(c, 0, 256 * sizeof(int32_t));
    int pos = 0, found = 0;
    while (found < tau && pos + 2 <= (int)sizeof(buf)) {
        int idx = buf[pos] % 256;
        int sign = (buf[pos + 1] & 1) ? 1 : -1;
        pos += 2;
        if (c[idx] == 0) { c[idx] = sign; found++; }
    }
}

static void dilithium_mvp(int32_t *out, const int32_t *a, const int32_t *vec, int k) {
    memset(out, 0, k * 256 * sizeof(int32_t));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            int32_t term[256];
            dilithium_poly_mul(term, a + (i * k + j) * 256, vec + j * 256);
            dilithium_poly_add(out + i * 256, out + i * 256, term);
        }
    }
}

/**
 * @brief Generate Dilithium.
 *
 * @param pk [out] Pk value.
 * @param sk [out] Sk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dilithium_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    int s_bytes = (eta == 2) ? 96 : 128;
    uint8_t rho[32], rho_prime[32];
    SNEPPX_random_bytes(rho, 32);
    SNEPPX_random_bytes(rho_prime, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    if (!a || !s1 || !s2 || !t || !t1 || !t0) {
        free(a); free(s1); free(s2); free(t); free(t1); free(t0);
        return -1;
    }
    dilithium_expand_a(a, rho, k);
    for (int i = 0; i < k; i++) {
        if (eta == 2) {
            cbd_eta2(s1 + i * 256, rho_prime, i);
            cbd_eta2(s2 + i * 256, rho_prime, k + i);
        } else {
            cbd_eta4(s1 + i * 256, rho_prime, i);
            cbd_eta4(s2 + i * 256, rho_prime, k + i);
        }
    }
    dilithium_mvp(t, a, s1, k);
    for (int i = 0; i < k; i++) dilithium_poly_add(t + i * 256, t + i * 256, s2 + i * 256);
    for (int i = 0; i < k; i++) {
        dilithium_poly_power2round(t1 + i * 256, t0 + i * 256, t + i * 256);
    }
    memcpy(pk, rho, 32);
    for (int i = 0; i < k; i++) pack_10bit(pk + 32 + i * 320, t1 + i * 256);
    int sk_pos = 0;
    memcpy(sk + sk_pos, rho, 32); sk_pos += 32;
    memcpy(sk + sk_pos, rho_prime, 32); sk_pos += 32;
    memset(sk + sk_pos, 0, 32); sk_pos += 32;
    for (int i = 0; i < k; i++) pack_eta_signed(sk + sk_pos + i * s_bytes, s1 + i * 256, eta);
    sk_pos += k * s_bytes;
    for (int i = 0; i < k; i++) pack_eta_signed(sk + sk_pos + i * s_bytes, s2 + i * 256, eta);
    sk_pos += k * s_bytes;
    for (int i = 0; i < k; i++) pack_13bit_signed(sk + sk_pos + i * 416, t0 + i * 256);
    sk_pos += k * 416;
    free(a); free(s1); free(s2); free(t); free(t1); free(t0);
    return 0;
}

int SNEPPX_dilithium_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant) {
    if (!sig || !siglen || !m || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    int s_bytes = (eta == 2) ? 96 : 128;
    int tau = (variant == 2) ? 39 : (variant == 3) ? 49 : 60;
    int gamma1 = (variant == 2) ? 1 << 17 : 1 << 19;
    int gamma2 = dilithium_gamma2(variant);
    int alpha = 2 * gamma2;
    int beta = 2 * tau;
    uint8_t rho[32], rho_prime[32];
    memcpy(rho, sk, 32);
    memcpy(rho_prime, sk + 32, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *y = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *z = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *cs1 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *cs2 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *ct0 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *h = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *hint_in = (int32_t*)calloc(256, sizeof(int32_t));
    if (!a || !s1 || !s2 || !t0 || !y || !w || !w1 || !w0 || !z || !cs1 || !cs2 || !ct0 || !h || !hint_in) {
        free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
        free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
        return -1;
    }
    int sk_pos = 96;
    for (int i = 0; i < k; i++) unpack_eta_signed(s1 + i * 256, sk + sk_pos + i * s_bytes, eta);
    sk_pos += k * s_bytes;
    for (int i = 0; i < k; i++) unpack_eta_signed(s2 + i * 256, sk + sk_pos + i * s_bytes, eta);
    sk_pos += k * s_bytes;
    for (int i = 0; i < k; i++) unpack_13bit_signed(t0 + i * 256, sk + sk_pos + i * 416);
    dilithium_expand_a(a, rho, k);
    uint8_t c_seed[32];
    int32_t c_poly[256];
    int reject;
    int max_attempts = 1000;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 256; j++) {
                uint32_t r;
                SNEPPX_random_bytes((uint8_t*)&r, 4);
                r %= (uint32_t)(2 * gamma1);
                y[i * 256 + j] = (int32_t)r - gamma1;
            }
        }
        dilithium_mvp(w, a, y, k);
        memset(w1, 0, k * 256 * sizeof(int32_t));
        memset(w0, 0, k * 256 * sizeof(int32_t));
        for (int i = 0; i < k; i++)
            dilithium_poly_decompose(w1 + i * 256, w0 + i * 256, w + i * 256, alpha);
        int w1_bytes = (gamma2 == (DILITHIUM_Q - 1) / 32) ? 128 : 192;
        uint8_t *w1_enc = (uint8_t*)calloc((size_t)k * w1_bytes, 1);
        if (!w1_enc) { reject = 1; }
        else {
            for (int i = 0; i < k; i++) dilithium_poly_w1_pack(w1_enc + i * w1_bytes, w1 + i * 256, gamma2);
            uint8_t hash_buf[64];
            SNEPPXSHA512Context ctx;
            SNEPPX_sha512_init(&ctx);
            SNEPPX_sha512_update(&ctx, m, mlen);
            SNEPPX_sha512_update(&ctx, w1_enc, (size_t)k * w1_bytes);
            SNEPPX_sha512_finish(&ctx, hash_buf);
            memcpy(c_seed, hash_buf, 32);
            free(w1_enc);
        }
        dilithium_poly_challenge(c_poly, c_seed, tau);
        reject = 0;
        int z_reject = 0;
        for (int i = 0; i < k; i++) {
            memset(cs1, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(cs1, c_poly, s1 + i * 256);
            for (int j = 0; j < 256; j++) {
                z[i * 256 + j] = y[i * 256 + j] + cs1[j];
                if (z[i * 256 + j] > gamma1 - beta || z[i * 256 + j] < -(gamma1 - beta))
                    z_reject = 1;
            }
        }
        reject = z_reject;
        if (reject) continue;
        int w0_reject = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 256; j++) {
                if (w0[i * 256 + j] > alpha / 2 || w0[i * 256 + j] < -(alpha / 2))
                    w0_reject = 1;
            }
        }
        if (w0_reject) continue;
        int r0_reject = 0;
        for (int i = 0; i < k; i++) {
            memset(cs2, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(cs2, c_poly, s2 + i * 256);
            int32_t r0poly[256];
            for (int j = 0; j < 256; j++) r0poly[j] = w0[i * 256 + j] - cs2[j];
            for (int j = 0; j < 256; j++) {
                if (r0poly[j] > alpha / 2 - beta || r0poly[j] < -(alpha / 2 - beta))
                    r0_reject = 1;
            }
        }
        if (r0_reject) continue;
        for (int i = 0; i < k; i++) {
            dilithium_poly_mul(ct0, c_poly, t0 + i * 256);
            dilithium_poly_mul(cs2, c_poly, s2 + i * 256);
            int32_t g2 = dilithium_gamma2(variant);
            for (int j = 0; j < 256; j++) {
                int32_t a0v = w0[i * 256 + j] - cs2[j] + ct0[j];
                h[i * 256 + j] = (int32_t)dilithium_make_hint_scalar(a0v, w1[i * 256 + j], g2);
            }
        }
        size_t pos = 0;
        for (int i = 0; i < k; i++) {
            pack_z(sig + pos, z + i * 256, gamma1);
            int z_bits = 0; while ((1 << z_bits) < 2 * gamma1) z_bits++;
            pos += (256 * z_bits + 7) / 8;
        }
        for (int i = 0; i < k; i++) {
            uint8_t hint_byte = 0;
            int hbits = 0;
            for (int j = 0; j < 256; j++) {
                hint_byte |= (uint8_t)(h[i * 256 + j] & 1) << hbits;
                hbits++;
                if (hbits == 8) { sig[pos++] = hint_byte; hint_byte = 0; hbits = 0; }
            }
            if (hbits > 0) sig[pos++] = hint_byte;
        }
        memcpy(sig + pos, c_seed, 32);
        pos += 32;
        *siglen = pos;
        free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
        free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
        return 0;
    }
    free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
    free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
    return -1;
}

/**
 * @brief Verify Dilithium.
 *
 * @param sig [in] Sig value.
 * @param siglen [in] Siglen value.
 * @param m [in] M value.
 * @param mlen [in] Mlen value.
 * @param pk [in] Pk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dilithium_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant) {
    if (!sig || !pk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    int tau = (variant == 2) ? 39 : (variant == 3) ? 49 : 60;
    int gamma1 = (variant == 2) ? 1 << 17 : 1 << 19;
    int gamma2 = dilithium_gamma2(variant);
    int alpha = 2 * gamma2;
    int beta = 2 * tau;
    int z_bits = 0;
    while ((1 << z_bits) < 2 * gamma1) z_bits++;
    size_t z_bytes = (256 * z_bits + 7) / 8;
    size_t hint_bytes = (k * 256 + 7) / 8;
    size_t expected_siglen = k * z_bytes + hint_bytes + 32;
    if (siglen < expected_siglen) return -1;
    uint8_t rho[32];
    memcpy(rho, pk, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *t1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *zvec = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *h = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *az = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *ct1_shifted = (int32_t*)calloc(256, sizeof(int32_t));
    if (!a || !t1 || !zvec || !h || !az || !w1 || !ct1_shifted) {
        free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
        return -1;
    }
    dilithium_expand_a(a, rho, k);
    for (int i = 0; i < k; i++) unpack_10bit(t1 + i * 256, pk + 32 + i * 320);
    size_t pos = 0;
    for (int i = 0; i < k; i++) {
        unpack_z(zvec + i * 256, sig + pos, gamma1);
        pos += z_bytes;
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 256; j++) {
            int byte_idx = (int)(pos + (i * 256 + j) / 8);
            int bit_idx = (int)((i * 256 + j) % 8);
            if ((size_t)byte_idx < siglen)
                h[i * 256 + j] = (sig[byte_idx] >> bit_idx) & 1;
        }
    }
    pos += hint_bytes;
    uint8_t c_seed[32];
    memcpy(c_seed, sig + pos, 32);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 256; j++) {
            int32_t c = zvec[i * 256 + j];
            if (c > gamma1 - beta || c < -(gamma1 - beta)) {
                free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
                return -1;
            }
        }
    }
    int32_t c_poly[256];
    dilithium_poly_challenge(c_poly, c_seed, tau);
    memset(az, 0, k * 256 * sizeof(int32_t));
    dilithium_mvp(az, a, zvec, k);
    for (int i = 0; i < k; i++) {
        memset(ct1_shifted, 0, 256 * sizeof(int32_t));
        dilithium_poly_mul(ct1_shifted, c_poly, t1 + i * 256);
        for (int j = 0; j < 256; j++) ct1_shifted[j] = (int32_t)(((int64_t)ct1_shifted[j] * 8192) % DILITHIUM_Q);
        int32_t diff[256];
        for (int j = 0; j < 256; j++) {
            diff[j] = az[i * 256 + j] - ct1_shifted[j];
            diff[j] = (diff[j] % DILITHIUM_Q + DILITHIUM_Q) % DILITHIUM_Q;
        }
        dilithium_poly_use_hint(w1 + i * 256, diff, h + i * 256, alpha);
    }
    int w1_bytes = (gamma2 == (DILITHIUM_Q - 1) / 32) ? 128 : 192;
    uint8_t *w1_enc = (uint8_t*)calloc((size_t)k * w1_bytes, 1);
    if (!w1_enc) { free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted); return -1; }
    for (int i = 0; i < k; i++) dilithium_poly_w1_pack(w1_enc + i * w1_bytes, w1 + i * 256, gamma2);
    uint8_t hash_buf[64];
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, m, mlen);
    SNEPPX_sha512_update(&ctx, w1_enc, (size_t)k * w1_bytes);
    SNEPPX_sha512_finish(&ctx, hash_buf);
    free(w1_enc);
    if (memcmp(c_seed, hash_buf, 32) != 0) {
        free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
        return -1;
    }
    free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
    return 0;
}
