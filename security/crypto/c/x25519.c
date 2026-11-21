#include "x25519.h"
#include "cryptographic_random_generator.h"
#include <string.h>
#include <stdlib.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #include <intrin.h>
#endif

static const uint8_t basepoint[32] = {9};

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
static uint64_t umul128(uint64_t a, uint64_t b, uint64_t *hi) {
    return _umul128(a, b, hi);
}
#else
static uint64_t umul128(uint64_t a, uint64_t b, uint64_t *hi) {
    __uint128_t p = (__uint128_t)a * b;
    *hi = (uint64_t)(p >> 64);
    return (uint64_t)p;
}
#endif
/*
 * SNEPPX - X25519 Key Exchange (ECDH over Curve25519)
 *
 * WHAT
 *   X25519 Key Exchange (ECDH over Curve25519).
 *
 * CONCEPT
 *   X25519 ECDH key exchange computing shared secrets from private/public key pairs.
 *
 * ROLE
 *   Used by the key-vault for establishing shared secrets with peers (X25519 + HKDF = Noise pattern).
 *
 * REFERENCES
 *   RFC 7748 (X25519), Daniel Bernstein Curve25519.
 */



static void mul(uint64_t r[8], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t p[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo = umul128(a[i], b[j], &hi);
            uint64_t sum = p[i + j] + lo + carry;
            p[i + j] = sum;
            carry = hi + (sum < lo ? 1ULL : 0) + (sum < p[i + j] ? 1ULL : 0);
        }
        p[i + 4] += carry;
    }
    memcpy(r, p, 64);
}

static void reduce(uint64_t r[4], const uint64_t t[8]) {
    uint64_t t2[8];
    memcpy(t2, t, 64);
    if (t2[3] >> 63) {
        uint64_t c = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = t2[i] + c;
            t2[i] = sum;
            c = (sum < c) ? 1 : 0;
        }
        t2[2] += 19;
        if (t2[2] < 19) t2[3]++;
        t2[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
    for (int i = 0; i < 4; i++) r[i] = t2[i] & 0xFFFFFFFFULL;
    r[0] = (r[0] + (t2[4] & 0xFFFFFFFFULL) * 38) & 0xFFFFFFFFULL;
    r[1] = (r[1] + ((t2[0] >> 32) + (t2[4] >> 32) * 38 + (t2[5] & 0xFFFFFFFFULL) * 38)) & 0xFFFFFFFFULL;
    r[2] = (r[2] + ((t2[1] >> 32) + (t2[5] >> 32) * 38 + (t2[6] & 0xFFFFFFFFULL) * 38)) & 0xFFFFFFFFULL;
    r[3] = (r[3] + ((t2[2] >> 32) + (t2[6] >> 32) * 38 + (t2[7] & 0xFFFFFFFFULL) * 38)) & 0xFFFFFFFFULL;
    uint64_t carry = r[1] >> 32; r[0] += carry * 38; r[1] &= 0xFFFFFFFFULL;
    carry = r[2] >> 32; r[1] += carry; r[2] &= 0xFFFFFFFFULL;
    carry = r[3] >> 32; r[2] += carry; r[3] &= 0xFFFFFFFFULL;
    carry = r[0] >> 32; r[0] &= 0xFFFFFFFFULL; r[1] += carry;
    carry = r[0] >> 32; r[0] &= 0xFFFFFFFFULL; r[1] += carry;
    uint64_t r3_t = r[3];
    if (r3_t > 0x7FFFFFFFFFFFFFFFULL) {
        r[0] += 19;
        if (r[0] < 19) r[1]++;
        if (r[1] == 0) { r[2]++; if (r[2] == 0) r[3]++; }
        r[3] &= 0x7FFFFFFFFFFFFFFFULL;
    }
}

static void fe_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 0; i < 4; i++) r[i] = a[i] + b[i];
}

static void fe_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    r[0] = a[0] + (0xFFFFFFFFFFFFFFFFULL - b[0]);
    uint64_t c = (r[0] < a[0]) ? 0 : 1;
    r[1] = a[1] + (0xFFFFFFFFFFFFFFFFULL - b[1]) + c;
    c = (r[1] < (a[1] + c)) ? 0 : 1;
    r[2] = a[2] + (0xFFFFFFFFFFFFFFFFULL - b[2]) + c;
    c = (r[2] < (a[2] + c)) ? 0 : 1;
    r[3] = a[3] + (0xFFFFFFFFFFFFFFFFULL - b[3]) + c;
}

static void fe_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t t[8];
    mul(t, a, b);
    reduce(r, t);
}

static void fe_sq(uint64_t r[4], const uint64_t a[4]) {
    fe_mul(r, a, a);
}

static void fe_invert(uint64_t r[4], const uint64_t a[4]) {
    uint64_t t[4], t2[4], t5[4], t10[4], t20[4], t50[4], t100[4];
    memcpy(t, a, 32);
    fe_sq(t, t); fe_sq(t2, t); fe_mul(t, t2, a);
    fe_sq(t2, t); fe_sq(t10, t2); fe_mul(t5, t10, t);
    fe_sq(t10, t5); fe_sq(t10, t10); fe_mul(t10, t10, t5);
    fe_sq(t20, t10); fe_sq(t20, t20); fe_sq(t20, t20); fe_sq(t20, t20);
    fe_mul(t20, t20, t10);
    fe_sq(t50, t20); fe_sq(t50, t50); fe_mul(t50, t50, t5);
    fe_sq(t100, t50); fe_sq(t100, t100); fe_mul(t100, t100, t20);
    fe_sq(t100, t100); fe_sq(t100, t100); fe_mul(t100, t100, t10);
    fe_sq(t50, t100); fe_sq(t50, t50); fe_mul(t50, t50, t5);
    fe_sq(t2, t50); fe_sq(t2, t2);
    fe_mul(r, t2, a);
}

static void fe_cswap(uint64_t x[4], uint64_t z[4], uint64_t mask) {
    for (int i = 0; i < 4; i++) {
        uint64_t dx = (x[i] ^ z[i]) & mask;
        x[i] ^= dx;
        z[i] ^= dx;
    }
}

static void mask_l64(uint64_t *out, int bit) {
    *out = 0 - (uint64_t)(bit & 1);
}

/**
 * @brief Perform X25519 Clamp.
 */
void SNEPPX_x25519_clamp(uint8_t scalar[32]) {
    scalar[0] &= 248;
    scalar[31] &= 127;
    scalar[31] |= 64;
}

static void fe_mul_a24(uint64_t r[4], const uint64_t e[4]) {
    uint64_t a24 = 121665;
    uint64_t t[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo = umul128(e[i], a24, &hi);
        uint64_t sum = t[i] + lo;
        t[i] = sum;
        uint64_t carry = hi + (sum < lo ? 1ULL : 0);
        int j = i + 1;
        while (carry) {
            sum = t[j] + carry;
            t[j] = sum;
            carry = (sum < carry) ? 1 : 0;
            j++;
        }
    }
    reduce(r, t);
}

/**
 * @brief Perform X25519 Scalar Mult.
 */
void SNEPPX_x25519_scalar_mult(uint8_t out[32], const uint8_t scalar[32], const uint8_t point[32]) {
    uint64_t x2[4], z2[4], x3[4], z3[4];
    uint64_t a[4], b[4], c[4], d[4];
    uint64_t aa[4], bb[4], e[4];
    uint64_t da_cb[4], da_m_cb[4];
    uint64_t x_point[4];
    uint8_t e_[32];
    memcpy(e_, scalar, 32);
    SNEPPX_x25519_clamp(e_);
    for (int i = 0; i < 4; i++) x_point[i] = ((const uint64_t *)point)[i];

    memset(x2, 0, 32); memset(z2, 0, 32); x2[0] = 1;
    memcpy(x3, x_point, 32);
    memset(z3, 0, 32); z3[0] = 1;

    uint64_t prev = 0;
    for (int i = 254; i >= 0; i--) {
        uint64_t bit = (e_[i / 8] >> (i % 8)) & 1;
        uint64_t sw = bit ^ prev;
        uint64_t m = 0 - sw;
        fe_cswap(x2, x3, m);
        fe_cswap(z2, z3, m);
        prev = bit;

        fe_add(a, x2, z2);
        fe_sub(b, x2, z2);
        fe_add(c, x3, z3);
        fe_sub(d, x3, z3);

        fe_mul(aa, a, a);
        fe_mul(bb, b, b);
        fe_mul(x3, a, d);
        fe_mul(z3, b, c);

        fe_add(da_cb, x3, z3);
        fe_sub(da_m_cb, x3, z3);
        fe_sq(x3, da_cb);
        fe_sq(da_m_cb, da_m_cb);
        fe_mul(z3, da_m_cb, x_point);

        fe_sub(e, aa, bb);
        fe_mul(x2, aa, bb);
        fe_mul_a24(aa, e);
        fe_add(aa, aa, e);
        fe_mul(z2, e, aa);
    }
    fe_cswap(x2, x3, 0 - prev);
    fe_cswap(z2, z3, 0 - prev);
    fe_invert(z2, z2);
    fe_mul(x2, x2, z2);
    uint8_t *p = (uint8_t *)x2;
    for (int i = 0; i < 32; i++) out[31 - i] = p[i];
}

/**
 * @brief Generate X25519.
 */
void SNEPPX_x25519_keygen(uint8_t public_key[32], uint8_t secret_key[32]) {
    SNEPPX_random_bytes(secret_key, 32);
    SNEPPX_x25519_clamp(secret_key);
    SNEPPX_x25519_scalar_mult(public_key, secret_key, basepoint);
}

/**
 * @brief Perform X25519 Shared Secret.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_x25519_shared_secret(uint8_t shared[32], const uint8_t secret_key[32], const uint8_t public_key[32]) {
    if (!shared || !secret_key || !public_key) return -1;
    SNEPPX_x25519_scalar_mult(shared, secret_key, public_key);
    return 0;
}

/**
 * @brief Perform X25519 Scalar Valid.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_x25519_scalar_valid(const uint8_t scalar[32]) {
    if (!scalar) return 0;
    for (int i = 0; i < 32; i++) if (scalar[i]) return 1;
    return 0;
}
