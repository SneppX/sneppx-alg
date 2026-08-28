#include "ed25519_signature_verification.h"
#include "sha512_hashing_implementation.h"
#include "constant_time_operations.h"
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <stdio.h>

/*
 * SNEPPX - Ed25519
 *
 * WHAT
 *   Ed25519.
 *
 * CONCEPT
 *   Provides Ed25519 signatures (RFC 8032).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Random Bytes.
 *
 * @param buffer [out] Buffer value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_random_bytes(uint8_t* buffer, size_t len);

/* GF(2^255-19) field element: 5 limbs of 51 bits */
typedef struct { uint64_t v[5]; } field;
typedef struct { uint64_t lo, hi; } uint128;

static uint128 add128(uint128 a, uint128 b) {
    uint128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0); return r;
}
#ifdef _MSC_VER
static uint128 mul64_64(uint64_t a, uint64_t b) {
    uint128 r; r.lo = _umul128(a, b, &r.hi); return r;
}
#else
static uint128 mul64_64(uint64_t a, uint64_t b) {
    unsigned __int128 r = (unsigned __int128)a * b;
    uint128 result; result.lo = (uint64_t)r; result.hi = (uint64_t)(r >> 64); return result;
}
#endif
/*
 * SNEPPX - Ed25519 Digital Signatures
 *
 * WHAT
 *   Ed25519 Digital Signatures.
 *
 * CONCEPT
 *   Ed25519 signature generation and verification on the Edwards curve Ed25519.
 *
 * ROLE
 *   Used for code-signing, release manifests, and compact (64-byte) authentication.
 *
 * REFERENCES
 *   RFC 8032 (EdDSA), Daniel Bernstein high-speed signatures.
 */



static const field D = {{
    0x00034DCA135978A3ULL, 0x0001A8283B156EBDULL, 0x0005E7A26001C029ULL,
    0x000739C663A03CBBULL, 0x00052036CEE2B6FFULL
}};
static const field D2 = {{
    0x00069B9426B2F159ULL, 0x00035050762ADD7AULL, 0x0003CF44C0038052ULL,
    0x0006738CC7407977ULL, 0x0002406D9DC56DFFULL
}};
static const field SQRTM1 = {{
    0x61b274a0ea0b0ULL, 0xd5a5fc8f189dULL, 0x7ef5e9cbd0c60ULL,
    0x78595a6804c9eULL, 0x2b8324804fc1dULL
}};

static uint64_t mask51(uint64_t x) { return x & 0x7ffffffffffffULL; }
static uint64_t reduce51(uint64_t x) { return (x >> 51) + (x & 0x7ffffffffffffULL); }
static uint128 rshift128(uint128 x, unsigned s) {
    uint128 r; r.lo = (x.lo >> s) | (x.hi << (64 - s)); r.hi = x.hi >> s; return r;
}

static void fe_add(field* r, const field* a, const field* b) {
    for (int i = 0; i < 5; i++) r->v[i] = a->v[i] + b->v[i];
}

static void fe_sub(field* r, const field* a, const field* b) {
    int64_t t[5];
    for (int i = 0; i < 5; i++) t[i] = (int64_t)a->v[i] - (int64_t)b->v[i];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < 4; i++) {
            if (t[i] < 0) { t[i] += 0x8000000000000LL; t[i+1]--; }
        }
        if (t[4] < 0) { t[4] += 0x8000000000000LL; t[0] -= 19; }
    }
    for (int i = 0; i < 5; i++) r->v[i] = (uint64_t)t[i];
}

static void fe_mul(field* r, const field* a, const field* b) {
    uint128 t[9] = {0};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            int k = i + j;
            if (k < 9) t[k] = add128(t[k], mul64_64(a->v[i], b->v[j]));
        }
    /* Fold high terms (×19) back into lower 5 */
    for (int i = 8; i >= 5; i--) {
        uint64_t low51 = t[i].lo & 0x7ffffffffffffULL;
        uint128 high = rshift128(t[i], 51);
        t[i-5] = add128(t[i-5], mul64_64(19, low51));
        t[i-4] = add128(t[i-4], mul64_64(19, high.lo));
    }
    /* Two rounds of carry propagation */
    for (int round = 0; round < 2; round++) {
        uint128 carry;
        carry = rshift128(t[0], 51); t[0].lo = mask51(t[0].lo); t[0].hi = 0;
        t[1] = add128(t[1], carry);
        carry = rshift128(t[1], 51); t[1].lo = mask51(t[1].lo); t[1].hi = 0;
        t[2] = add128(t[2], carry);
        carry = rshift128(t[2], 51); t[2].lo = mask51(t[2].lo); t[2].hi = 0;
        t[3] = add128(t[3], carry);
        carry = rshift128(t[3], 51); t[3].lo = mask51(t[3].lo); t[3].hi = 0;
        t[4] = add128(t[4], carry);
        carry = rshift128(t[4], 51); t[4].lo = mask51(t[4].lo); t[4].hi = 0;
        if (carry.lo || carry.hi) {
            t[0] = add128(t[0], mul64_64(19, carry.lo));
            t[0] = add128(t[0], mul64_64(19, carry.hi));
        }
    }
    for (int i = 0; i < 5; i++) r->v[i] = t[i].lo;
}

static void fe_sq(field* r, const field* a) { fe_mul(r, a, a); }

static void fe_from_bytes(field* r, const uint8_t b[32]) {
    r->v[0] = (uint64_t)b[0] | (uint64_t)b[1]<<8 | (uint64_t)b[2]<<16 | (uint64_t)b[3]<<24 |
              (uint64_t)b[4]<<32 | (uint64_t)b[5]<<40 | ((uint64_t)b[6]&7)<<48;
    r->v[1] = (uint64_t)b[6]>>3 | (uint64_t)b[7]<<5 | (uint64_t)b[8]<<13 | (uint64_t)b[9]<<21 |
              (uint64_t)b[10]<<29 | (uint64_t)b[11]<<37 | ((uint64_t)b[12]&0x3F)<<45;
    r->v[2] = (uint64_t)b[12]>>6 | (uint64_t)b[13]<<2 | (uint64_t)b[14]<<10 | (uint64_t)b[15]<<18 |
              (uint64_t)b[16]<<26 | (uint64_t)b[17]<<34 | (uint64_t)b[18]<<42 | ((uint64_t)b[19]&1)<<50;
    r->v[3] = ((uint64_t)b[19]>>1)&0x7F | (uint64_t)b[20]<<7 | (uint64_t)b[21]<<15 | (uint64_t)b[22]<<23 |
              (uint64_t)b[23]<<31 | (uint64_t)b[24]<<39 | ((uint64_t)b[25]&0x0F)<<47;
    r->v[4] = (uint64_t)b[25]>>4 | (uint64_t)b[26]<<4 | (uint64_t)b[27]<<12 | (uint64_t)b[28]<<20 |
              (uint64_t)b[29]<<28 | (uint64_t)b[30]<<36 | ((uint64_t)b[31]&0x7F)<<44;
}

static void fe_to_bytes(uint8_t b[32], const field* r) {
    uint64_t t[5];
    memcpy(t, r->v, sizeof(t));
    uint64_t c = 19 * (t[4] >> 51);
    t[0] += c; t[4] = mask51(t[4]);
    for (int round = 0; round < 2; round++) {
        uint64_t c0 = t[0] >> 51; t[1] += c0; t[0] = mask51(t[0]);
        uint64_t c1 = t[1] >> 51; t[2] += c1; t[1] = mask51(t[1]);
        uint64_t c2 = t[2] >> 51; t[3] += c2; t[2] = mask51(t[2]);
        uint64_t c3 = t[3] >> 51; t[4] += c3; t[3] = mask51(t[3]);
        uint64_t c4 = t[4] >> 51; t[0] += 19 * c4; t[4] = mask51(t[4]);
    }
    { int ge = 1;
      for (int i = 4; i >= 1; i--) { if (t[i] != 0x7FFFFFFFFFFFFULL) { ge = 0; break; } }
      if (ge && t[0] >= 0x7FFFFFFFFFFEDULL) {
          t[0] -= 0x7FFFFFFFFFFEDULL;
          for (int i = 1; i < 5; i++) t[i] -= 0x7FFFFFFFFFFFFULL;
      }
    }
    b[0] = (uint8_t)t[0]; b[1] = (uint8_t)(t[0] >> 8); b[2] = (uint8_t)(t[0] >> 16); b[3] = (uint8_t)(t[0] >> 24);
    b[4] = (uint8_t)(t[0] >> 32); b[5] = (uint8_t)(t[0] >> 40);
    b[6] = (uint8_t)(t[0] >> 48) | (uint8_t)((t[1] & 0x1F) << 3);
    b[7] = (uint8_t)(t[1] >> 5); b[8] = (uint8_t)(t[1] >> 13); b[9] = (uint8_t)(t[1] >> 21);
    b[10] = (uint8_t)(t[1] >> 29); b[11] = (uint8_t)(t[1] >> 37);
    b[12] = (uint8_t)(t[1] >> 45) | (uint8_t)((t[2] & 3) << 6);
    b[13] = (uint8_t)(t[2] >> 2); b[14] = (uint8_t)(t[2] >> 10); b[15] = (uint8_t)(t[2] >> 18);
    b[16] = (uint8_t)(t[2] >> 26); b[17] = (uint8_t)(t[2] >> 34); b[18] = (uint8_t)(t[2] >> 42);
    b[19] = (uint8_t)(t[2] >> 50) | (uint8_t)((t[3] & 0x7F) << 1);
    b[20] = (uint8_t)(t[3] >> 7); b[21] = (uint8_t)(t[3] >> 15); b[22] = (uint8_t)(t[3] >> 23);
    b[23] = (uint8_t)(t[3] >> 31); b[24] = (uint8_t)(t[3] >> 39);
    b[25] = (uint8_t)(t[3] >> 47) | (uint8_t)((t[4] & 0x0F) << 4);
    b[26] = (uint8_t)(t[4] >> 4); b[27] = (uint8_t)(t[4] >> 12); b[28] = (uint8_t)(t[4] >> 20);
    b[29] = (uint8_t)(t[4] >> 28); b[30] = (uint8_t)(t[4] >> 36);
    b[31] = (uint8_t)(t[4] >> 44);
    { int ge = 1;
      for (int i = 30; i >= 0; i--) {
          uint8_t p_byte = (i == 0) ? 0xED : 0xFF;
          if (b[i] != p_byte) { ge = (b[i] > p_byte); break; }
      }
      if (ge) {
          uint16_t brw = 0;
          for (int i = 0; i < 32; i++) {
              uint16_t sub = (i == 0) ? 0xED : (i == 31) ? 0x7F : 0xFF;
              uint16_t d = (uint16_t)b[i] - sub - brw;
              b[i] = (uint8_t)(d & 0xFF);
              brw = (d >> 8) & 1;
          }
      }
    }
}

/* Point in extended coordinates (X, Y, Z, T) where x=X/Z, y=Y/Z, x*y=T/Z */
typedef struct { field X, Y, Z, T; } point;

static void fe_neg(field* r, const field* a) {
    field zero; memset(&zero, 0, sizeof(field)); fe_sub(r, &zero, a);
}

static void point_negate(point* r, const point* p) {
    fe_neg(&r->X, &p->X);
    fe_neg(&r->Y, &p->Y);
    memcpy(&r->Z, &p->Z, sizeof(field));
    memcpy(&r->T, &p->T, sizeof(field));
}

static void point_set_neutral(point* p) {
    memset(p, 0, sizeof(point));
    p->Y.v[0] = 1; p->Z.v[0] = 1;
}

/* Compute a^((p-5)/8) = a^(2^252 - 3) via square-and-multiply */
static void fe_pow22523(field* r, const field* a) {
    field base; memcpy(&base, a, sizeof(field));
    memcpy(r, a, sizeof(field));
    for (int i = 250; i >= 0; i--) {
        fe_sq(r, r);
        int bit = (i >= 2) || (i == 0);
        if (bit) fe_mul(r, r, &base);
    }
}

/* Compute a^(p-2) = a^(2^255 - 21) = (a^(2^252-3))^8 * a^3.
 * Self-aliasing safe: r may be the same object as a. */
static void fe_invert(field* r, const field* a) {
    field base; memcpy(&base, a, sizeof(field));
    fe_pow22523(r, a);
    fe_sq(r, r);
    fe_sq(r, r);
    fe_sq(r, r);
    fe_mul(r, r, &base);
    fe_mul(r, r, &base);
    fe_mul(r, r, &base);
}

/* Decode Edwards point from 32 bytes: Y in bytes, X parity in top bit of byte[31] */
static int point_from_bytes(point* p, const uint8_t b[32]) {
    fe_from_bytes(&p->Y, b);
    memset(&p->Z, 0, sizeof(field));
    p->Z.v[0] = 1;
    field u, v, v3, vxx, check;
    fe_sq(&u, &p->Y);
    fe_sub(&u, &u, &p->Z);  /* u = Y^2 - 1 */
    fe_sq(&v, &p->Y); fe_mul(&v, &v, &D); fe_add(&v, &v, &p->Z);  /* v = d*Y^2 + 1 */
    fe_sq(&v3, &v); fe_mul(&v3, &v3, &v);     /* v3 = v^3 */
    fe_sq(&p->X, &v3); fe_mul(&p->X, &p->X, &v);
    fe_mul(&p->X, &p->X, &u);                  /* X = u * v^7 */
    fe_pow22523(&p->X, &p->X);                 /* X = (u * v^7)^((p-5)/8) */
    fe_mul(&p->X, &p->X, &v3); fe_mul(&p->X, &p->X, &u);  /* X = u * v^3 * (u*v^7)^((p-5)/8) */
    fe_sq(&vxx, &p->X); fe_mul(&vxx, &vxx, &v);
    fe_sub(&check, &vxx, &u);
    uint8_t ck[32]; fe_to_bytes(ck, &check);
    if (!SNEPPX_ct_is_zero(ck, 32)) {
        fe_add(&check, &vxx, &u); fe_to_bytes(ck, &check);
        if (!SNEPPX_ct_is_zero(ck, 32)) { return -1; }
        fe_mul(&p->X, &p->X, &SQRTM1);
    }
    if ((p->X.v[0] & 1) != (b[31] >> 7)) fe_neg(&p->X, &p->X);
    fe_mul(&p->T, &p->X, &p->Y);
    return 0;
}

static int point_is_on_curve(const point* p);

static point B;

static void init_base_point(void) {
    static int initialized = 0;
    if (initialized) return;
    B.Z.v[0] = 1;
    B.Y.v[0] = 0x0006666666666658ULL; B.Y.v[1] = 0x0004CCCCCCCCCCCCULL;
    B.Y.v[2] = 0x0001999999999999ULL; B.Y.v[3] = 0x0003333333333333ULL;
    B.Y.v[4] = 0x0006666666666666ULL;
    /* Compute X from Y using sqrt formula */
    { field u, v, v3, vxx, check;
      fe_sq(&u, &B.Y); fe_sub(&u, &u, &B.Z);
      fe_sq(&v, &B.Y); fe_mul(&v, &v, &D); fe_add(&v, &v, &B.Z);
      fe_sq(&v3, &v); fe_mul(&v3, &v3, &v);
      fe_sq(&B.X, &v3); fe_mul(&B.X, &B.X, &v); fe_mul(&B.X, &B.X, &u);
      fe_pow22523(&B.X, &B.X);
      fe_mul(&B.X, &B.X, &v3); fe_mul(&B.X, &B.X, &u);
      fe_sq(&vxx, &B.X); fe_mul(&vxx, &vxx, &v);
      fe_sub(&check, &vxx, &u);
      uint8_t ck[32]; fe_to_bytes(ck, &check);
      if (!SNEPPX_ct_is_zero(ck, 32)) {
          fe_add(&check, &vxx, &u); fe_to_bytes(ck, &check);
          if (!SNEPPX_ct_is_zero(ck, 32)) { return; }
          fe_mul(&B.X, &B.X, &SQRTM1);
      }
    }
    if (B.X.v[0] & 1) fe_neg(&B.X, &B.X);
    fe_mul(&B.T, &B.X, &B.Y);
    initialized = 1;
}

static void point_add(point* r, const point* p, const point* q) {
    field a, b, c, d, e, f, g, h;
    fe_sub(&a, &p->Y, &p->X); fe_sub(&b, &q->Y, &q->X); fe_mul(&a, &a, &b);
    fe_add(&c, &p->Y, &p->X); fe_add(&d, &q->Y, &q->X); fe_mul(&c, &c, &d);
    fe_mul(&e, &p->T, &q->T); fe_mul(&e, &e, &D2);
    fe_mul(&f, &p->Z, &q->Z); fe_add(&f, &f, &f);
    fe_add(&g, &c, &a); fe_sub(&h, &c, &a);
    fe_sub(&c, &f, &e); fe_add(&a, &f, &e);
    fe_mul(&r->X, &h, &c); fe_mul(&r->Y, &a, &g);
    fe_mul(&r->T, &h, &g); fe_mul(&r->Z, &c, &a);
}

static void point_double(point* r, const point* p) {
    field a, b, c, d, e, f, g, h, t;
    fe_sub(&a, &p->Y, &p->X); fe_sq(&a, &a);  /* A = (Y-X)^2 */
    fe_add(&b, &p->Y, &p->X); fe_sq(&b, &b);  /* B = (Y+X)^2 */
    fe_sq(&c, &p->T); fe_mul(&c, &c, &D2);    /* C = 2*d*T^2 */
    fe_sq(&d, &p->Z); fe_add(&d, &d, &d);     /* D = 2*Z^2 */
    fe_sub(&e, &b, &a);  /* E = B-A = 4*X*Y */
    fe_sub(&f, &d, &c);  /* F = D-C = 2*(Z^2 - d*T^2) */
    fe_add(&g, &d, &c);  /* G = D+C = 2*(Z^2 + d*T^2) */
    fe_add(&h, &b, &a);  /* H = B+A = 2*(Y^2+X^2) */
    fe_mul(&r->X, &e, &f);
    fe_mul(&r->Y, &g, &h);
    fe_mul(&r->T, &e, &h);
    fe_mul(&r->Z, &f, &g);
    /* Identity guard: if X == 0 and Y == Z, force output to (0,1,1,0) */
    { uint8_t xb[32], yzb[32]; fe_to_bytes(xb, &p->X);
      fe_sub(&t, &p->Y, &p->Z); fe_to_bytes(yzb, &t);
      uint64_t idm = (uint64_t)(-(int)(SNEPPX_ct_is_zero(xb, 32) && SNEPPX_ct_is_zero(yzb, 32)));
      for (int i = 0; i < 5; i++) {
          r->X.v[i] &= ~idm; r->Y.v[i] = (r->Y.v[i] & ~idm) | (idm & 1ULL);
          r->Z.v[i] = (r->Z.v[i] & ~idm) | (idm & 1ULL); r->T.v[i] &= ~idm;
      }
    }
}

static int point_is_on_curve(const point* p) {
    field lhs, v;
    fe_sq(&lhs, &p->Y); fe_sq(&v, &p->X); fe_sub(&lhs, &lhs, &v);
    fe_sq(&v, &p->Z); fe_sub(&lhs, &lhs, &v);
    fe_sq(&v, &p->T); fe_mul(&v, &v, &D); fe_sub(&lhs, &lhs, &v);
    uint8_t b[32]; fe_to_bytes(b, &lhs);
    return SNEPPX_ct_is_zero(b, 32);
}

static void point_scalar_mult(point* r, const uint8_t* scalar, size_t sc_len, const point* base) {
    point_set_neutral(r);
    for (int i = (int)sc_len * 8 - 1; i >= 0; i--) {
        point_double(r, r);
        uint8_t bit = (scalar[i / 8] >> (i % 8)) & 1;
        point t; memcpy(&t, r, sizeof(t));
        point_add(&t, &t, base);
        uint64_t mask = (uint64_t)(-(int)bit);
        for (int j = 0; j < 5; j++) {
            r->X.v[j] ^= mask & (r->X.v[j] ^ t.X.v[j]);
            r->Y.v[j] ^= mask & (r->Y.v[j] ^ t.Y.v[j]);
            r->Z.v[j] ^= mask & (r->Z.v[j] ^ t.Z.v[j]);
            r->T.v[j] ^= mask & (r->T.v[j] ^ t.T.v[j]);
        }
    }
}

static void point_normalize(point* p) {
    field zinv;
    fe_invert(&zinv, &p->Z);
    fe_mul(&p->X, &p->X, &zinv);
    fe_mul(&p->Y, &p->Y, &zinv);
    memset(&p->Z, 0, sizeof(field));
    p->Z.v[0] = 1;
    fe_mul(&p->T, &p->X, &p->Y);
}

static void clamp_scalar(uint8_t* s) {
    s[0] &= 248; s[31] &= 127; s[31] |= 64;
}

static const uint8_t L_BYTES[32] = {
    0xed,0xd3,0xf5,0x5c,0x1a,0x63,0x12,0x58,
    0xd6,0x9c,0xf7,0xa2,0xde,0xf9,0xde,0x14,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x10
};

/* Subtract L << (8*shift) from t[64]; return 0 if result >= 0, 1 if borrow */
static int sub_sc_shifted(uint8_t t[64], int shift) {
    uint16_t borrow = 0;
    for (int i = 0; i < 64; i++) {
        uint16_t sub = (i >= shift && i < shift + 32) ? (uint16_t)L_BYTES[i - shift] : 0;
        uint16_t diff = (uint16_t)t[i] - sub - borrow;
        t[i] = (uint8_t)(diff & 0xFF);
        borrow = (diff >> 8) & 1;
    }
    return (int)borrow;
}

/* Reduce a 64-byte value mod L to 32 bytes by repeatedly subtracting L << (8*k) */
static void sc_reduce64(uint8_t r[32], const uint8_t a[64]) {
    uint8_t t[64]; memcpy(t, a, 64);
    /* Find highest non-zero byte in t */
    int hi = 63;
    while (hi >= 0 && t[hi] == 0) hi--;
    if (hi < 0) { memset(r, 0, 32); return; }
    /* Subtract L << (8*k) for k from (hi-31) down to 0 */
    for (int k = hi - 31; k >= 0; k--) {
        while (1) {
            /* Check if t >= L << (8*k) by comparing bytes from hi down to k */
            int ge = 0;
            for (int i = hi; i >= k + 32; i--) { if (t[i]) { ge = 1; break; } }
            if (!ge) {
                /* Compare the overlapping 32-byte window */
                int cmp = 0;
                for (int i = k + 31; i >= k && cmp == 0; i--) {
                    uint8_t tv = t[i];
                    uint8_t lv = L_BYTES[i - k];
                    if (tv > lv) { cmp = 1; } else if (tv < lv) { cmp = -1; }
                }
                if (cmp < 0) break; /* t < L << (8*k) */
            }
            if (sub_sc_shifted(t, k)) break; /* borrow = overshot */
        }
    }
    /* Final subtraction of L (shift 0) up to 4 times */
    for (int iter = 0; iter < 4; iter++) {
        int cmp = 0;
        for (int i = 31; i >= 0 && cmp == 0; i--) {
            if (t[i] > L_BYTES[i]) cmp = 1; else if (t[i] < L_BYTES[i]) cmp = -1;
        }
        if (cmp < 0) break;
        sub_sc_shifted(t, 0);
    }
    memcpy(r, t, 32);
}

/* Multiply two 32-byte scalars -> 64-byte product (schoolbook, 16-bit limbs) */
static void sc_mul256(uint8_t p[64], const uint8_t a[32], const uint8_t b[32]) {
    uint32_t t[64] = {0};
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            t[i + j] += (uint32_t)a[i] * (uint32_t)b[j];
    uint64_t carry = 0;
    for (int i = 0; i < 64; i++) {
        carry += t[i];
        p[i] = (uint8_t)(carry & 0xFF);
        carry >>= 8;
    }
}

/**
 * @brief Perform Ed25519 Secret Key Expand.
 *
 * @param expanded_sk [out] Expanded Sk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed) {
    if (!seed || !expanded_sk) return -1;
    uint8_t hash[64];
    SNEPPX_sha512(seed, 32, hash);
    memcpy(expanded_sk, hash, 64);
    clamp_scalar(expanded_sk);
    return 0;
}

/**
 * @brief Perform Ed25519 Keypair Generate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_keypair_generate(SNEPPXEd25519Keypair* kp) {
    if (!kp) return -1;
    init_base_point();
    memset(kp, 0, sizeof(SNEPPXEd25519Keypair));
    uint8_t seed[32];
    if (SNEPPX_random_bytes(seed, 32) != 0) return -1;
    if (SNEPPX_ed25519_secret_key_expand(kp->private_key, seed) != 0) return -1;
    point pub; point_scalar_mult(&pub, kp->private_key, 32, &B);
    point_normalize(&pub);
    fe_to_bytes(kp->public_key, &pub.Y);
    kp->public_key[31] |= (uint8_t)((pub.X.v[0] & 1) << 7);
    return 0;
}

/**
 * @brief Derive the Ed25519 public key from a 32-byte seed.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_publickey_from_seed(const uint8_t* seed, uint8_t* public_key) {
    if (!seed || !public_key) return -1;
    init_base_point();
    uint8_t expanded[64];
    if (SNEPPX_ed25519_secret_key_expand(expanded, seed) != 0) return -1;
    point pub; point_scalar_mult(&pub, expanded, 32, &B);
    point_normalize(&pub);
    fe_to_bytes(public_key, &pub.Y);
    public_key[31] |= (uint8_t)((pub.X.v[0] & 1) << 7);
    return 0;
}

/**
 * @brief Sign Ed25519.
 *
 * @param kp [in] Kp value.
 * @param message [in] Message value.
 * @param msg_len [in] Msg Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_sign(const SNEPPXEd25519Keypair* kp, const uint8_t* message, size_t msg_len, SNEPPXEd25519Signature* sig) {
    if (!kp || !message || !sig) return -1;
    init_base_point();
    uint8_t r_seed[64], r_scalar[32], h_scalar[32], hram[64], product[64], sum[64];
    /* r = SHA-512(nonce_seed || message) */
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, kp->private_key + 32, 32);
    SNEPPX_sha512_update(&ctx, message, msg_len);
    SNEPPX_sha512_finish(&ctx, r_seed);
    sc_reduce64(r_scalar, r_seed);
    /* R = r_scalar * B */
    point R; point_scalar_mult(&R, r_scalar, 32, &B);
    point_normalize(&R);
    fe_to_bytes(sig->data, &R.Y);
    sig->data[31] |= (uint8_t)((R.X.v[0] & 1) << 7);
    /* h = SHA-512(R || A || message) */
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, sig->data, 32);
    SNEPPX_sha512_update(&ctx, kp->public_key, 32);
    SNEPPX_sha512_update(&ctx, message, msg_len);
    SNEPPX_sha512_finish(&ctx, hram);
    /* Reduce full 64-byte challenge hash mod L */
    sc_reduce64(h_scalar, hram);
    /* S = (r + h*a) mod L */
    /* a is the clamped private key (kp->private_key[0:32]) */
    sc_mul256(product, h_scalar, kp->private_key);
    /* product + r_scalar as 64-byte sum */
    uint32_t carry = 0;
    for (int i = 0; i < 32; i++) {
        carry = (uint32_t)product[i] + (uint32_t)r_scalar[i] + carry;
        sum[i] = (uint8_t)(carry & 0xFF);
        carry >>= 8;
    }
    for (int i = 32; i < 64; i++) { carry += product[i]; sum[i] = (uint8_t)(carry & 0xFF); carry >>= 8; }
    /* Reduce sum mod L to get S */
    sc_reduce64(sig->data + 32, sum);
    return 0;
}

/**
 * @brief Core Ed25519 verify with selectable hash path.
 *
 * @param public_key [in] 32-byte public key.
 * @param message [in]    Message bytes.
 * @param msg_len [in]    Message length.
 * @param sig [in]        64-byte signature struct.
 * @param legacy [in]     If non-zero, use the pre-fix (legacy) SHA-512 digest.
 *
 * @return 1 on valid signature, 0 on invalid, -1 on error.
 */
static int ed25519_verify_core(const uint8_t* public_key, const uint8_t* message, size_t msg_len,
                               const SNEPPXEd25519Signature* sig, int legacy) {
    if (!public_key || !message || !sig) return -1;
    init_base_point();
    uint8_t hram[64];
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, sig->data, 32);
    SNEPPX_sha512_update(&ctx, public_key, 32);
    SNEPPX_sha512_update(&ctx, message, msg_len);
    if (legacy)
        SNEPPX_sha512_legacy_finish(&ctx, hram);
    else
        SNEPPX_sha512_finish(&ctx, hram);
    point A; int ret_A = point_from_bytes(&A, public_key);
    if (ret_A != 0) return -1;
    point R;
    int ret_R = point_from_bytes(&R, sig->data);
    if (ret_R != 0) return -1;

    /* Verify: [S]B = R + [h]A */
    uint8_t h_scalar[32];
    sc_reduce64(h_scalar, hram);
    point S_point;
    point_scalar_mult(&S_point, sig->data + 32, 32, &B);
    
    point hA;
    point_scalar_mult(&hA, h_scalar, 32, &A);
    
    point_add(&hA, &R, &hA);
    
    point_normalize(&S_point);
    point_normalize(&hA);
    
    /* Compare in affine coordinates (canonical bytes) */
    uint8_t bx[32], cx[32], by[32], cy[32];
    fe_to_bytes(bx, &S_point.X); fe_to_bytes(cx, &hA.X);
    fe_to_bytes(by, &S_point.Y); fe_to_bytes(cy, &hA.Y);
    return SNEPPX_ct_equal(bx, cx, 32) && SNEPPX_ct_equal(by, cy, 32);
}

/**
 * @brief Verify Ed25519.
 *
 * @param public_key [in] Public Key value.
 * @param message [in] Message value.
 * @param msg_len [in] Msg Len value.
 *
 * @return 1 on valid, 0 on invalid, -1 on error.
 */
int SNEPPX_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const SNEPPXEd25519Signature* sig) {
    return ed25519_verify_core(public_key, message, msg_len, sig, 0);
}

/**
 * @brief Verify Ed25519 with optional legacy-hash fallback.
 *
 * First attempts verification with the RFC-8032 compliant SHA-512.  If
 * that fails with an invalid-signature result (0) and *allow_legacy* is
 * non-zero, recomputes the challenge hash using the legacy (pre-fix)
 * SHA-512 digest and retries.  This allows manifests signed before the
 * SHA-512 length-encoding fix to still be verified.
 *
 * @param sig [in]          64-byte raw signature (R || S).
 * @param m [in]            Message bytes.
 * @param mlen [in]         Message length.
 * @param pk [in]           32-byte public key.
 * @param allow_legacy [in] If non-zero, fall back to legacy SHA-512 on failure.
 *
 * @return 1 on valid, 0 on invalid, -1 on error.
 */
int SNEPPX_ed25519_verify_compat(const uint8_t* sig, const uint8_t* m, size_t mlen, const uint8_t* pk, int allow_legacy) {
    if (!sig || !m || !pk) return -1;
    SNEPPXEd25519Signature sig_struct;
    memcpy(sig_struct.data, sig, SNEPPX_ED25519_SIGNATURE_LEN);

    /* RFC-correct verify first */
    int ret = ed25519_verify_core(pk, m, mlen, &sig_struct, 0);
    if (ret == 1) return 1;
    /* On hard error (-1), do not retry with legacy */
    if (ret == -1 || !allow_legacy) return ret;
    /* Legacy fallback: re-derive challenge with old H1 digest */
    return ed25519_verify_core(pk, m, mlen, &sig_struct, 1);
}

/**
 * @brief Perform Ed25519 Scalar Multiply.
 *
 * @param result [out] 32-byte compressed result point.
 * @param scalar [in] 32-byte scalar value.
 * @param point [in] 32-byte compressed input point.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* pt) {
    if (!result || !scalar || !pt) return -1;
    init_base_point();
    point P;
    if (point_from_bytes(&P, pt) != 0) return -1;
    point R;
    point_scalar_mult(&R, scalar, 32, &P);
    point_normalize(&R);
    fe_to_bytes(result, &R.Y);
    result[31] |= (uint8_t)((R.X.v[0] & 1) << 7);
    return 0;
}
