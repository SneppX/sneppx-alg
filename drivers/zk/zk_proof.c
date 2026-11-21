#include "zk_proof.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef SNEPPX_BUILD_ZK

/*
 * SNEPPX - Zk Proof
 *
 * WHAT
 *   Zk Proof.
 *
 * CONCEPT
 *   Provides the Zk Proof.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ------------------------------------------------------------------ */
/* Embedded SHA-256 (self-contained, FIPS 180-4)                       */
/* ------------------------------------------------------------------ */
static const uint32_t SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t total = len + 1;
    uint64_t bitlen = (uint64_t)len * 8;
    size_t padded = (len + 9 + 63) & ~(size_t)63;
    uint8_t* buf = (uint8_t*)malloc(padded ? padded : 64);
    if (!buf) { memset(out, 0, 32); return; }
    memset(buf, 0, padded);
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    for (int i = 0; i < 8; i++)
        buf[padded - 1 - i] = (uint8_t)(bitlen >> (i * 8));
    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)buf[off+4*i]<<24)|((uint32_t)buf[off+4*i+1]<<16)|((uint32_t)buf[off+4*i+2]<<8)|((uint32_t)buf[off+4*i+3]);
        for (int i = 16; i < 64; i++)
            w[i] = w[i-16] + (rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3))
                 + w[i-7] + (rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10));
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh + S1 + ch + SHA_K[i] + w[i];
            uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(h[i]>>24);
        out[4*i+1] = (uint8_t)(h[i]>>16);
        out[4*i+2] = (uint8_t)(h[i]>>8);
        out[4*i+3] = (uint8_t)(h[i]);
    }
    free(buf);
}

/* ------------------------------------------------------------------ */
/* 256-bit bignum (8 x uint32, little-endian)                          */
/* ------------------------------------------------------------------ */
#define BNW 8
typedef uint32_t bn_t[BNW];

static void bn_zero(bn_t x) { memset(x, 0, sizeof(bn_t)); }
static void bn_set_u32(bn_t x, uint32_t v) { x[0]=v; memset(x+1,0,(BNW-1)*4); }
static void bn_load(const uint8_t* b, bn_t x) {
    for (int i = 0; i < BNW; i++)
        x[i] = ((uint32_t)b[4*i]) | ((uint32_t)b[4*i+1]<<8) | ((uint32_t)b[4*i+2]<<16) | ((uint32_t)b[4*i+3]<<24);
}
static void bn_store(const bn_t x, uint8_t* b) {
    for (int i = 0; i < BNW; i++) {
        b[4*i]=(uint8_t)x[i]; b[4*i+1]=(uint8_t)(x[i]>>8); b[4*i+2]=(uint8_t)(x[i]>>16); b[4*i+3]=(uint8_t)(x[i]>>24);
    }
}
static int bn_cmp(const bn_t a, const bn_t b) {
    for (int i = BNW-1; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    }
    return 0;
}
/* returns carry */
static uint32_t bn_add(bn_t d, const bn_t a, const bn_t b) {
    uint64_t c = 0;
    for (int i = 0; i < BNW; i++) {
        uint64_t s = (uint64_t)a[i] + b[i] + c;
        d[i] = (uint32_t)s; c = s >> 32;
    }
    return (uint32_t)c;
}
/* assumes a >= b */
static void bn_sub(bn_t d, const bn_t a, const bn_t b) {
    uint64_t borrow = 0;
    for (int i = 0; i < BNW; i++) {
        uint64_t s = (uint64_t)a[i] - b[i] - borrow;
        d[i] = (uint32_t)s;
        borrow = (s >> 63) & 1;
    }
}

static void mul256(const bn_t a, const bn_t b, uint32_t p[16]) {
    memset(p, 0, sizeof(uint32_t)*16);
    for (int i = 0; i < BNW; i++) {
        uint64_t k = 0;
        for (int j = 0; j < BNW; j++) {
            uint64_t cur = (uint64_t)a[i] * b[j] + p[i+j] + k;
            p[i+j] = (uint32_t)cur;
            k = cur >> 32;
        }
        int idx = i + BNW;
        uint64_t carry = k;
        while (carry && idx < 16) {
            uint64_t s = (uint64_t)p[idx] + carry;
            p[idx] = (uint32_t)s;
            carry = s >> 32;
            idx++;
        }
    }
}

static int hi_bit(const uint32_t* a, int n) {
    for (int i = n-1; i >= 0; i--)
        if (a[i]) {
            int b = 31;
            while (!((a[i] >> b) & 1)) b--;
            return i*32 + b;
        }
    return -1;
}
static int ucmp(const uint32_t* a, int na, const uint32_t* b, int nb) {
    int ia = hi_bit(a, na), ib = hi_bit(b, nb);
    if (ia != ib) return ia > ib ? 1 : -1;
    if (ia < 0) return 0;
    for (int i = ia/32; i >= 0; i--)
        if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    return 0;
}
/* a (n words) -= b (nb words), a >= b */
static void usub(uint32_t* a, int n, const uint32_t* b, int nb) {
    uint64_t borrow = 0;
    for (int i = 0; i < n; i++) {
        uint64_t sb = (i < nb) ? b[i] : 0;
        uint64_t s = (uint64_t)a[i] - sb - borrow;
        a[i] = (uint32_t)s;
        borrow = (s >> 63) & 1;
    }
}

/* x (16 words) mod m (BNW words) -> r (BNW words) */
static void bn_mod(const uint32_t* x, const bn_t m, bn_t r) {
    uint32_t a[16];
    memcpy(a, x, sizeof(a));
    int bl = hi_bit(m, BNW);
    int al = hi_bit(a, 16);
    if (bl < 0) { bn_zero(r); return; }
    if (al < 0) { bn_zero(r); return; }
    for (int s = al - bl; s >= 0; s--) {
        uint32_t t[16] = {0};
        int ws = s/32, bs = s%32;
        for (int i = 0; i < BNW; i++) {
            int di = i + ws;
            if (di < 16) t[di] |= m[i] << bs;
            if (bs && di + 1 < 16) t[di+1] |= m[i] >> (32 - bs);
        }
        if (ucmp(a, 16, t, 16) >= 0) usub(a, 16, t, 16);
    }
    memcpy(r, a, sizeof(bn_t));
}

static void modmul(const bn_t a, const bn_t b, const bn_t m, bn_t r) {
    uint32_t p[16];
    mul256(a, b, p);
    bn_mod(p, m, r);
}
static void modadd(const bn_t a, const bn_t b, const bn_t m, bn_t r) {
    uint32_t t[16];
    memset(t, 0, sizeof(t));
    memcpy(t, a, sizeof(bn_t));
    uint64_t carry = 0;
    for (int i = 0; i < BNW; i++) {
        uint64_t s = (uint64_t)t[i] + b[i] + carry;
        t[i] = (uint32_t)s; carry = s >> 32;
    }
    t[BNW] = (uint32_t)carry; /* at most 1 */
    bn_mod(t, m, r);
}
static void modsub(const bn_t a, const bn_t b, const bn_t m, bn_t r) {
    if (bn_cmp(a, b) >= 0) {
        bn_sub(r, a, b);
    } else {
        bn_t d; bn_sub(d, b, a);   /* d = b - a (>= 0, < m) */
        bn_sub(r, m, d);           /* r = m - d = a - b + m */
    }
}

static void modexp(const bn_t base, const bn_t exp, const bn_t m, bn_t r) {
    bn_set_u32(r, 1);
    bn_t g; memcpy(g, base, sizeof(bn_t));
    int hb = hi_bit(exp, BNW);
    for (int i = hb; i >= 0; i--) {
        modmul(r, r, m, r);
        int bit = (exp[i/32] >> (i%32)) & 1;
        if (bit) modmul(r, g, m, r);
    }
}

/* group parameters: p = 2^255 - 19 (prime), order = p - 1, g = 2 */
static const bn_t ZK_P = {0xFFFFFFED,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0x7FFFFFFF};
static bn_t ZK_ORD;          /* p - 1 */
static const bn_t ZK_G = {2,0,0,0,0,0,0,0};

/* Self-contained xorshift128 RNG (demo-grade randomness for the nonce). */
static uint32_t g_rng[4] = {0x9E3779B9, 0x243F6A88, 0xB7E15162, 0x1B873593};
static void rng_seed(void) {
    uint32_t t = (uint32_t)time(NULL);
    g_rng[0] ^= t; g_rng[1] ^= t * 2654435761u + 1; g_rng[2] ^= t ^ 0x85EBCA6B; g_rng[3] ^= t * 40503 + 7;
}
static uint32_t rng_u32(void) {
    uint32_t s = g_rng[3];
    uint32_t r = g_rng[0];
    g_rng[3] = g_rng[2]; g_rng[2] = g_rng[1]; g_rng[1] = r;
    s ^= s << 11; s ^= s >> 8;
    r ^= s ^ (s >> 19) ^ (r >> 5);
    g_rng[0] = r;
    return r;
}

/**
 * @brief Initialize Zk.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_init(void) {
    rng_seed();
    bn_t one; bn_set_u32(one, 1);
    bn_sub(ZK_ORD, ZK_P, one);  /* order = p - 1 */
    return 0;
}
/**
 * @brief Perform Zk Shutdown.
 */
void SNEPPX_zk_shutdown(void) {}

static void bn_rand_mod(const bn_t m, bn_t r) {
    uint8_t buf[32];
    uint32_t x[16];
    for (;;) {
        for (int i = 0; i < 32; i += 4) { uint32_t v = rng_u32(); buf[i]=v; buf[i+1]=v>>8; buf[i+2]=v>>16; buf[i+3]=v>>24; }
        memset(x, 0, sizeof(x));
        bn_t tmp; bn_load(buf, tmp);
        memcpy(x, tmp, sizeof(bn_t));
        bn_mod(x, m, r);
        if (hi_bit(r, BNW) >= 0 && bn_cmp(r, m) < 0) return;
    }
}

/**
 * @brief Generate Zk.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [out] Pub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_keygen(const uint8_t* secret, size_t secret_len, uint8_t* pub, size_t* pub_len) {
    if (!secret || !pub || !pub_len) return -1;
    if (secret_len > 32) secret_len = 32;
    uint8_t sb[32] = {0};
    memcpy(sb, secret, secret_len);
    bn_t x; bn_load(sb, x);
    bn_t xr; uint32_t xw[16]; memset(xw,0,sizeof(xw)); memcpy(xw, x, sizeof(bn_t));
    bn_mod(xw, ZK_ORD, xr);
    bn_t y; modexp(ZK_G, xr, ZK_P, y);
    bn_store(y, pub);
    *pub_len = 32;
    return 0;
}

/**
 * @brief Perform Zk Prove.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [out] Proof value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_prove(const uint8_t* secret, size_t secret_len,
                    const uint8_t* pub, size_t pub_len,
                    const uint8_t* msg, size_t msg_len,
                    uint8_t* proof, size_t* proof_len) {
    (void)pub; (void)pub_len;
    if (!secret || !proof || !proof_len) return -1;
    if (secret_len > 32) secret_len = 32;
    uint8_t sb[32] = {0};
    memcpy(sb, secret, secret_len);
    bn_t x; bn_load(sb, x);
    uint32_t xw[16]; memset(xw,0,sizeof(xw)); memcpy(xw,x,sizeof(bn_t));
    bn_t xr; bn_mod(xw, ZK_ORD, xr);

    bn_t r; bn_rand_mod(ZK_ORD, r);
    bn_t R; modexp(ZK_G, r, ZK_P, R);
    uint8_t Rb[32]; bn_store(R, Rb);

    /* e = SHA256(R || pub || msg) mod order */
    size_t cap = 32 + (pub?pub_len:0) + msg_len;
    uint8_t* blob = (uint8_t*)malloc(cap ? cap : 1);
    size_t off = 0;
    memcpy(blob+off, Rb, 32); off += 32;
    if (pub) { memcpy(blob+off, pub, pub_len); off += pub_len; }
    if (msg) { memcpy(blob+off, msg, msg_len); off += msg_len; }
    uint8_t eh[32]; sha256(blob, off, eh);
    free(blob);
    bn_t e; uint32_t ew[16]; memset(ew,0,sizeof(ew)); bn_load(eh, e); memcpy(ew,e,sizeof(bn_t));
    bn_t er; bn_mod(ew, ZK_ORD, er);

    /* s = (r + e*x) mod order */
    bn_t ex; modmul(e, xr, ZK_ORD, ex);
    bn_t s; modadd(r, ex, ZK_ORD, s);

    uint8_t sb_out[32]; bn_store(s, sb_out);
    memcpy(proof, Rb, 32);
    memcpy(proof + 32, sb_out, 32);
    *proof_len = 64;
    return 0;
}

/**
 * @brief Verify Zk.
 *
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [in] Proof value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_verify(const uint8_t* pub, size_t pub_len,
                     const uint8_t* msg, size_t msg_len,
                     const uint8_t* proof, size_t proof_len) {
    if (!pub || !proof || proof_len != 64) return -1;
    (void)pub_len;
    bn_t R, s;
    bn_load(proof, R);
    bn_load(proof + 32, s);

    uint8_t Rb[32]; bn_store(R, Rb);
    size_t cap = 32 + pub_len + msg_len;
    uint8_t* blob = (uint8_t*)malloc(cap ? cap : 1);
    size_t off = 0;
    memcpy(blob+off, Rb, 32); off += 32;
    memcpy(blob+off, pub, pub_len); off += pub_len;
    if (msg) { memcpy(blob+off, msg, msg_len); off += msg_len; }
    uint8_t eh[32]; sha256(blob, off, eh);
    free(blob);
    bn_t e; uint32_t ew[16]; memset(ew,0,sizeof(ew)); bn_load(eh, e); memcpy(ew,e,sizeof(bn_t));
    bn_t er; bn_mod(ew, ZK_ORD, er);

    bn_t y; bn_load(pub, y);
    bn_t lhs; modexp(ZK_G, s, ZK_P, lhs);
    bn_t ye; modexp(y, er, ZK_P, ye);
    bn_t rhs; modmul(R, ye, ZK_P, rhs);
    return bn_cmp(lhs, rhs) == 0 ? 0 : -1;
}

#else /* !SNEPPX_BUILD_ZK — UNSUPPORTED stub */

/**
 * @brief Initialize Zk.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_init(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Zk Shutdown.
 */
void SNEPPX_zk_shutdown(void) {}
/**
 * @brief Generate Zk.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [out] Pub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_keygen(const uint8_t* secret, size_t secret_len, uint8_t* pub, size_t* pub_len) {
    (void)secret; (void)secret_len; (void)pub; (void)pub_len;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
/**
 * @brief Perform Zk Prove.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [out] Proof value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_prove(const uint8_t* secret, size_t secret_len, const uint8_t* pub, size_t pub_len,
                    const uint8_t* msg, size_t msg_len, uint8_t* proof, size_t* proof_len) {
    (void)secret;(void)secret_len;(void)pub;(void)pub_len;(void)msg;(void)msg_len;(void)proof;(void)proof_len;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
/**
 * @brief Verify Zk.
 *
 * @param pub [in] Pub value.
 * @param pub_len [in] Pub Len value.
 * @param msg [in] Msg value.
 * @param msg_len [in] Msg Len value.
 * @param proof [in] Proof value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_zk_verify(const uint8_t* pub, size_t pub_len, const uint8_t* msg, size_t msg_len,
                     const uint8_t* proof, size_t proof_len) {
    (void)pub;(void)pub_len;(void)msg;(void)msg_len;(void)proof;(void)proof_len;
    return SNEPPX_DRIVER_UNSUPPORTED;
}

#endif
