#include "polynomial_authentication_mac.h"
#include <string.h>

#define MASK26 0x3ffffffULL
/*
 * SNEPPX - Poly1305 Message Authentication Code
 *
 * WHAT
 *   Poly1305 Message Authentication Code.
 *
 * CONCEPT
 *   RFC 8439 Poly1305-AES MAC: 128-bit tag from a 32-byte one-time key and message.
 *
 * ROLE
 *   Layer S2 authenticated encryption - the MAC half of ChaCha20-Poly1305.
 *
 * REFERENCES
 *   RFC 8439 (ChaCha20-Poly1305), Daniel Bernstein Poly1305-AES.
 */



static void poly1305_block(SNEPPXPoly1305State *s, const uint8_t *m, unsigned int hibit) {
    uint64_t h0 = s->h[0], h1 = s->h[1], h2 = s->h[2], h3 = s->h[3], h4 = s->h[4];
    uint64_t r0 = s->r[0], r1 = s->r[1], r2 = s->r[2], r3 = s->r[3], r4 = s->r[4];
    uint64_t d0, d1, d2, d3, d4, c;

    uint64_t b_lo = (uint64_t)m[0] | (uint64_t)m[1]<<8 | (uint64_t)m[2]<<16 | (uint64_t)m[3]<<24 |
                    (uint64_t)m[4]<<32 | (uint64_t)m[5]<<40 | (uint64_t)m[6]<<48 | (uint64_t)m[7]<<56;
    uint64_t b_hi = (uint64_t)m[8] | (uint64_t)m[9]<<8 | (uint64_t)m[10]<<16 | (uint64_t)m[11]<<24 |
                    (uint64_t)m[12]<<32 | (uint64_t)m[13]<<40 | (uint64_t)m[14]<<48 | (uint64_t)m[15]<<56;

    h0 += b_lo & MASK26;
    h1 += (b_lo >> 26) & MASK26;
    h2 += ((b_lo >> 52) | (b_hi << 12)) & MASK26;
    h3 += (b_hi >> 14) & MASK26;
    h4 += (b_hi >> 40) & MASK26;
    h4 += hibit;

    d0 = h0*r0 + h1*5*r4 + h2*5*r3 + h3*5*r2 + h4*5*r1;
    d1 = h0*r1 + h1*r0   + h2*5*r4 + h3*5*r3 + h4*5*r2;
    d2 = h0*r2 + h1*r1   + h2*r0   + h3*5*r4 + h4*5*r3;
    d3 = h0*r3 + h1*r2   + h2*r1   + h3*r0   + h4*5*r4;
    d4 = h0*r4 + h1*r3   + h2*r2   + h3*r1   + h4*r0;

    c = d0 >> 26; h0 = d0 & MASK26; d1 += c;
    c = d1 >> 26; h1 = d1 & MASK26; d2 += c;
    c = d2 >> 26; h2 = d2 & MASK26; d3 += c;
    c = d3 >> 26; h3 = d3 & MASK26; d4 += c;
    c = d4 >> 26; h4 = d4 & MASK26; h0 += c * 5;
    c = h0 >> 26; h0 &= MASK26; h1 += c;

    s->h[0] = h0; s->h[1] = h1; s->h[2] = h2; s->h[3] = h3; s->h[4] = h4;
}

/**
 * @brief Initialize Poly1305.
 *
 * @param s [out] S value.
 */
void SNEPPX_poly1305_init(SNEPPXPoly1305State *s, const uint8_t key[32]) {
    uint64_t r_lo, r_hi;
    uint8_t r[16];
    memcpy(r, key, 16);

    /* clamp: r &= 0x0ffffffc0ffffffc0ffffffc0fffffff */
    r[3] &= 0x0f; r[7] &= 0x0f; r[11] &= 0x0f; r[15] &= 0x0f;
    r[4] &= 0xfc; r[8] &= 0xfc; r[12] &= 0xfc;

    r_lo = (uint64_t)r[0] | (uint64_t)r[1]<<8 | (uint64_t)r[2]<<16 | (uint64_t)r[3]<<24 |
           (uint64_t)r[4]<<32 | (uint64_t)r[5]<<40 | (uint64_t)r[6]<<48 | (uint64_t)r[7]<<56;
    r_hi = (uint64_t)r[8] | (uint64_t)r[9]<<8 | (uint64_t)r[10]<<16 | (uint64_t)r[11]<<24 |
           (uint64_t)r[12]<<32 | (uint64_t)r[13]<<40 | (uint64_t)r[14]<<48 | (uint64_t)r[15]<<56;

    s->r[0] = r_lo & MASK26;
    s->r[1] = (r_lo >> 26) & MASK26;
    s->r[2] = ((r_lo >> 52) | (r_hi << 12)) & MASK26;
    s->r[3] = (r_hi >> 14) & MASK26;
    s->r[4] = (r_hi >> 40) & MASK26;

    s->s[0] = (uint64_t)key[16] | (uint64_t)key[17]<<8 | (uint64_t)key[18]<<16 | (uint64_t)key[19]<<24 |
              (uint64_t)key[20]<<32 | (uint64_t)key[21]<<40 | (uint64_t)key[22]<<48 | (uint64_t)key[23]<<56;
    s->s[1] = (uint64_t)key[24] | (uint64_t)key[25]<<8 | (uint64_t)key[26]<<16 | (uint64_t)key[27]<<24 |
              (uint64_t)key[28]<<32 | (uint64_t)key[29]<<40 | (uint64_t)key[30]<<48 | (uint64_t)key[31]<<56;

    s->h[0] = s->h[1] = s->h[2] = s->h[3] = s->h[4] = 0;
    s->buflen = 0;
}

/**
 * @brief Update Poly1305.
 *
 * @param s [out] S value.
 * @param data [in] Data value.
 */
void SNEPPX_poly1305_update(SNEPPXPoly1305State *s, const uint8_t *data, size_t len) {
    while (len) {
        size_t take = 16 - s->buflen;
        if (take > len) take = len;
        memcpy(s->buf + s->buflen, data, take);
        s->buflen += (unsigned int)take; data += take; len -= take;
        if (s->buflen == 16) {
            poly1305_block(s, s->buf, 1 << 24);
            s->buflen = 0;
        }
    }
}

/**
 * @brief Perform Poly1305 Finish.
 *
 * @param s [out] S value.
 */
void SNEPPX_poly1305_finish(SNEPPXPoly1305State *s, uint8_t mac[16]) {
    if (s->buflen) {
        memset(s->buf + s->buflen, 0, 16 - s->buflen);
        s->buf[s->buflen] = 1;
        poly1305_block(s, s->buf, 0);
    }

    uint64_t h0 = s->h[0], h1 = s->h[1], h2 = s->h[2], h3 = s->h[3], h4 = s->h[4], c;

    c = h0 >> 26; h0 &= MASK26; h1 += c;
    c = h1 >> 26; h1 &= MASK26; h2 += c;
    c = h2 >> 26; h2 &= MASK26; h3 += c;
    c = h3 >> 26; h3 &= MASK26; h4 += c;
    c = h4 >> 26; h4 &= MASK26; h0 += c * 5;
    c = h0 >> 26; h0 &= MASK26; h1 += c;

    uint64_t t0 = h0 - 0x3ffffffbULL; uint64_t b = (t0 > h0) ? 1ULL : 0ULL;
    uint64_t t1 = h1 - 0x3ffffffULL - b; b = (t1 > h1) ? 1ULL : 0ULL;
    uint64_t t2 = h2 - 0x3ffffffULL - b; b = (t2 > h2) ? 1ULL : 0ULL;
    uint64_t t3 = h3 - 0x3ffffffULL - b; b = (t3 > h3) ? 1ULL : 0ULL;
    uint64_t t4 = h4 - 0x3ffffffULL - b; b = (t4 > h4) ? 1ULL : 0ULL;
    uint64_t mask = 0ULL - b;
    h0 = (t0 & ~mask) | (h0 & mask);
    h1 = (t1 & ~mask) | (h1 & mask);
    h2 = (t2 & ~mask) | (h2 & mask);
    h3 = (t3 & ~mask) | (h3 & mask);
    h4 = (t4 & ~mask) | (h4 & mask);

    uint64_t h_lo = h0 | (h1 << 26) | ((h2 & 0xfff) << 52);
    uint64_t h_hi = (h2 >> 12) | (h3 << 14) | ((h4 & 0xffffff) << 40);
    uint64_t old_lo = h_lo;
    h_lo += s->s[0];
    h_hi += s->s[1] + (uint64_t)(h_lo < old_lo);

    mac[0] = (uint8_t)h_lo; mac[1] = (uint8_t)(h_lo >> 8); mac[2] = (uint8_t)(h_lo >> 16); mac[3] = (uint8_t)(h_lo >> 24);
    mac[4] = (uint8_t)(h_lo >> 32); mac[5] = (uint8_t)(h_lo >> 40); mac[6] = (uint8_t)(h_lo >> 48); mac[7] = (uint8_t)(h_lo >> 56);
    mac[8] = (uint8_t)h_hi; mac[9] = (uint8_t)(h_hi >> 8); mac[10] = (uint8_t)(h_hi >> 16); mac[11] = (uint8_t)(h_hi >> 24);
    mac[12] = (uint8_t)(h_hi >> 32); mac[13] = (uint8_t)(h_hi >> 40); mac[14] = (uint8_t)(h_hi >> 48); mac[15] = (uint8_t)(h_hi >> 56);
}
