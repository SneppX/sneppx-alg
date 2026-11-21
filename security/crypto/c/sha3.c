#include "keccak_sha3_hashing.h"
#include <string.h>

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
/*
 * SNEPPX - SHA-3 / Keccak Cryptographic Hash
 *
 * WHAT
 *   SHA-3 / Keccak Cryptographic Hash.
 *
 * CONCEPT
 *   FIPS 202 SHA-3 using the Keccak sponge construction for 224/256/384/512-bit digests.
 *
 * ROLE
 *   Used by DRBG (Hash-DRBG), key-vault hashing, and general-purpose hashing.
 *
 * REFERENCES
 *   FIPS 202 (SHA-3 / Keccak).
 */



static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static void keccak_f1600(uint64_t state[25]) {
    static const int rho[5][5] = {
        { 0, 36,  3, 41, 18},
        { 1, 44, 10, 45,  2},
        {62,  6, 43, 15, 61},
        {28, 55, 25, 21, 56},
        {27, 20, 39,  8, 14}
    };
    for (int r = 0; r < 24; r++) {
        uint64_t C[5], D[5];
        for (int i = 0; i < 5; i++)
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        for (int i = 0; i < 5; i++)
            D[i] = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
        for (int i = 0; i < 25; i++) state[i] ^= D[i % 5];

        uint64_t B[25];
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                B[y + 5 * ((2 * x + 3 * y) % 5)] = ROTL64(state[x + 5 * y], rho[x][y]);
        for (int i = 0; i < 25; i++)
            state[i] = B[i] ^ (~B[(i + 1) % 5 + 5 * (i / 5)] & B[(i + 2) % 5 + 5 * (i / 5)]);
        state[0] ^= RC[r];
    }
}

static void sha3_absorb(SNEPPXSHA3State* s, const uint8_t* data, size_t len) {
    while (len) {
        size_t take = len < s->rate ? len : s->rate;
        for (size_t i = 0; i < take; i++)
            ((uint8_t*)s->state)[i] ^= data[i];
        data += take; len -= take;
        if (take == s->rate) keccak_f1600(s->state);
    }
}

/**
 * @brief Initialize Sha3 256.
 */
void SNEPPX_sha3_256_init(SNEPPXSHA3State* s) {
    memset(s, 0, sizeof(*s));
    s->rate = 136;
    s->capacity = 64;
    s->digest_size = 32;
}

/**
 * @brief Initialize Sha3 512.
 */
void SNEPPX_sha3_512_init(SNEPPXSHA3State* s) {
    memset(s, 0, sizeof(*s));
    s->rate = 72;
    s->capacity = 128;
    s->digest_size = 64;
}

/**
 * @brief Update Sha3.
 *
 * @param s [out] S value.
 * @param data [in] Data value.
 */
void SNEPPX_sha3_update(SNEPPXSHA3State* s, const uint8_t* data, size_t len) {
    if (s->buflen) {
        size_t fill = s->rate - s->buflen;
        if (len < fill) { memcpy(s->buffer + s->buflen, data, len); s->buflen += (unsigned int)len; return; }
        memcpy(s->buffer + s->buflen, data, fill);
        sha3_absorb(s, s->buffer, s->rate);
        data += fill; len -= fill; s->buflen = 0;
    }
    if (len >= s->rate) {
        size_t aligned = len - (len % s->rate);
        sha3_absorb(s, data, aligned);
        data += aligned; len -= aligned;
    }
    if (len) { memcpy(s->buffer, data, len); s->buflen = (unsigned int)len; }
}

/**
 * @brief Perform Sha3 Finish.
 *
 * @param s [out] S value.
 */
void SNEPPX_sha3_finish(SNEPPXSHA3State* s, uint8_t* hash) {
    uint8_t buf[200];
    memcpy(buf, s->buffer, s->buflen);
    buf[s->buflen] = 0x06;
    memset(buf + s->buflen + 1, 0, s->rate - s->buflen - 1);
    buf[s->rate - 1] |= 0x80;
    for (size_t i = 0; i < s->rate; i++) ((uint8_t*)s->state)[i] ^= buf[i];
    keccak_f1600(s->state);
    memcpy(hash, s->state, s->digest_size);
}
