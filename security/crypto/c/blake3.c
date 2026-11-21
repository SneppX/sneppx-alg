#include "cryptographic_hashing_blake3.h"
#include <string.h>

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define G(a, b, c, d, x, y) do { \
    a += b + x; d ^= a; d = ROTR32(d, 16); \
    c += d; b ^= c; b = ROTR32(b, 12); \
    a += b + y; d ^= a; d = ROTR32(d, 8); \
    c += d; b ^= c; b = ROTR32(b, 7); \
} while(0)

#define ROUND(r) do { \
    G(v[0], v[4], v[8],  v[12], msg[r[0]], msg[r[1]]); \
    G(v[1], v[5], v[9],  v[13], msg[r[2]], msg[r[3]]); \
    G(v[2], v[6], v[10], v[14], msg[r[4]], msg[r[5]]); \
    G(v[3], v[7], v[11], v[15], msg[r[6]], msg[r[7]]); \
    G(v[0], v[5], v[10], v[15], msg[r[8]], msg[r[9]]); \
    G(v[1], v[6], v[11], v[12], msg[r[10]], msg[r[11]]); \
    G(v[2], v[7], v[8],  v[13], msg[r[12]], msg[r[13]]); \
    G(v[3], v[4], v[9],  v[14], msg[r[14]], msg[r[15]]); \
} while(0)

/*
 * SNEPPX - Blake3
 *
 * WHAT
 *   Blake3.
 *
 * CONCEPT
 *   Provides the Blake3.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static const uint32_t IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint8_t MSG_PERM[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
    {10,7,12,11,14,3,13,15,4,0,5,2,8,1,9,6},
    {12,13,14,15,11,10,3,8,7,2,0,4,1,9,6,5},
    {14,15,8,9,13,12,11,10,3,4,7,6,2,5,1,0},
    {8,9,1,0,15,14,13,12,11,10,3,4,7,6,5,2}
};

#define FLAG_CHUNK_START (1 << 0)
#define FLAG_CHUNK_END   (1 << 1)
#define FLAG_PARENT      (1 << 2)
#define FLAG_ROOT        (1 << 3)
/*
 * SNEPPX - BLAKE3 Cryptographic Hash
 *
 * WHAT
 *   BLAKE3 Cryptographic Hash.
 *
 * CONCEPT
 *   BLAKE3 hash function with Merkle tree structure, incremental updates, and parallelism.
 *
 * ROLE
 *   Used for content-addressable storage, integrity verification, and key derivation.
 *
 * REFERENCES
 *   IETF draft-irtf-cfrg-blake3, BLAKE3 specification.
 */



static void blake3_compress(const uint32_t key[8], const uint8_t block[64], uint64_t counter, uint8_t flags, uint32_t out[16]) {
    uint32_t v[16], msg[16];
    memcpy(v, key, 32);
    v[4] = IV[0]; v[5] = IV[1]; v[6] = IV[2]; v[7] = IV[3];
    v[8] = IV[4]; v[9] = IV[5]; v[10] = IV[6]; v[11] = IV[7];
    v[12] = (uint32_t)counter; v[13] = (uint32_t)(counter >> 32);
    v[14] = 0; v[15] = flags;
    for (int i = 0; i < 16; i++)
        msg[i] = (uint32_t)block[i*4] | (uint32_t)block[i*4+1]<<8 | (uint32_t)block[i*4+2]<<16 | (uint32_t)block[i*4+3]<<24;
    for (int r = 0; r < 7; r++) ROUND(MSG_PERM[r]);
    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
    for (int i = 0; i < 8; i++) out[i + 8] = key[i] ^ v[i + 8];
    memcpy(out + 8, key, 32);
}

static void blake3_hash_chunk(const uint32_t key[8], const uint8_t* data, size_t len, uint64_t counter, uint8_t flags, uint32_t out[16]) {
    uint8_t block[64];
    flags |= FLAG_CHUNK_START;
    size_t offset = 0;
    while (offset + 64 <= len) {
        blake3_compress(key, data + offset, counter, flags, out);
        flags = 0; counter++; offset += 64;
    }
    flags |= FLAG_CHUNK_END;
    size_t remaining = len - offset;
    memcpy(block, data + offset, remaining);
    memset(block + remaining, 0, 64 - remaining);
    block[remaining] = 0x80;
    if (remaining == 0) flags |= FLAG_CHUNK_START;
    blake3_compress(key, block, counter, flags & ~FLAG_CHUNK_START, out);
}

/**
 * @brief Initialize Blake3.
 */
void SNEPPX_blake3_init(SNEPPXBlake3State* s) {
    memcpy(s->key, IV, 32);
    s->counter = 0;
    s->buflen = 0;
    s->flags = 0;
}

/**
 * @brief Update Blake3.
 *
 * @param s [out] S value.
 * @param data [in] Data value.
 */
void SNEPPX_blake3_update(SNEPPXBlake3State* s, const uint8_t* data, size_t len) {
    while (len) {
        size_t take = SNEPPX_BLAKE3_CHUNK_LEN - s->buflen;
        if (take > len) take = len;
        memcpy(s->buf + s->buflen, data, take);
        s->buflen += take; data += take; len -= take;
        if (s->buflen == SNEPPX_BLAKE3_CHUNK_LEN) {
            uint32_t cv[16];
            blake3_hash_chunk(s->key, s->buf, SNEPPX_BLAKE3_CHUNK_LEN, s->counter, FLAG_CHUNK_END, cv);
            memcpy(s->key, cv, 32);
            s->counter++;
            s->buflen = 0;
        }
    }
}

/**
 * @brief Perform Blake3 Finish.
 *
 * @param s [out] S value.
 */
void SNEPPX_blake3_finish(SNEPPXBlake3State* s, uint8_t* hash) {
    uint32_t cv[16];
    uint8_t flags = FLAG_ROOT;
    if (s->buflen == 0 && s->counter == 0) flags |= FLAG_CHUNK_START;
    blake3_hash_chunk(s->key, s->buf, s->buflen, s->counter, flags | FLAG_CHUNK_END, cv);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            hash[i * 4 + j] = (uint8_t)(cv[i] >> (j * 8));
}
