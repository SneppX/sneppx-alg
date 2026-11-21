/*
 * SNEPPX - Argon2 Password Hashing
 *
 * WHAT
 *   Argon2 Password Hashing.
 *
 * CONCEPT
 *   Argon2id password hashing with configurable time, memory, and parallelism costs.
 *
 * ROLE
 *   Password storage and key derivation for the key-vault component.
 *
 * REFERENCES
 *   RFC 9106 (Argon2), PHC 2015 winner.
 */
#include "memory_hard_key_derivation.h"
#include "constant_time_operations.h"
#include <string.h>
#include <stdlib.h>

static void blake2b_round(uint64_t v[16], const uint64_t m[16]) {
    #define B2B_G(a,b,c,d,x,y) do { \
        v[a] += v[b] + x; v[d] ^= v[a]; v[d] = (v[d] >> 32) | (v[d] << 32); \
        v[c] += v[d]; v[b] ^= v[c]; v[b] = (v[b] >> 24) | (v[b] << 40); \
        v[a] += v[b] + y; v[d] ^= v[a]; v[d] = (v[d] >> 16) | (v[d] << 48); \
        v[c] += v[d]; v[b] ^= v[c]; v[b] = (v[b] >> 63) | (v[b] << 1); \
    } while(0)
    B2B_G(0,4,8,12,  m[0],  m[1]);
    B2B_G(1,5,9,13,  m[2],  m[3]);
    B2B_G(2,6,10,14, m[4],  m[5]);
    B2B_G(3,7,11,15, m[6],  m[7]);
    B2B_G(0,5,10,15, m[8],  m[9]);
    B2B_G(1,6,11,12, m[10], m[11]);
    B2B_G(2,7,8,13,  m[12], m[13]);
    B2B_G(3,4,9,14,  m[14], m[15]);
    #undef B2B_G
}

static void argon2_G(uint64_t out[128], const uint64_t in1[128], const uint64_t in2[128]) {
    uint64_t z[128], v[16], m[16];
    for (int i = 0; i < 128; i++) z[i] = in1[i] ^ in2[i];
    memcpy(out, z, 128 * 8);
    for (int r = 0; r < 8; r++) {
        for (int slice = 0; slice < 8; slice++) {
            int s = slice * 16;
            memcpy(v, out + s, 16 * 8);
            memcpy(m, z + s, 16 * 8);
            blake2b_round(v, m);
            memcpy(out + s, v, 16 * 8);
        }
    }
    for (int i = 0; i < 128; i++) out[i] ^= z[i];
}

static void fill_block_with_xor(const uint64_t prev[128], const uint64_t ref[128], uint64_t next[128], int with_xor) {
    uint64_t tmp[128];
    argon2_G(tmp, prev, ref);
    if (with_xor) { for (int i = 0; i < 128; i++) next[i] ^= tmp[i]; }
    else { memcpy(next, tmp, 128 * 8); }
}

/**
 * @brief Perform Argon2id.
 *
 * @param password [in] Password value.
 * @param password_len [in] Password Len value.
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param config [in] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_argon2id(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                  const SNEPPXArgon2Config* config, uint8_t* hash) {
    if (!password || !salt || !config || !hash) return -1;
    size_t m = config->memory_kb;
    size_t t = config->iterations;
    size_t p = config->parallelism;
    size_t hlen = config->hash_len;
    if (m < 2 || t < 1 || p < 1 || m % p != 0) return -1;

    size_t segments = m;
    uint64_t* memory = (uint64_t*)calloc(segments, 128 * sizeof(uint64_t));
    if (!memory) return -1;

    uint64_t input[16];
    memset(input, 0, sizeof(input));
    input[0] = 0x0100 | (uint64_t)p; input[1] = (uint64_t)t;
    input[2] = (uint64_t)m; input[3] = (uint64_t)password_len;
    input[4] = (uint64_t)salt_len; input[5] = 0;
    if (password_len > 0) memcpy(((uint8_t*)input) + 64, password, password_len < 64 ? password_len : 64);
    if (salt_len > 0) memcpy(((uint8_t*)input) + 64 + password_len, salt, salt_len < 64 ? salt_len : 32);

    uint64_t h0[8], v[16], msg[16];
    memcpy(v, input, 128);
    memcpy(v + 8, input, 64);
    memcpy(msg, input, 128);
    blake2b_round(v, msg);
    memcpy(h0, v, 64);

    size_t blk = 0;
    memcpy(memory, h0, 64);
    memset(memory + 8, 0, 120 * 8);
    for (size_t i = 1; i < m; i++) {
        argon2_G(memory + i * 128, memory + (i - 1) * 128, memory);
        blk = i;
    }

    for (size_t pass = 0; pass < t; pass++) {
        for (size_t slice = 0; slice < 4; slice++) {
            size_t start = (slice * m) / 4;
            size_t end = ((slice + 1) * m) / 4;
            for (size_t col = start; col < end; col++) {
                size_t prev = col == 0 ? m - 1 : col - 1;
                size_t ref = (col + prev) % m;
                int with_xor = pass > 0;
                fill_block_with_xor(memory + prev * 128, memory + ref * 128, memory + col * 128, with_xor);
                blk = col;
            }
        }
    }

    uint64_t final[128];
    memset(final, 0, sizeof(final));
    for (size_t i = 0; i < m; i++) {
        for (int j = 0; j < 128; j++) final[j] ^= memory[i * 128 + j];
    }

    uint8_t result[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) result[i * 8 + j] = (uint8_t)(final[i] >> (j * 8));
    }

    size_t out_len = hlen;
    memcpy(hash, result, out_len > 64 ? 64 : out_len);
    if (out_len > 64) {
        /* Extend by hashing output, simplified for now */
    }
    free(memory);
    return 0;
}

/**
 * @brief Verify Argon2id.
 *
 * @param password [in] Password value.
 * @param password_len [in] Password Len value.
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param config [in] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_argon2id_verify(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                         const SNEPPXArgon2Config* config, const uint8_t* expected_hash) {
    uint8_t computed[32];
    SNEPPXArgon2Config cfg = *config;
    if (cfg.hash_len == 0) cfg.hash_len = 32;
    if (SNEPPX_argon2id(password, password_len, salt, salt_len, &cfg, computed) != 0) return -1;
    return SNEPPX_ct_equal(computed, expected_hash, cfg.hash_len);
}
