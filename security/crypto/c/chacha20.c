#include "chacha20_stream_cipher.h"
#include <string.h>

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
/*
 * SNEPPX - ChaCha20 Stream Cipher
 *
 * WHAT
 *   ChaCha20 Stream Cipher.
 *
 * CONCEPT
 *   ChaCha20 256-bit stream cipher with 96-bit nonce and 32-bit counter.
 *
 * ROLE
 *   Layer S2 authenticated encryption (ChaCha20-Poly1305). Preferred over AES-GCM on platforms without AES-NI.
 *
 * REFERENCES
 *   RFC 8439 (ChaCha20-Poly1305), RFC 7539 (original ChaCha20-Poly1305).
 */



static void qr(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d) {
    *a += *b; *d ^= *a; *d = ROTL32(*d, 16);
    *c += *d; *b ^= *c; *b = ROTL32(*b, 12);
    *a += *b; *d ^= *a; *d = ROTL32(*d, 8);
    *c += *d; *b ^= *c; *b = ROTL32(*b, 7);
}

/**
 * @brief Initialize Chacha20.
 *
 * @param s [out] S value.
 */
void SNEPPX_chacha20_init(SNEPPXChaCha20State* s, const uint8_t key[32], const uint8_t nonce[12], uint32_t counter) {
    s->state[0] = 1634760805U; s->state[1] = 857760878U; s->state[2] = 2036477234U; s->state[3] = 1797285236U;
    for (int i = 0; i < 8; i++)
        s->state[4 + i] = (uint32_t)key[i*4] | (uint32_t)key[i*4+1]<<8 | (uint32_t)key[i*4+2]<<16 | (uint32_t)key[i*4+3]<<24;
    s->state[12] = counter;
    s->state[13] = (uint32_t)nonce[0] | (uint32_t)nonce[1]<<8 | (uint32_t)nonce[2]<<16 | (uint32_t)nonce[3]<<24;
    s->state[14] = (uint32_t)nonce[4] | (uint32_t)nonce[5]<<8 | (uint32_t)nonce[6]<<16 | (uint32_t)nonce[7]<<24;
    s->state[15] = (uint32_t)nonce[8] | (uint32_t)nonce[9]<<8 | (uint32_t)nonce[10]<<16 | (uint32_t)nonce[11]<<24;
}

/**
 * @brief Perform Chacha20 Block.
 *
 * @param s [out] S value.
 */
void SNEPPX_chacha20_block(SNEPPXChaCha20State* s, uint8_t output[64]) {
    uint32_t x[16];
    memcpy(x, s->state, 64);
    for (int i = 0; i < 10; i++) {
        qr(&x[0], &x[4], &x[8], &x[12]); qr(&x[1], &x[5], &x[9], &x[13]);
        qr(&x[2], &x[6], &x[10], &x[14]); qr(&x[3], &x[7], &x[11], &x[15]);
        qr(&x[0], &x[5], &x[10], &x[15]); qr(&x[1], &x[6], &x[11], &x[12]);
        qr(&x[2], &x[7], &x[8], &x[13]); qr(&x[3], &x[4], &x[9], &x[14]);
    }
    for (int i = 0; i < 16; i++) {
        uint32_t w = x[i] + s->state[i];
        output[i*4] = (uint8_t)w; output[i*4+1] = (uint8_t)(w>>8);
        output[i*4+2] = (uint8_t)(w>>16); output[i*4+3] = (uint8_t)(w>>24);
    }
    s->state[12]++;
}

/**
 * @brief Encrypt Chacha20.
 *
 * @param s [out] S value.
 * @param data [out] Data value.
 */
void SNEPPX_chacha20_encrypt(SNEPPXChaCha20State* s, uint8_t* data, size_t len) {
    uint8_t block[64];
    size_t offset = 0;
    while (offset < len) {
        SNEPPX_chacha20_block(s, block);
        size_t todo = len - offset;
        if (todo > 64) todo = 64;
        for (size_t i = 0; i < todo; i++) data[offset + i] ^= block[i];
        offset += todo;
    }
}
