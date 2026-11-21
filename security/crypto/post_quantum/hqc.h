#ifndef SNEPPX_HQC_H
#define SNEPPX_HQC_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_HQC_128_SECRETKEY_SIZE 6220
#define SNEPPX_HQC_128_PUBLICKEY_SIZE 5874
#define SNEPPX_HQC_128_CIPHERTEXT_SIZE 6053
#define SNEPPX_HQC_128_SHAREDSECRET_SIZE 64

#define SNEPPX_HQC_192_SECRETKEY_SIZE 11348
#define SNEPPX_HQC_192_PUBLICKEY_SIZE 10826
#define SNEPPX_HQC_192_CIPHERTEXT_SIZE 11114
#define SNEPPX_HQC_192_SHAREDSECRET_SIZE 64

#define SNEPPX_HQC_256_SECRETKEY_SIZE 17388
#define SNEPPX_HQC_256_PUBLICKEY_SIZE 16690
#define SNEPPX_HQC_256_CIPHERTEXT_SIZE 17090
#define SNEPPX_HQC_256_SHAREDSECRET_SIZE 64

/*
 * SNEPPX - Hqc
 *
 * WHAT
 *   Hqc.
 *
 * CONCEPT
 *   Provides the Hqc.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_HQC_128,
    SNEPPX_HQC_192,
    SNEPPX_HQC_256
} SNEPPXHQCLevel;

typedef struct {
    SNEPPXHQCLevel level;
    uint32_t n_param;
    uint32_t n1_param;
    uint32_t n2_param;
    uint32_t w_param;
    uint32_t w_half;
    uint32_t delta;
    uint64_t* x;
    uint64_t* y;
    uint64_t* h;
    uint64_t* secret_key;
    size_t secret_key_len;
    uint64_t* public_key;
    size_t public_key_len;
    uint64_t* vector_space_basis;
    uint64_t* reed_muller_generator;
} SNEPPXHQCKeypair;

typedef struct {
    uint64_t* u;
    uint64_t* v;
    uint64_t* d;
    uint64_t* salt;
    uint64_t* ciphertext;
    size_t ciphertext_len;
    uint64_t* shared_secret;
    size_t shared_secret_len;
    uint8_t* syndrome;
    size_t syndrome_len;
} SNEPPXHQCCiphertext;

int snepx_hqc_keygen(SNEPPXHQCKeypair* keypair, SNEPPXHQCLevel level);
int snepx_hqc_encapsulate(const SNEPPXHQCKeypair* keypair, SNEPPXHQCCiphertext* ct);
int snepx_hqc_decapsulate(const SNEPPXHQCKeypair* keypair, const SNEPPXHQCCiphertext* ct, uint8_t* shared_secret, size_t* ss_len);
int snepx_hqc_keypair_destroy(SNEPPXHQCKeypair* keypair);

// HQC internal arithmetic
int snepx_hqc_vect_mul(uint64_t* o, const uint64_t* v, const uint64_t* h, uint32_t n);
int snepx_hqc_vect_add(uint64_t* o, const uint64_t* v1, const uint64_t* v2, uint32_t n);
int snepx_hqc_vect_rotate(uint64_t* o, const uint64_t* v, uint32_t n, int32_t shift);
int snepx_hqc_reed_muller_encode(uint64_t* codeword, const uint64_t* message);
int snepx_hqc_reed_muller_decode(uint64_t* message, const uint64_t* codeword);

#endif
