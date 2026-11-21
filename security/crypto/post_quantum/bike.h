#ifndef SNEPPX_BIKE_H
#define SNEPPX_BIKE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_BIKE_128_SECRETKEY_SIZE 5228
#define SNEPPX_BIKE_128_PUBLICKEY_SIZE 3176
#define SNEPPX_BIKE_128_CIPHERTEXT_SIZE 3176
#define SNEPPX_BIKE_128_SHAREDSECRET_SIZE 32

#define SNEPPX_BIKE_192_SECRETKEY_SIZE 10164
#define SNEPPX_BIKE_192_PUBLICKEY_SIZE 6216
#define SNEPPX_BIKE_192_CIPHERTEXT_SIZE 6216
#define SNEPPX_BIKE_192_SHAREDSECRET_SIZE 48

#define SNEPPX_BIKE_256_SECRETKEY_SIZE 16916
#define SNEPPX_BIKE_256_PUBLICKEY_SIZE 10348
#define SNEPPX_BIKE_256_CIPHERTEXT_SIZE 10348
#define SNEPPX_BIKE_256_SHAREDSECRET_SIZE 64

/*
 * SNEPPX - Bike
 *
 * WHAT
 *   Bike.
 *
 * CONCEPT
 *   Provides the Bike.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_BIKE_128,
    SNEPPX_BIKE_192,
    SNEPPX_BIKE_256
} SNEPPXBIKEVariant;

typedef struct {
    SNEPPXBIKEVariant variant;
    uint32_t r_param;
    uint32_t w_param;
    uint32_t t_param;
    uint64_t* h0;
    uint64_t* h1;
    uint64_t* sigma0;
    uint64_t* sigma1;
    uint8_t* secret_key;
    size_t secret_key_len;
    uint8_t* public_key;
    size_t public_key_len;
    uint64_t* error_generator_state;
} SNEPPXBIKEKeypair;

typedef struct {
    uint8_t* ciphertext;
    size_t ciphertext_len;
    uint8_t* shared_secret;
    size_t shared_secret_len;
    uint64_t* error_vector;
    size_t error_vector_len;
    uint32_t decoding_attempts;
    uint8_t decoded : 1;
} SNEPPXBIKECiphertext;

int snepx_bike_keygen(SNEPPXBIKEKeypair* keypair, SNEPPXBIKEVariant variant);
int snepx_bike_encapsulate(const SNEPPXBIKEKeypair* keypair, SNEPPXBIKECiphertext* ct);
int snepx_bike_decapsulate(const SNEPPXBIKEKeypair* keypair, const SNEPPXBIKECiphertext* ct, uint8_t* shared_secret, size_t* ss_len);
int snepx_bike_export_pubkey(const SNEPPXBIKEKeypair* keypair, uint8_t* out, size_t* out_len);
int snepx_bike_import_pubkey(SNEPPXBIKEKeypair* keypair, const uint8_t* in, size_t in_len);
int snepx_bike_export_seckey(const SNEPPXBIKEKeypair* keypair, uint8_t* out, size_t* out_len);
int snepx_bike_import_seckey(SNEPPXBIKEKeypair* keypair, const uint8_t* in, size_t in_len);
int snepx_bike_keypair_destroy(SNEPPXBIKEKeypair* keypair);

// BIKE decoder
int snepx_bike_decode(uint64_t* syndrome, const uint64_t* h0, const uint64_t* h1, uint32_t r, uint32_t w, uint32_t t, uint64_t* error_out);

// Constant-time helpers
int snepx_bike_ct_cmp(const uint8_t* a, const uint8_t* b, size_t len);
void snepx_bike_ct_swap(uint8_t* a, uint8_t* b, size_t len, uint8_t swap);

#endif
