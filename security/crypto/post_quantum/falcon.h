#ifndef SNEPPX_FALCON_H
#define SNEPPX_FALCON_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_FALCON_512_SECRETKEY_SIZE 1281
#define SNEPPX_FALCON_512_PUBLICKEY_SIZE 897
#define SNEPPX_FALCON_512_SIGNATURE_SIZE 690
#define SNEPPX_FALCON_512_MESSAGE_SIZE 33000

#define SNEPPX_FALCON_1024_SECRETKEY_SIZE 2305
#define SNEPPX_FALCON_1024_PUBLICKEY_SIZE 1793
#define SNEPPX_FALCON_1024_SIGNATURE_SIZE 1330
#define SNEPPX_FALCON_1024_MESSAGE_SIZE 66000

/*
 * SNEPPX - Falcon
 *
 * WHAT
 *   Falcon.
 *
 * CONCEPT
 *   Provides the Falcon.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_FALCON_512,
    SNEPPX_FALCON_1024
} SNEPPXFalconVariant;

typedef struct {
    SNEPPXFalconVariant variant;
    uint8_t* secret_key;
    size_t secret_key_len;
    uint8_t* public_key;
    size_t public_key_len;
    uint8_t* compressed_pk;
    size_t compressed_pk_len;
    int8_t* f_inner;
    int8_t* g_inner;
    int8_t* F_inner;
    int8_t* G_inner;
    uint16_t* h_fft;
    size_t logn;
    uint64_t signing_counter;
} SNEPPXFalconKeypair;

typedef struct {
    uint8_t* signature;
    size_t signature_len;
    uint8_t* salt;
    size_t salt_len;
    uint8_t* message_hash;
    size_t hash_len;
    uint64_t signing_time_ns;
    uint8_t compressed : 1;
} SNEPPXFalconSignature;

int snepx_falcon_keygen(SNEPPXFalconKeypair* keypair, SNEPPXFalconVariant variant);
int snepx_falcon_sign(SNEPPXFalconKeypair* keypair, const uint8_t* message, size_t message_len, SNEPPXFalconSignature* sig);
int snepx_falcon_verify(const SNEPPXFalconKeypair* keypair, const uint8_t* message, size_t message_len, const SNEPPXFalconSignature* sig);
int snepx_falcon_sign_compress(SNEPPXFalconSignature* sig);
int snepx_falcon_sign_decompress(SNEPPXFalconSignature* sig);
int snepx_falcon_export_pubkey(const SNEPPXFalconKeypair* keypair, uint8_t* out, size_t* out_len);
int snepx_falcon_import_pubkey(SNEPPXFalconKeypair* keypair, const uint8_t* in, size_t in_len);
int snepx_falcon_export_seckey(const SNEPPXFalconKeypair* keypair, uint8_t* out, size_t* out_len);
int snepx_falcon_import_seckey(SNEPPXFalconKeypair* keypair, const uint8_t* in, size_t in_len);
int snepx_falcon_keypair_destroy(SNEPPXFalconKeypair* keypair);

// Falcon-1024 (NIST Level 5)
int snepx_falcon_1024_keygen(SNEPPXFalconKeypair* keypair);
int snepx_falcon_1024_sign(SNEPPXFalconKeypair* keypair, const uint8_t* message, size_t message_len, uint8_t* signature, size_t* signature_len);
int snepx_falcon_1024_verify(const SNEPPXFalconKeypair* keypair, const uint8_t* message, size_t message_len, const uint8_t* signature, size_t signature_len);

#endif
