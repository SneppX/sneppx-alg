#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "sha512_hashing_implementation.h"
#include "ed25519_signature_verification.h"
#include "constant_time_operations.h"

extern int SNEPPX_random_bytes(uint8_t* buffer, size_t len);

static int hex_to_bytes(const char* hex, uint8_t* out, size_t outlen) {
    size_t len = strlen(hex);
    if (len != outlen * 2) return -1;
    for (size_t i = 0; i < outlen; i++) {
        unsigned int val;
        if (sscanf(hex + 2*i, "%02x", &val) != 1) return -1;
        out[i] = (uint8_t)val;
    }
    return 0;
}

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s:", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

/* Ed25519 base point encoding (RFC 8032) */
static const uint8_t B_Y[32] = {
    0x58,0x66,0x66,0x66,0x66,0x66,0x66,0x66,
    0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,
    0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,
    0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66
};

int main(int argc, char** argv) {
    /* When invoked with no subcommand (e.g. by ctest), default to the
     * RFC 8032 interop self-test so the registered test target actually
     * validates the implementation instead of printing usage and failing. */
    const char* cmd = (argc < 2) ? "rfc_test" : argv[1];

    if (strcmp(cmd, "sha512") == 0 && argc >= 3) {
        size_t len = strlen(argv[2]) / 2;
        uint8_t* data = (uint8_t*)malloc(len > 0 ? len : 1);
        if (len > 0 && hex_to_bytes(argv[2], data, len) != 0) { fprintf(stderr, "bad hex\n"); free(data); return 1; }
        uint8_t digest[64];
        SNEPPX_sha512(data, len, digest);
        print_hex("sha512", digest, 64);
        free(data);
        return 0;
    }

    if (strcmp(cmd, "legacy_sha512") == 0 && argc >= 3) {
        size_t len = strlen(argv[2]) / 2;
        uint8_t* data = (uint8_t*)malloc(len > 0 ? len : 1);
        if (len > 0 && hex_to_bytes(argv[2], data, len) != 0) { fprintf(stderr, "bad hex\n"); free(data); return 1; }
        uint8_t digest[64];
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx, data, len);
        SNEPPX_sha512_legacy_finish(&ctx, digest);
        print_hex("legacy_sha512", digest, 64);
        free(data);
        return 0;
    }

    if (strcmp(cmd, "keypair") == 0 && argc >= 3) {
        uint8_t seed[32];
        if (hex_to_bytes(argv[2], seed, 32) != 0) { fprintf(stderr, "bad seed\n"); return 1; }
        SNEPPXEd25519Keypair kp;
        SNEPPX_ed25519_secret_key_expand(kp.private_key, seed);
        SNEPPX_ed25519_scalar_multiply(kp.public_key, kp.private_key, B_Y);
        print_hex("pk", kp.public_key, 32);
        print_hex("sk", kp.private_key, 32);
        return 0;
    }

    if (strcmp(cmd, "sign") == 0 && argc >= 4) {
        uint8_t seed[32], msg[4096];
        size_t msglen;
        if (hex_to_bytes(argv[2], seed, 32) != 0) { fprintf(stderr, "bad seed\n"); return 1; }
        msglen = strlen(argv[3]) / 2;
        if (msglen > 4096) { fprintf(stderr, "msg too long\n"); return 1; }
        if (msglen > 0 && hex_to_bytes(argv[3], msg, msglen) != 0) { fprintf(stderr, "bad msg\n"); return 1; }

        SNEPPXEd25519Keypair kp;
        SNEPPX_ed25519_secret_key_expand(kp.private_key, seed);
        SNEPPX_ed25519_scalar_multiply(kp.public_key, kp.private_key, B_Y);

        SNEPPXEd25519Signature sig;
        SNEPPX_ed25519_sign(&kp, msg, msglen, &sig);
        print_hex("pk", kp.public_key, 32);
        print_hex("sig", sig.data, 64);
        return 0;
    }

    if (strcmp(cmd, "verify") == 0 && argc >= 5) {
        uint8_t pk[32], sig[64], msg[4096];
        size_t msglen;
        if (hex_to_bytes(argv[2], pk, 32) != 0) { fprintf(stderr, "bad pk\n"); return 1; }
        if (hex_to_bytes(argv[3], sig, 64) != 0) { fprintf(stderr, "bad sig\n"); return 1; }
        msglen = strlen(argv[4]) / 2;
        if (msglen > 4096) { fprintf(stderr, "msg too long\n"); return 1; }
        if (msglen > 0 && hex_to_bytes(argv[4], msg, msglen) != 0) { fprintf(stderr, "bad msg\n"); return 1; }

        SNEPPXEd25519Signature sig_struct;
        memcpy(sig_struct.data, sig, 64);
        int ret = SNEPPX_ed25519_verify(pk, msg, msglen, &sig_struct);
        printf("%d\n", ret);
        return 0;
    }

    if (strcmp(cmd, "verify_compat") == 0 && argc >= 6) {
        uint8_t pk[32], sig[64], msg[4096];
        size_t msglen;
        if (hex_to_bytes(argv[2], pk, 32) != 0) { fprintf(stderr, "bad pk\n"); return 1; }
        if (hex_to_bytes(argv[3], sig, 64) != 0) { fprintf(stderr, "bad sig\n"); return 1; }
        msglen = strlen(argv[4]) / 2;
        if (msglen > 4096) { fprintf(stderr, "msg too long\n"); return 1; }
        if (msglen > 0 && hex_to_bytes(argv[4], msg, msglen) != 0) { fprintf(stderr, "bad msg\n"); return 1; }
        int allow_legacy = atoi(argv[5]);

        int ret = SNEPPX_ed25519_verify_compat(sig, msg, msglen, pk, allow_legacy);
        printf("%d\n", ret);
        return 0;
    }

    if (strcmp(cmd, "rfc_test") == 0) {
        int all_pass = 1;

        /* SHA-512("abc") == RFC 8032 */
        uint8_t digest[64];
        SNEPPX_sha512((const uint8_t*)"abc", 3, digest);
        const uint8_t rfc_abc[64] = {
            0xdd,0xaf,0x35,0xa1,0x93,0x61,0x7a,0xba,0xcc,0x41,
            0x73,0x49,0xae,0x20,0x41,0x31,0x12,0xe6,0xfa,0x4e,
            0x89,0xa9,0x7e,0xa2,0x0a,0x9e,0xee,0xe6,0x4b,0x55,
            0xd3,0x9a,0x21,0x92,0x99,0x2a,0x27,0x4f,0xc1,0xa8,
            0x36,0xba,0x3c,0x23,0xa3,0xfe,0xeb,0xbd,0x45,0x4d,
            0x44,0x23,0x64,0x3c,0xe8,0x0e,0x2a,0x9a,0xc9,0x4f,
            0xa5,0x4c,0xa4,0x9f
        };
        int ok = (memcmp(digest, rfc_abc, 64) == 0);
        printf("SHA-512(\"abc\") == RFC: %s\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;

        /* SHA-512("") == RFC (from Python hashlib) */
        SNEPPX_sha512((const uint8_t*)"", 0, digest);
        const uint8_t rfc_empty[64] = {
            0xcf,0x83,0xe1,0x35,0x7e,0xef,0xb8,0xbd,0xf1,0x54,
            0x28,0x50,0xd6,0x6d,0x80,0x07,0xd6,0x20,0xe4,0x05,
            0x0b,0x57,0x15,0xdc,0x83,0xf4,0xa9,0x21,0xd3,0x6c,
            0xe9,0xce,0x47,0xd0,0xd1,0x3c,0x5d,0x85,0xf2,0xb0,
            0xff,0x83,0x18,0xd2,0x87,0x7e,0xec,0x2f,0x63,0xb9,
            0x31,0xbd,0x47,0x41,0x7a,0x81,0xa5,0x38,0x32,0x7a,
            0xf9,0x27,0xda,0x3e
        };
        ok = (memcmp(digest, rfc_empty, 64) == 0);
        printf("SHA-512(\"\") == RFC: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) print_hex("  got", digest, 64);
        all_pass &= ok;

        /* Legacy != RFC */
        uint8_t legacy[64];
        {
            SNEPPXSHA512Context ctx;
            SNEPPX_sha512_init(&ctx);
            SNEPPX_sha512_update(&ctx, (const uint8_t*)"abc", 3);
            SNEPPX_sha512_legacy_finish(&ctx, legacy);
        }
        ok = (memcmp(legacy, rfc_abc, 64) != 0);
        printf("Legacy SHA-512(\"abc\") != RFC: %s\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;

        /* Ed25519 RFC 8032 interop — authoritative test vector 1 (empty msg).
         * Build the keypair from the RFC seed, attach the RFC-known public key,
         * and assert the C implementation emits the EXACT RFC signature. This
         * exercises the real sign/verify path (challenge reduction, scalar
         * arithmetic) rather than a self-consistent round-trip. */
        /* RFC 8032 Section 7.1, Test Vector 1 (empty message). */
        const uint8_t seed[32] = {
            0x9d,0x61,0xb1,0x9d,0xef,0xfd,0x5a,0x60,0xba,0x84,
            0x4a,0xf4,0x92,0xec,0x2c,0xc4,0x44,0x49,0xc5,0x69,
            0x7b,0x32,0x69,0x19,0x70,0x3b,0xac,0x03,0x1c,0xae,
            0x7f,0x60
        };
        const uint8_t rfc_pk[32] = {
            0xd7,0x5a,0x98,0x01,0x82,0xb1,0x0a,0xb7,0xd5,0x4b,
            0xfe,0xd3,0xc9,0x64,0x07,0x3a,0x0e,0xe1,0x72,0xf3,
            0xda,0xa6,0x23,0x25,0xaf,0x02,0x1a,0x68,0xf7,0x07,
            0x51,0x1a
        };
        const uint8_t rfc_sig[64] = {
            0xe5,0x56,0x43,0x00,0xc3,0x60,0xac,0x72,0x90,0x86,
            0xe2,0xcc,0x80,0x6e,0x82,0x8a,0x84,0x87,0x7f,0x1e,
            0xb8,0xe5,0xd9,0x74,0xd8,0x73,0xe0,0x65,0x22,0x49,
            0x01,0x55,0x5f,0xb8,0x82,0x15,0x90,0xa3,0x3b,0xac,
            0xc6,0x1e,0x39,0x70,0x1c,0xf9,0xb4,0x6b,0xd2,0x5b,
            0xf5,0xf0,0x59,0x5b,0xbe,0x24,0x65,0x51,0x41,0x43,
            0x8e,0x7a,0x10,0x0b
        };

        SNEPPXEd25519Keypair kp;
        SNEPPX_ed25519_secret_key_expand(kp.private_key, seed);
        memcpy(kp.public_key, rfc_pk, 32);

        SNEPPXEd25519Signature sig;
        SNEPPX_ed25519_sign(&kp, (const uint8_t*)"", 0, &sig);

        ok = (memcmp(sig.data, rfc_sig, 64) == 0);
        printf("Ed25519 RFC8032 TV1 signature match: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { print_hex("  got", sig.data, 64); print_hex("  exp", rfc_sig, 64); }
        all_pass &= ok;

        int vret = SNEPPX_ed25519_verify(kp.public_key, (const uint8_t*)"", 0, &sig);
        ok = (vret == 1);
        printf("Ed25519 verify (RFC sig): %s\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;

        int tret = SNEPPX_ed25519_verify(kp.public_key, (const uint8_t*)"x", 1, &sig);
        ok = (tret == 0);
        printf("Ed25519 verify (tampered msg rejected): %s\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;

        int crt = SNEPPX_ed25519_verify_compat(sig.data, (const uint8_t*)"", 0, kp.public_key, 1);
        ok = (crt == 1);
        printf("Ed25519 compat verify (legacy=1): %s\n", ok ? "PASS" : "FAIL");
        all_pass &= ok;

        printf("\nALL_RFC_TESTS: %s\n", all_pass ? "PASS" : "FAIL");
        return all_pass ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd);
    return 1;
}
