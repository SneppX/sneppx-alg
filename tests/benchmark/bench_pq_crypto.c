#include "bench_common.h"
#include <kyber.h>
#include <dilithium.h>
#include <sphincsplus.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Bench Pq Crypto
 *
 * WHAT
 *   Bench Pq Crypto.
 *
 * CONCEPT
 *   Provides the Bench Pq Crypto.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ------------------------------------------------------------------ */
/*  Kyber KEM benchmarks                                               */
/* ------------------------------------------------------------------ */
static void bench_kyber_keygen(void) {
    BENCH_INIT(bs);
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES];
    BENCH_START(bs, 1000, 100, {
        SNEPPX_kyber_keygen(pk, sk, 3);
    });
    bench_print("Kyber-768 keygen", &bs);
}
static void bench_kyber_encaps(void) {
    BENCH_INIT(bs);
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES], ct[KYBER_CIPHERTEXTBYTES], ss[KYBER_SSBYTES];
    SNEPPX_kyber_keygen(pk, sk, 3);
    BENCH_START(bs, 1000, 100, {
        SNEPPX_kyber_encaps(ct, ss, pk, 3);
    });
    bench_print("Kyber-768 encaps", &bs);
}
static void bench_kyber_decaps(void) {
    BENCH_INIT(bs);
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES], ct[KYBER_CIPHERTEXTBYTES], ss[KYBER_SSBYTES];
    SNEPPX_kyber_keygen(pk, sk, 3);
    SNEPPX_kyber_encaps(ct, ss, pk, 3);
    BENCH_START(bs, 1000, 100, {
        SNEPPX_kyber_decaps(ss, ct, sk, 3);
    });
    bench_print("Kyber-768 decaps", &bs);
}

/* ------------------------------------------------------------------ */
/*  Dilithium benchmarks                                               */
/* ------------------------------------------------------------------ */
static void bench_dilithium_keygen(void) {
    BENCH_INIT(bs);
    uint8_t pk[DILITHIUM_PUBLICKEYBYTES], sk[DILITHIUM_SECRETKEYBYTES];
    BENCH_START(bs, 500, 50, {
        SNEPPX_dilithium_keygen(pk, sk, 3);
    });
    bench_print("Dilithium-3 keygen", &bs);
}
static void bench_dilithium_sign(void) {
    BENCH_INIT(bs);
    uint8_t pk[DILITHIUM_PUBLICKEYBYTES], sk[DILITHIUM_SECRETKEYBYTES];
    uint8_t sig[DILITHIUM_SIGBYTES];
    size_t siglen;
    const uint8_t msg[] = "benchmark message";
    SNEPPX_dilithium_keygen(pk, sk, 3);
    BENCH_START(bs, 500, 50, {
        SNEPPX_dilithium_sign(sig, &siglen, msg, sizeof(msg), sk, 3);
    });
    bench_print("Dilithium-3 sign", &bs);
}
static void bench_dilithium_verify(void) {
    BENCH_INIT(bs);
    uint8_t pk[DILITHIUM_PUBLICKEYBYTES], sk[DILITHIUM_SECRETKEYBYTES];
    uint8_t sig[DILITHIUM_SIGBYTES];
    size_t siglen;
    const uint8_t msg[] = "benchmark message";
    SNEPPX_dilithium_keygen(pk, sk, 3);
    SNEPPX_dilithium_sign(sig, &siglen, msg, sizeof(msg), sk, 3);
    BENCH_START(bs, 500, 50, {
        SNEPPX_dilithium_verify(sig, siglen, msg, sizeof(msg), pk, 3);
    });
    bench_print("Dilithium-3 verify", &bs);
}

/* ------------------------------------------------------------------ */
/*  SPHINCS+ benchmarks                                                */
/* ------------------------------------------------------------------ */
static void bench_sphincs_keygen(void) {
    printf("  (this may take several minutes...)\n");
    BENCH_INIT(bs);
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    BENCH_START(bs, 1, 0, {
        SNEPPX_sphincs_keygen(pk, sk, 128);
    });
    bench_print("SPHINCS+-128s keygen", &bs);
}
static void bench_sphincs_sign(void) {
    printf("  (this may take several minutes...)\n");
    BENCH_INIT(bs);
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    uint8_t sig[SPHINCS_SIGBYTES];
    size_t siglen;
    const uint8_t msg[] = "benchmark message";
    SNEPPX_sphincs_keygen(pk, sk, 128);
    BENCH_START(bs, 1, 0, {
        SNEPPX_sphincs_sign(sig, &siglen, msg, sizeof(msg), sk, 128);
    });
    bench_print("SPHINCS+-128s sign", &bs);
}

/* ------------------------------------------------------------------ */
/*  Variant sweep (Kyber 512/768/1024)                                 */
/* ------------------------------------------------------------------ */
static void bench_kyber_variants(void) {
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES];
    uint8_t ct[KYBER_CIPHERTEXTBYTES], ss[KYBER_SSBYTES];
    int variants[] = {2, 3, 4};
    const char* names[] = {"Kyber-512", "Kyber-768", "Kyber-1024"};
    for (int i = 0; i < 3; i++) {
        int v = variants[i];
        BENCH_INIT(bs);
        BENCH_START(bs, 500, 50, {
            SNEPPX_kyber_keygen(pk, sk, v);
        });
        bench_print(names[i], &bs);
    }
    for (int i = 0; i < 3; i++) {
        int v = variants[i];
        SNEPPX_kyber_keygen(pk, sk, v);
        BENCH_INIT(bs);
        BENCH_START(bs, 500, 50, {
            SNEPPX_kyber_encaps(ct, ss, pk, v);
        });
        printf("  %-35s %8.3f ms\n", names[i], bs.mean_time * 1000.0);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Post-Quantum Cryptography Benchmarks ===\n\n");

    printf("--- Kyber KEM ---\n");
    BENCH_RUN("Kyber-768 keygen", bench_kyber_keygen);
    BENCH_RUN("Kyber-768 encaps", bench_kyber_encaps);
    BENCH_RUN("Kyber-768 decaps", bench_kyber_decaps);
    printf("\n");

    printf("--- Dilithium ---\n");
    BENCH_RUN("Dilithium-3 keygen", bench_dilithium_keygen);
    BENCH_RUN("Dilithium-3 sign", bench_dilithium_sign);
    BENCH_RUN("Dilithium-3 verify", bench_dilithium_verify);
    printf("\n");

    printf("--- SPHINCS+ (single run, may take minutes) ---\n");
    BENCH_RUN("SPHINCS+-128s keygen", bench_sphincs_keygen);
    BENCH_RUN("SPHINCS+-128s sign", bench_sphincs_sign);
    printf("\n");

    BENCH_MAIN();
}
