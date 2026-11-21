#include "sphincsplus.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Sphincsplus
 *
 * WHAT
 *   Test Sphincsplus.
 *
 * CONCEPT
 *   Provides SPHINCS+ stateless hash signatures (FIPS 205).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_sphincsplus, suite) {
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    uint8_t sig[SPHINCS_SIGBYTES];
    size_t siglen;
    uint8_t msg[] = "SNEPPX-Algo SPHINCS+ test message";
    if (SNEPPX_sphincs_keygen(pk, sk, 128) != 0) { printf("FAIL: keygen\n"); FAIL() << "early exit (legacy return 1)"; }
    if (SNEPPX_sphincs_sign(sig, &siglen, msg, sizeof(msg), sk, 128) != 0) { printf("FAIL: sign\n"); FAIL() << "early exit (legacy return 1)"; }
    if (siglen != SPHINCS_SIGBYTES) { printf("FAIL: siglen %zu != %d\n", siglen, SPHINCS_SIGBYTES); FAIL() << "siglen mismatch"; }
    if (SNEPPX_sphincs_verify(sig, siglen, msg, sizeof(msg), pk, 128) != 0) { printf("FAIL: verify\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("PASS: SPHINCS+ sign/verify OK\n");
    return;
}

TEST(test_sphincsplus, tamper) {
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    uint8_t sig[SPHINCS_SIGBYTES], bad[SPHINCS_SIGBYTES];
    size_t siglen;
    uint8_t msg[] = "a second message for tamper checks";
    if (SNEPPX_sphincs_keygen(pk, sk, 128) != 0) FAIL() << "keygen";
    if (SNEPPX_sphincs_sign(sig, &siglen, msg, sizeof(msg), sk, 128) != 0) FAIL() << "sign";
    if (SNEPPX_sphincs_verify(sig, siglen, msg, sizeof(msg), pk, 128) != 0) FAIL() << "baseline verify";
    memcpy(bad, sig, siglen);
    bad[siglen / 2] ^= 0x01;
    if (SNEPPX_sphincs_verify(bad, siglen, msg, sizeof(msg), pk, 128) == 0) FAIL() << "tampered sig accepted";
    uint8_t other_pk[SPHINCS_PUBLICKEYBYTES], other_sk[SPHINCS_SECRETKEYBYTES];
    if (SNEPPX_sphincs_keygen(other_pk, other_sk, 128) != 0) FAIL() << "keygen2";
    if (SNEPPX_sphincs_verify(sig, siglen, msg, sizeof(msg), other_pk, 128) == 0) FAIL() << "wrong pk accepted";
    if (SNEPPX_sphincs_verify(sig, siglen, (const uint8_t *)"different message", 17, pk, 128) == 0) FAIL() << "wrong message accepted";
    printf("PASS: SPHINCS+ tamper/wrong-key/wrong-msg rejected\n");
    return;
}