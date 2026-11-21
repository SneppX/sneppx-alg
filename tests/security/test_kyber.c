#include "kyber.h"
#include "test_gtest.h"
#include "cryptographic_random_generator.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Kyber
 *
 * WHAT
 *   Test Kyber.
 *
 * CONCEPT
 *   Provides Kyber post-quantum KEM (FIPS 203).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_kyber, suite) {
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES];
    uint8_t ct[KYBER_CIPHERTEXTBYTES], ss1[KYBER_SSBYTES], ss2[KYBER_SSBYTES];
    if (SNEPPX_kyber_keygen(pk, sk, 3) != 0) { printf("FAIL: keygen\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("pk[0]=%02x pk[1]=%02x pk[32]=%02x pk[33]=%02x\n", pk[0], pk[1], pk[32], pk[33]);
    printf("sk[0]=%02x sk[1]=%02x\n", sk[0], sk[1]);
    if (SNEPPX_kyber_encaps(ct, ss1, pk, 3) != 0) { printf("FAIL: encaps\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("ss1[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %02x", ss1[i]);
    printf("\n");
    if (SNEPPX_kyber_decaps(ss2, ct, sk, 3) != 0) { printf("FAIL: decaps\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("ss2[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %02x", ss2[i]);
    printf("\n");
    if (memcmp(ss1, ss2, KYBER_SSBYTES) != 0) { printf("FAIL: shared secret mismatch\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("PASS: Kyber KEM round-trip OK\n");
    return;
}
