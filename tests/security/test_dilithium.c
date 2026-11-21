#include "dilithium.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Dilithium
 *
 * WHAT
 *   Test Dilithium.
 *
 * CONCEPT
 *   Provides ML-DSA lattice signatures (FIPS 204).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_dilithium, suite) {
    uint8_t pk[DILITHIUM_PUBLICKEYBYTES], sk[DILITHIUM_SECRETKEYBYTES];
    uint8_t sig[DILITHIUM_SIGBYTES];
    size_t siglen;
    uint8_t msg[] = "SNEPPX-Algo Dilithium test message";
    
    printf("Testing keygen...\n");
    int ret = SNEPPX_dilithium_keygen(pk, sk, 2);
    printf("keygen returned: %d\n", ret);
    if (ret != 0) { printf("FAIL: keygen\n"); FAIL() << "early exit (legacy return 1)"; }
    
    printf("Testing sign...\n");
    ret = SNEPPX_dilithium_sign(sig, &siglen, msg, sizeof(msg), sk, 2);
    printf("sign returned: %d, siglen=%zu\n", ret, siglen);
    if (ret != 0) { printf("FAIL: sign\n"); FAIL() << "early exit (legacy return 1)"; }
    
    printf("Testing verify...\n");
    ret = SNEPPX_dilithium_verify(sig, siglen, msg, sizeof(msg), pk, 2);
    printf("verify returned: %d\n", ret);
    if (ret != 0) { printf("FAIL: verify\n"); FAIL() << "early exit (legacy return 1)"; }
    
    printf("PASS: Dilithium sign/verify OK\n");
    return;
}
