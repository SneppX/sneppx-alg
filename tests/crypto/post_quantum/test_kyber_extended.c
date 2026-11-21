#include "security/crypto/post_quantum/kyber_extended.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Kyber Extended
 *
 * WHAT
 *   Test Kyber Extended.
 *
 * CONCEPT
 *   Provides Kyber post-quantum KEM (FIPS 203).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_kyber_keygen(void) {
    SNEPPXKyberKeypair kp;
    memset(&kp, 0, sizeof(kp));
    int ret = snepx_kyber_keygen(&kp, SNEPPX_KYBER_768);
    assert(ret == 0);
    assert(kp.public_key_len > 0);
    assert(kp.secret_key_len > 0);
    snepx_kyber_keypair_destroy(&kp);
}

static void test_kyber_encap_decap(void) {
    SNEPPXKyberKeypair kp;
    snepx_kyber_keygen(&kp, SNEPPX_KYBER_768);
    SNEPPXKyberCiphertext ct;
    memset(&ct, 0, sizeof(ct));
    int ret = snepx_kyber_encapsulate(&kp, &ct);
    assert(ret == 0);
    assert(ct.ciphertext_len > 0);
    assert(ct.shared_secret_len > 0);
    uint8_t ss[64];
    size_t ss_len = sizeof(ss);
    ret = snepx_kyber_decapsulate(&kp, &ct, ss, &ss_len);
    assert(ret == 0);
    assert(ss_len == ct.shared_secret_len);
    assert(memcmp(ss, ct.shared_secret, ss_len) == 0);
    snepx_kyber_ciphertext_destroy(&ct);
    snepx_kyber_keypair_destroy(&kp);
}

int main(void) {
    test_kyber_keygen();
    test_kyber_encap_decap();
    return 0;
}
