#include "security/crypto/protocols/noise_protocol_extended.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Noise Protocol
 *
 * WHAT
 *   Test Noise Protocol.
 *
 * CONCEPT
 *   Provides the Test Noise Protocol.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_noise_handshake_init(void) {
    SNEPPXNoiseHandshakeState hs;
    memset(&hs, 0, sizeof(hs));
    int ret = snepx_noise_handshake_init(&hs, SNEPPX_NOISE_PATTERN_NN, SNEPPX_NOISE_CIPHER_CHACHAPOLY, 1);
    assert(ret == 0);
    assert(hs.initiator == 1);
    assert(hs.pattern == SNEPPX_NOISE_PATTERN_NN);
    assert(hs.handshake_finished == 0);
    snepx_noise_handshake_destroy(&hs);
}

static void test_noise_transport(void) {
    SNEPPXNoiseHandshakeState hs;
    snepx_noise_handshake_init(&hs, SNEPPX_NOISE_PATTERN_NN, SNEPPX_NOISE_CIPHER_CHACHAPOLY, 1);
    uint8_t prologue[] = "noise_protocol_test";
    snepx_noise_handshake_set_prologue(&hs, prologue, sizeof(prologue) - 1);
    SNEPPXNoiseTransportState ts;
    memset(&ts, 0, sizeof(ts));
    uint8_t msg_buf[4096];
    size_t msg_len = sizeof(msg_buf);
    int ret = snepx_noise_handshake_write_message(&hs, NULL, 0, msg_buf, &msg_len, &ts);
    assert(ret == 0);
    assert(msg_len > 0);
    snepx_noise_handshake_destroy(&hs);
    snepx_noise_transport_destroy(&ts);
}

int main(void) {
    test_noise_handshake_init();
    test_noise_transport();
    return 0;
}
