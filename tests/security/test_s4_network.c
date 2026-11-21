#include "transport_security.h"
#include "test_gtest.h"
#include "identity_management.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test S4 Network
 *
 * WHAT
 *   Test S4 Network.
 *
 * CONCEPT
 *   Provides the Test S4 Network.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_transport_init(void) {
    SNEPPXTransportSecurity ts;
    SX_ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    SX_ASSERT(ts.enabled == 1, "enabled by default");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_session(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid >= 0, "new session");
    int sid2 = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid2 != sid, "distinct sessions");
    SX_ASSERT(SNEPPX_transport_close_session(&ts, sid) == 0, "close session");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_encrypt_decrypt(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    uint8_t plaintext[] = "hello transport security!";
    uint8_t ciphertext[256];
    uint8_t nonce[12];
    SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, strlen((char*)plaintext)+1, ciphertext, nonce) == 0, "encrypt");
    uint8_t decrypted[256];
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, strlen((char*)plaintext)+1, nonce, decrypted) == 0, "decrypt");
    SX_ASSERT(strcmp((char*)plaintext, (char*)decrypted) == 0, "roundtrip");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_noise_handshake(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    uint8_t msg[64];
    size_t msg_len = sizeof(msg);
    SX_ASSERT(SNEPPX_transport_noise_handshake(&ts, NULL, 0, msg, &msg_len) == 0, "noise handshake");
    SX_ASSERT(msg_len == 48, "handshake msg 48 bytes");
    SNEPPX_transport_shutdown(&ts);
}

static void test_identity_init(void) {
    SNEPPXIdentityManager mgr;
    SX_ASSERT(SNEPPX_identity_init(&mgr) == 0, "identity init");
    SX_ASSERT(mgr.ddos_protection_enabled == 1, "ddos enabled");
    SNEPPX_identity_shutdown(&mgr);
}

static void test_identity_pin_verify(void) {
    SNEPPXIdentityManager mgr;
    SNEPPX_identity_init(&mgr);
    uint8_t fp[32];
    memset(fp, 0xAB, 32);
    SX_ASSERT(SNEPPX_identity_pin_cert(&mgr, fp, "test.SNEPPX.local", 0) >= 0, "pin cert");
    SX_ASSERT(SNEPPX_identity_verify_cert(&mgr, fp) == 1, "verify pinned cert");
    uint8_t bad_fp[32];
    memset(bad_fp, 0, 32);
    SX_ASSERT(SNEPPX_identity_verify_cert(&mgr, bad_fp) == 0, "unpinned cert rejected");
    SX_ASSERT(SNEPPX_identity_unpin_cert(&mgr, fp) == 0, "unpin cert");
    SX_ASSERT(SNEPPX_identity_verify_cert(&mgr, fp) == 0, "unpinned cert not verified");
    SNEPPX_identity_shutdown(&mgr);
}

static void test_ddos_protection(void) {
    SNEPPXIdentityManager mgr;
    SNEPPX_identity_init(&mgr);
    mgr.ddos_request_limit = 5;
    for (int i = 0; i < 5; i++) SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "under limit");
    SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 1, "over limit");
    SNEPPX_identity_ddos_reset(&mgr);
    SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "after reset");
    SNEPPX_identity_shutdown(&mgr);
}


TEST(test_s4_network, transport_init) { test_transport_init(); }
TEST(test_s4_network, transport_session) { test_transport_session(); }
TEST(test_s4_network, transport_encrypt_decrypt) { test_transport_encrypt_decrypt(); }
TEST(test_s4_network, transport_noise_handshake) { test_transport_noise_handshake(); }
TEST(test_s4_network, identity_init) { test_identity_init(); }
TEST(test_s4_network, identity_pin_verify) { test_identity_pin_verify(); }
TEST(test_s4_network, ddos_protection) { test_ddos_protection(); }
