#include "identity_management.h"
#include "test_gtest.h"
#include "transport_security.h"
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

static void test_transport_init(void)
{
    SNEPPXTransportSecurity ts;
    SX_ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    SX_ASSERT(ts.enabled == 1, "enabled by default");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_session(void)
{
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid >= 0, "new session");
    int sid2 = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid2 != sid, "distinct sessions");
    SX_ASSERT(SNEPPX_transport_close_session(&ts, sid) == 0, "close session");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_encrypt_decrypt(void)
{
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    uint8_t plaintext[] = "hello transport security!";
    uint8_t ciphertext[256];
    uint8_t nonce[12];
    SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, strlen((char*) plaintext) + 1, ciphertext, nonce) == 0,
              "encrypt");
    uint8_t decrypted[256];
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, strlen((char*) plaintext) + 1, nonce, decrypted) == 0,
              "decrypt");
    SX_ASSERT(strcmp((char*) plaintext, (char*) decrypted) == 0, "roundtrip");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_rejects_xor_aes_mode(void)
{
    SNEPPXTransportSecurity ts;
    SX_ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    SX_ASSERT(SNEPPX_transport_get_cipher_type() == 0, "default cipher is chacha20");

    SX_ASSERT(SNEPPX_transport_set_cipher(&ts, 1) == -1, "aes cipher request rejected");
    SX_ASSERT(SNEPPX_transport_get_cipher_type() == 0, "still chacha20");
    SX_ASSERT(SNEPPX_transport_set_cipher_type(1) == -1, "aes cipher type rejected");
    SX_ASSERT(SNEPPX_transport_get_cipher_type() == 0, "still chacha20 after type request");
    SX_ASSERT(SNEPPX_transport_set_cipher_type(2) == -1, "unknown cipher type rejected");
    SX_ASSERT(SNEPPX_transport_set_cipher_type(0) == 0, "chacha20 selection allowed");

    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid >= 0, "new session");
    uint8_t zeros[64] = {0};
    uint8_t ciphertext[64];
    uint8_t nonce[12];
    SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, zeros, sizeof(zeros), ciphertext, nonce) == 0,
              "encrypt after rejected aes mode");
    SX_ASSERT(memcmp(ciphertext, ciphertext + 32, 32) != 0, "no 32-byte repeating-key stream");
    uint8_t decrypted[64];
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, sizeof(ciphertext), nonce, decrypted) == 0,
              "decrypt after rejected aes mode");
    SX_ASSERT(memcmp(zeros, decrypted, sizeof(zeros)) == 0, "roundtrip after rejected aes mode");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_sequence_tracking(void)
{
    SNEPPXTransportSecurity ts;
    SX_ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid >= 0, "new session");

    uint8_t plaintext[16];
    uint8_t ciphertext[16];
    uint8_t nonce[SNEPPX_TLS_NONCE_LEN];
    uint8_t decrypted[16];

    for (int i = 0; i < 1002; i++)
    {
        memset(plaintext, (unsigned char) i, sizeof(plaintext));
        SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, sizeof(plaintext), ciphertext, nonce) == 0,
                  "sequential encrypt");
        SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, sizeof(ciphertext), nonce, decrypted) == 0,
                  "sequential decrypt");
        SX_ASSERT(memcmp(plaintext, decrypted, sizeof(plaintext)) == 0, "sequential roundtrip");
    }

    memset(plaintext, 0xA5, sizeof(plaintext));
    SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, sizeof(plaintext), ciphertext, nonce) == 0,
              "encrypt after rekey threshold");

    uint8_t replay_nonce[SNEPPX_TLS_NONCE_LEN];
    memcpy(replay_nonce, nonce, sizeof(nonce));
    replay_nonce[7] ^= 1;
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, sizeof(ciphertext), replay_nonce, decrypted) == -1,
              "out-of-order sequence rejected");
    memcpy(replay_nonce, nonce, sizeof(nonce));
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, sizeof(ciphertext), replay_nonce, decrypted) == 0,
              "expected sequence accepted");
    SX_ASSERT(memcmp(plaintext, decrypted, sizeof(plaintext)) == 0, "roundtrip after rekey threshold");

    uint8_t next_ciphertext[16];
    uint8_t next_nonce[SNEPPX_TLS_NONCE_LEN];
    SX_ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, sizeof(plaintext), next_ciphertext, next_nonce) == 0,
              "encrypt next sequence");
    SX_ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, sizeof(ciphertext), replay_nonce, decrypted) == -1,
              "replayed sequence rejected");

    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_aead(void)
{
    SNEPPXTransportSecurity ts;
    SX_ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    SX_ASSERT(sid >= 0, "new session");

    uint8_t plaintext[] = "aad protected message";
    uint8_t aad[] = "session-context";
    uint8_t ciphertext[64];
    uint8_t nonce[SNEPPX_TLS_NONCE_LEN];
    uint8_t tag[SNEPPX_TRANSPORT_AAD_TAG_LEN];
    uint8_t decrypted[64];
    size_t len = strlen((char*) plaintext) + 1;
    size_t aad_len = strlen((char*) aad) + 1;

    SX_ASSERT(SNEPPX_transport_encrypt_aead(&ts, sid, plaintext, len, aad, aad_len, ciphertext, nonce, tag) == 0,
              "encrypt aead");
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, ciphertext, len, aad, aad_len, nonce, tag, decrypted) == 0,
              "decrypt aead");
    SX_ASSERT(memcmp(plaintext, decrypted, len) == 0, "aead roundtrip");

    uint8_t bad_aad[sizeof(aad)];
    memcpy(bad_aad, aad, sizeof(aad));
    bad_aad[0] ^= 1;
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, ciphertext, len, bad_aad, aad_len, nonce, tag, decrypted) == -1,
              "tampered aad rejected");

    uint8_t bad_tag[SNEPPX_TRANSPORT_AAD_TAG_LEN];
    memcpy(bad_tag, tag, sizeof(tag));
    bad_tag[0] ^= 1;
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, ciphertext, len, aad, aad_len, nonce, bad_tag, decrypted) == -1,
              "tampered tag rejected");

    uint8_t bad_ciphertext[sizeof(ciphertext)];
    memcpy(bad_ciphertext, ciphertext, sizeof(ciphertext));
    bad_ciphertext[0] ^= 1;
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, bad_ciphertext, len, aad, aad_len, nonce, tag, decrypted) == -1,
              "tampered ciphertext rejected");

    uint8_t legacy_ciphertext[64];
    uint8_t legacy_nonce[SNEPPX_TLS_NONCE_LEN];
    SX_ASSERT(SNEPPX_transport_encrypt_with_aad(&ts, sid, plaintext, len, aad, aad_len, legacy_ciphertext,
                                                 legacy_nonce) == 0,
              "encrypt with aad");
    SX_ASSERT(SNEPPX_transport_decrypt_with_aad(&ts, sid, legacy_ciphertext, len, aad, aad_len, legacy_nonce,
                                                decrypted) == 0,
              "decrypt with aad");
    SX_ASSERT(memcmp(plaintext, decrypted, len) == 0, "aad roundtrip");

    uint8_t empty_ciphertext[64];
    uint8_t empty_nonce[SNEPPX_TLS_NONCE_LEN];
    SX_ASSERT(SNEPPX_transport_encrypt_aead(&ts, sid, plaintext, len, NULL, 0, empty_ciphertext, empty_nonce, tag) == 0,
              "null aad accepted when length is zero");
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, empty_ciphertext, len, NULL, 0, empty_nonce, tag, decrypted) == 0,
              "null aad decrypt accepted when length is zero");

    SX_ASSERT(SNEPPX_transport_encrypt_with_aad(&ts, sid, plaintext, len, NULL, 1, ciphertext, nonce) == -1,
              "null aad rejected when length is nonzero");
    SX_ASSERT(SNEPPX_transport_decrypt_with_aad(&ts, sid, ciphertext, len, NULL, 1, nonce, decrypted) == -1,
              "null aad decrypt rejected when length is nonzero");
    SX_ASSERT(SNEPPX_transport_encrypt_aead(&ts, sid, plaintext, len, NULL, 1, ciphertext, nonce, tag) == -1,
              "null aad aead rejected when length is nonzero");
    SX_ASSERT(SNEPPX_transport_decrypt_aead(&ts, sid, ciphertext, len, NULL, 1, nonce, tag, decrypted) == -1,
              "null aad aead decrypt rejected when length is nonzero");

    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_noise_handshake(void)
{
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    uint8_t msg[64];
    size_t msg_len = sizeof(msg);
    SX_ASSERT(SNEPPX_transport_noise_handshake(&ts, NULL, 0, msg, &msg_len) == 0, "noise handshake");
    SX_ASSERT(msg_len == 48, "handshake msg 48 bytes");
    SNEPPX_transport_shutdown(&ts);
}

static void test_identity_init(void)
{
    SNEPPXIdentityManager mgr;
    SX_ASSERT(SNEPPX_identity_init(&mgr) == 0, "identity init");
    SX_ASSERT(mgr.ddos_protection_enabled == 1, "ddos enabled");
    SNEPPX_identity_shutdown(&mgr);
}

static void test_identity_pin_verify(void)
{
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

static void test_ddos_protection(void)
{
    SNEPPXIdentityManager mgr;
    SNEPPX_identity_init(&mgr);
    mgr.ddos_request_limit = 5;
    for (int i = 0; i < 5; i++)
        SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "under limit");
    SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 1, "over limit");
    SNEPPX_identity_ddos_reset(&mgr);
    SX_ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "after reset");
    SNEPPX_identity_shutdown(&mgr);
}

TEST(test_s4_network, transport_init)
{
    test_transport_init();
}
TEST(test_s4_network, transport_session)
{
    test_transport_session();
}
TEST(test_s4_network, transport_encrypt_decrypt)
{
    test_transport_encrypt_decrypt();
}
TEST(test_s4_network, transport_rejects_xor_aes_mode)
{
    test_transport_rejects_xor_aes_mode();
}
TEST(test_s4_network, transport_sequence_tracking)
{
    test_transport_sequence_tracking();
}
TEST(test_s4_network, transport_aead)
{
    test_transport_aead();
}
TEST(test_s4_network, transport_noise_handshake)
{
    test_transport_noise_handshake();
}
TEST(test_s4_network, identity_init)
{
    test_identity_init();
}
TEST(test_s4_network, identity_pin_verify)
{
    test_identity_pin_verify();
}
TEST(test_s4_network, ddos_protection)
{
    test_ddos_protection();
}
