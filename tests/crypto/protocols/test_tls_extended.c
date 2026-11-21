#include "security/crypto/protocols/tls_extended.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Tls Extended
 *
 * WHAT
 *   Test Tls Extended.
 *
 * CONCEPT
 *   Provides the Test Tls Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_tls_session_init(void) {
    SNEPPXTlsSession session;
    memset(&session, 0, sizeof(session));
    int ret = snepx_tls_client_hello(&session);
    assert(ret == 0);
    snepx_tls_key_update(&session);
    snepx_tls_new_session_ticket(&session, 3600);
}

static void test_tls_derive(void) {
    SNEPPXTlsSession session;
    memset(&session, 0, sizeof(session));
    int ret = snepx_tls13_derive_early_secret(&session, NULL, 0);
    assert(ret == 0);
    ret = snepx_tls13_derive_handshake_secret(&session, NULL, 0);
    assert(ret == 0);
    ret = snepx_tls13_derive_master_secret(&session);
    assert(ret == 0);
}

int main(void) {
    test_tls_session_init();
    test_tls_derive();
    return 0;
}
