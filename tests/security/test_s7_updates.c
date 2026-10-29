#include "signed_update.h"
#include "test_gtest.h"
#include "s7_extensions.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test S7 Updates
 *
 * WHAT
 *   Test S7 Updates.
 *
 * CONCEPT
 *   Provides the Test S7 Updates.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_update_verifier_init(void) {
    SNEPPXUpdateVerifier uv;
    SX_ASSERT(SNEPPX_update_verifier_init(&uv) == 0, "init");
    SX_ASSERT(uv.rollback_protection_enabled == 1, "rollback protection");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_rollback_detect(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    uint32_t older[] = {0, 9, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, older) == 1, "rollback detected");
    uint32_t same[] = {1, 0, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, same) == 0, "same version ok");
    uint32_t newer[] = {2, 0, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, newer) == 0, "newer version ok");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_min_version(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SX_ASSERT(SNEPPX_update_verifier_set_min_version(&uv, 1, 5, 0) == 0, "set min version");
    uint32_t below_min[] = {1, 2, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, below_min) == 1, "below min rejected");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_apply_requires_trusted_key(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    memset(&update, 0, sizeof(update));
    update.version_major = 2;
    update.version_minor = 0;
    update.version_patch = 0;
    /* No trusted key configured -> fail closed, apply rejected. */
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, NULL, 0) == -1, "apply without trusted key rejected");
    SX_ASSERT(uv.current_version[0] == 1, "version not updated");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_apply_succeeds_with_trusted_key(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    memset(&update, 0, sizeof(update));
    update.version_major = 2;
    update.version_minor = 0;
    update.version_patch = 0;

    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    memset(seed, 0x01, sizeof(seed));
    uint8_t pubkey[SNEPPX_ED25519_PUBLIC_KEY_LEN];
    SX_ASSERT(SNEPPX_ed25519_publickey_from_seed(seed, pubkey) == 0, "derive pubkey");
    SX_ASSERT(SNEPPX_update_verifier_set_trusted_pubkey(&uv, pubkey) == 0, "set trusted pubkey");
    SX_ASSERT(SNEPPX_update_verifier_sign_update(&update, seed, sizeof(seed)) == 0, "sign update");

    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, NULL, 0) == 0, "apply signed update");
    SX_ASSERT(uv.current_version[0] == 2, "version updated");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_tuf_init(void) {
    SNEPPXTUFMetadata tuf;
    SX_ASSERT(SNEPPX_tuf_init(&tuf) == 0, "tuf init");
    SX_ASSERT(tuf.initialized == 1, "tuf initialized");
}

static void test_ab_partition(void) {
    SNEPPXABPartition ab;
    SX_ASSERT(SNEPPX_ab_partition_init(&ab) == 0, "ab init");
    SX_ASSERT(SNEPPX_ab_partition_mark_good(&ab, 0) == 0, "mark slot 0 good");
    SX_ASSERT(SNEPPX_ab_partition_swap(&ab) == 1, "swap ready");
}

static void test_canary_rollout(void) {
    SNEPPXCanaryRollout cr;
    SX_ASSERT(SNEPPX_canary_rollout_init(&cr, 100, 5) == 0, "canary init");
    SX_ASSERT(cr.total_nodes == 100, "total nodes");
    SX_ASSERT(cr.canary_nodes == 5, "canary nodes");
    SX_ASSERT(SNEPPX_canary_rollout_promote(&cr) == 0, "promote");
    SX_ASSERT(cr.promoted == 1, "promoted");
}

static void test_dep_resolver(void) {
    SNEPPXDepResolver dr;
    SX_ASSERT(SNEPPX_dep_resolver_init(&dr) == 0, "dep init");
    SX_ASSERT(SNEPPX_dep_resolver_add_dep(&dr, "libfoo", 1, 2, 3) == 0, "add dep");
    SX_ASSERT(SNEPPX_dep_resolver_resolve(&dr) == 0, "resolve");
    SX_ASSERT(dr.resolved == 1, "resolved");
}


TEST(test_s7_updates, update_verifier_init) { test_update_verifier_init(); }
TEST(test_s7_updates, update_verifier_rollback_detect) { test_update_verifier_rollback_detect(); }
TEST(test_s7_updates, update_verifier_min_version) { test_update_verifier_min_version(); }
TEST(test_s7_updates, signed_update_apply_requires_trusted_key) { test_signed_update_apply_requires_trusted_key(); }
TEST(test_s7_updates, signed_update_apply_succeeds_with_trusted_key) { test_signed_update_apply_succeeds_with_trusted_key(); }
TEST(test_s7_updates, tuf_init) { test_tuf_init(); }
TEST(test_s7_updates, ab_partition) { test_ab_partition(); }
TEST(test_s7_updates, canary_rollout) { test_canary_rollout(); }
TEST(test_s7_updates, dep_resolver) { test_dep_resolver(); }
